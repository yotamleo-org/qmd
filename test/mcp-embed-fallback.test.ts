/**
 * HIMMEL-8764: typed searches answer inline, so a hung remote embedder must not hold one for the
 * 120 s request timeout. Real store, real indexed and embedded collection, a real HTTP stub for the
 * remote /embeddings endpoint that accepts the request and never answers (store.search is not stubbed).
 * Each test embeds with its own model id, so the embed breaker (keyed by endpoint and model) never carries across tests.
 */
import { describe, test, expect, afterEach, beforeEach, vi } from "vitest";
import { createServer, type Server } from "node:http";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore } from "../src/index.ts";
import { startMcpHttpServer, type HttpServerHandle } from "../src/mcp/server.ts";
import { _resetProductionModeForTesting } from "../src/store.ts";

type Rpc = { result?: Record<string, any> };

// Put the environment back key by key: assigning a new object detaches it from child processes under Bun.
function restoreEnv(snapshot: NodeJS.ProcessEnv) {
  for (const key of Object.keys(process.env)) if (!(key in snapshot)) delete process.env[key];
  Object.assign(process.env, snapshot);
}

async function rpc(port: number, args: Record<string, unknown>): Promise<Rpc> {
  const res = await fetch(`http://localhost:${port}/mcp`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json", Accept: "application/json, text/event-stream",
      "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call", "Mcp-Name": "query",
    },
    body: JSON.stringify({
      jsonrpc: "2.0", id: 1, method: "tools/call",
      params: {
        name: "query", arguments: args,
        _meta: {
          "io.modelcontextprotocol/protocolVersion": "2026-07-28",
          "io.modelcontextprotocol/clientInfo": { name: "embed-test", version: "1.0.0" },
          "io.modelcontextprotocol/clientCapabilities": {},
        },
      },
    }),
  });
  return (await res.json()) as Rpc;
}

const NOTE = /meaning-based part of the search was skipped/;
const vecSearch = { searches: [{ type: "lex", query: "alpha" }, { type: "vec", query: "alpha things" }], rerank: false };
const vecOnly = { searches: [{ type: "vec", query: "alpha things" }], rerank: false };
const hydeOnly = { searches: [{ type: "hyde", query: "a note about alpha" }], rerank: false };
const hydeSearch = { searches: [{ type: "lex", query: "alpha" }, { type: "hyde", query: "a note about alpha" }], rerank: false };

describe("query tool: a hung remote embed never holds a search", () => {
  let dir: string;
  let stub: Server;
  let mode: "ok" | "hang" | "429" | "first-429";
  let embedCalls: number;
  let modelCount = 0;
  let handle: HttpServerHandle;
  const saved = { ...process.env };

  // The index is embedded while the stub still answers; the tests then flip it to hang.
  async function start() {
    const base = `http://127.0.0.1:${(stub.address() as { port: number }).port}/v1`;
    const docs = join(dir, "docs");
    mkdirSync(docs, { recursive: true });
    writeFileSync(join(docs, "a.md"), "# Alpha\n\nalpha bravo charlie\n");
    writeFileSync(join(docs, "b.md"), "# Beta\n\nalpha delta echo\n");
    const model = `openai:stub-embed-${++modelCount}`;
    writeFileSync(join(dir, "index.yml"),
      `collections:\n  notes:\n    path: ${docs}\n    pattern: "**/*.md"\nmodels:\n  embed: ${model}\n  generate: openai:stub-gen\n  openai_base_url: ${base}\n`);
    process.env.INDEX_PATH = join(dir, "index.sqlite");
    process.env.QMD_CONFIG_DIR = dir;
    process.env.QMD_OPENAI_API_KEY = "stub-key";
    const store = await createStore({ dbPath: join(dir, "index.sqlite"), configPath: join(dir, "index.yml") });
    await store.update();
    await store.embed();
    await store.close();
    embedCalls = 0; // count only what the searches ask
    mode = "hang";
    handle = await startMcpHttpServer(0, { quiet: true, dbPath: join(dir, "index.sqlite") });
  }

  beforeEach(async () => {
    dir = mkdtempSync(join(tmpdir(), "qmd-embed-fb-"));
    mode = "ok";
    embedCalls = 0;
    stub = createServer((req, res) => {
      if (!req.url?.endsWith("/embeddings")) { res.writeHead(404).end(); return; }
      embedCalls++;
      let body = "";
      req.on("data", c => { body += c; });
      req.on("end", () => {
        if (mode === "hang") return; // accepts the request, never answers
        if (mode === "429" || (mode === "first-429" && embedCalls === 1)) { res.writeHead(429, { "content-type": "application/json" }).end('{"error":"slow down"}'); return; }
        const n = (JSON.parse(body).input as unknown[]).length;
        const data = Array.from({ length: n }, (_, index) => ({ index, embedding: [0.1, 0.2, 0.3, 0.4] }));
        // first-429: the other worker answers a moment later, after the call has already failed.
        setTimeout(() => res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ data })), mode === "first-429" ? 150 : 0);
      });
    });
    await new Promise<void>(r => stub.listen(0, "127.0.0.1", r));
    process.env.QMD_EMBED_TIMEOUT_MS = "300";
    process.env.QMD_EMBED_BREAKER_MS = "60000";
  });
  afterEach(async () => {
    await handle?.stop();
    stub.closeAllConnections();
    await new Promise<void>(r => stub.close(() => r()));
    _resetProductionModeForTesting();
    vi.restoreAllMocks();
    restoreEnv(saved);
    rmSync(dir, { recursive: true, force: true });
  });

  test("a typed vec search answers inside the budget with the keyword results and the note", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await start();
    const t0 = Date.now();
    const res = await rpc(handle.port, vecSearch);
    expect(Date.now() - t0).toBeLessThan(5_000);
    expect(res.result?.isError).toBeUndefined();
    expect(res.result?.structuredContent.results.length).toBe(2);
    expect(res.result?.structuredContent.embedSkipped).toBe(true);
    expect(res.result?.content[0].text).toMatch(NOTE);
    expect(embedCalls).toBe(1); // one attempt, no retry
  }, 20_000);

  test("while the breaker is open a second search answers at once with the note and never calls the endpoint", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await start();
    await rpc(handle.port, vecSearch);
    expect(embedCalls).toBe(1);

    const t0 = Date.now();
    const second = await rpc(handle.port, vecSearch);
    expect(Date.now() - t0).toBeLessThan(1_500); // the request count below is the proof; this only rules out a wait
    expect(embedCalls).toBe(1);
    expect(second.result?.structuredContent.results.length).toBe(2);
    expect(second.result?.content[0].text).toMatch(NOTE);
  }, 20_000);

  test("once the breaker window has passed the endpoint is asked again", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    process.env.QMD_EMBED_BREAKER_MS = "400";
    await start();
    await rpc(handle.port, vecSearch);
    expect(embedCalls).toBe(1);
    await new Promise(r => setTimeout(r, 450));
    await rpc(handle.port, vecSearch);
    expect(embedCalls).toBe(2);
  }, 20_000);

  test("a hyde search falls back the same way", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await start();
    const t0 = Date.now();
    const res = await rpc(handle.port, hydeSearch);
    expect(Date.now() - t0).toBeLessThan(5_000);
    expect(res.result?.structuredContent.results.length).toBe(2);
    expect(res.result?.content[0].text).toMatch(NOTE);
  }, 20_000);

  // The answer must never be an empty success: with no row to carry a flag, the text and structuredContent say it.
  for (const [kind, args] of [["vec-only", vecOnly], ["hyde-only", hydeOnly]] as const) {
    test(`a ${kind} search with the embed skipped says so instead of answering an empty success`, async () => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      await start();
      const res = await rpc(handle.port, args);
      expect(res.result?.isError).toBeUndefined();
      expect(res.result?.structuredContent.results.length).toBe(0);
      expect(res.result?.structuredContent.embedSkipped).toBe(true);
      expect(res.result?.content[0].text).toMatch(NOTE);
      expect(res.result?.content[0].text).toMatch(/not a confirmed miss/);
      expect(res.result?.content[0].text).toMatch(/^No results returned/); // no keyword search ran, so it never says one did
      const second = await rpc(handle.port, args); // breaker open: the same answer, no request
      expect(embedCalls).toBe(1);
      expect(second.result?.structuredContent.embedSkipped).toBe(true);
      expect(second.result?.content[0].text).toMatch(NOTE);
    }, 20_000);
  }

  test("a search whose rows a minScore emptied still carries the flag and the note", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await start();
    const res = await rpc(handle.port, { ...vecSearch, minScore: 1.5 }); // above any blended score: every keyword row is filtered out
    expect(res.result?.structuredContent.results.length).toBe(0);
    expect(res.result?.structuredContent.embedSkipped).toBe(true);
    expect(res.result?.content[0].text).toMatch(NOTE);
  }, 20_000);

  test("a plain query falls back the same way", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await start();
    const res = await rpc(handle.port, { query: "alpha", rerank: false });
    expect(res.result?.isError).toBeUndefined();
    expect(res.result?.structuredContent.results.length).toBe(2);
    expect(res.result?.structuredContent.embedSkipped).toBe(true);
    expect(res.result?.content[0].text).toMatch(NOTE);
  }, 30_000);

  test("a 429 on the query embed takes the same fallback as a hang and opens the breaker", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await start();
    mode = "429";
    const res = await rpc(handle.port, vecSearch);
    expect(res.result?.isError).toBeUndefined();
    expect(res.result?.structuredContent.results.length).toBe(2);
    expect(res.result?.content[0].text).toMatch(NOTE);
    expect(embedCalls).toBe(1); // no retry
    await rpc(handle.port, vecSearch);
    expect(embedCalls).toBe(1); // breaker open
  }, 20_000);

  test("once one query batch has failed, the workers still running send no further request", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    process.env.QMD_OPENAI_EMBED_BATCH_SIZE = "1";
    process.env.QMD_OPENAI_EMBED_CONCURRENCY = "2";
    await start();
    mode = "first-429";
    const searches = [{ type: "lex", query: "alpha" }, ...["one", "two", "three", "four"].map(w => ({ type: "vec", query: `alpha ${w}` }))];
    const res = await rpc(handle.port, { searches, rerank: false });
    expect(res.result?.isError).toBeUndefined();
    expect(res.result?.content[0].text).toMatch(NOTE);
    await new Promise(r => setTimeout(r, 400)); // let the slow worker finish its batch and look for another
    expect(embedCalls).toBe(2); // the two first batches; the surviving worker claimed no third one
  }, 20_000);

  test("REST /query carries embedSkipped outside the rows, also when no row comes back", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await start();
    const post = async (body: unknown) => (await (await fetch(`http://localhost:${handle.port}/query`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    })).json()) as { results: Array<Record<string, unknown>>; embedSkipped?: true };
    const withRows = await post(vecSearch);
    expect(withRows.results.length).toBe(2);
    expect(withRows.embedSkipped).toBe(true);
    const empty = await post(vecOnly);
    expect(empty.results.length).toBe(0);
    expect(empty.embedSkipped).toBe(true);
  }, 20_000);
});
