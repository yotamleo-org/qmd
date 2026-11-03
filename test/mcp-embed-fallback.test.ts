/**
 * HIMMEL-8764: typed searches answer inline, so a hung remote embedder must not hold one for the
 * 120 s request timeout. Real store, real indexed and embedded collection, a real HTTP stub for the
 * remote /embeddings endpoint that accepts the request and never answers (store.search is not stubbed).
 * Each test embeds with its own model id, so the embed breaker (keyed by endpoint and model) never carries across tests.
 */
import { describe, test, expect, afterEach, beforeEach, vi } from "vitest";
import { spawn } from "node:child_process";
import { createServer, type Server, type ServerResponse } from "node:http";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createStore } from "../src/index.ts";
import { isBun } from "../src/db.ts";
import { OpenAiLLM, resetEmbedBreakerForTests } from "../src/llm.ts";
import { startMcpHttpServer, type HttpServerHandle } from "../src/mcp/server.ts";
import { _resetProductionModeForTesting } from "../src/store.ts";

type Rpc = { result?: Record<string, any> };

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
const cliArgs = isBun ? [qmdScript] : [join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs"), qmdScript];

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
  let chatCalls: number;
  let expansion: boolean; // /chat/completions answers a vec+hyde expansion instead of 404
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
    chatCalls = 0;
    expansion = false;
    stub = createServer((req, res) => {
      if (req.url?.endsWith("/chat/completions")) {
        chatCalls++;
        if (!expansion) { res.writeHead(404).end(); return; }
        const content = "vec: alpha things\nhyde: a note about alpha";
        res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ choices: [{ message: { content } }] }));
        return;
      }
      if (!req.url?.endsWith("/embeddings")) { res.writeHead(404).end(); return; }
      const callNo = ++embedCalls; // the index at arrival: a later request cannot change which one was first
      let body = "";
      req.on("data", c => { body += c; });
      req.on("end", () => {
        if (mode === "hang") return; // accepts the request, never answers
        if (mode === "429" || (mode === "first-429" && callNo === 1)) { res.writeHead(429, { "content-type": "application/json" }).end('{"error":"slow down"}'); return; }
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

  test("store.searchVector raises on a skipped embed, also while the breaker is open, instead of answering empty", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await start();
    const store = await createStore({ dbPath: join(dir, "index.sqlite"), configPath: join(dir, "index.yml") });
    try {
      await expect(store.searchVector("alpha things")).rejects.toThrow(/embedding service/);
      expect(embedCalls).toBe(1);
      await expect(store.searchVector("alpha things")).rejects.toThrow(/embedding service/); // breaker open: still an error
      expect(embedCalls).toBe(1);
    } finally {
      await store.close();
    }
  }, 20_000);

  test("qmd vsearch exits non-zero with the error and never prints No results found when the embed is skipped", async () => {
    await start();
    const proc = spawn(process.execPath, [...cliArgs, "vsearch", "alpha things"], {
      cwd: dir, env: { ...process.env, PWD: dir, QMD_DOCTOR_DEVICE_PROBE: "0" }, stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (c: Buffer) => { stdout += c.toString(); });
    proc.stderr.on("data", (c: Buffer) => { stderr += c.toString(); });
    const exitCode = await new Promise<number>((resolve, reject) => { proc.once("error", reject); proc.once("close", code => resolve(code ?? 1)); });
    expect(exitCode).not.toBe(0);
    expect(stdout).not.toMatch(/No results found/);
    expect(stdout).not.toMatch(/^\s*\[\s*\]\s*$/);
    expect(stderr).toMatch(/embedding service/);
  }, 60_000);

  test("a good cached expansion survives an embed outage", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await start();
    expansion = true; // vec + hyde only: with the embed skipped, none of its lists runs
    await rpc(handle.port, { query: "alpha", rerank: false });
    expect(chatCalls).toBe(1);
    await rpc(handle.port, { query: "alpha", rerank: false });
    expect(chatCalls).toBe(1); // served from the cache: the outage did not drop it
  }, 30_000);

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

describe("the embed breaker's key, and what a failed query batch leaves", () => {
  const servers: Server[] = [];
  const saved = { ...process.env };
  type Handler = (callNo: number, body: { model: string; input: string[] }, res: ServerResponse) => void;

  async function serve(handler: Handler) {
    const state = { calls: 0 };
    const server = createServer((req, res) => {
      const callNo = ++state.calls;
      let raw = "";
      req.on("data", c => { raw += c; });
      req.on("end", () => handler(callNo, JSON.parse(raw), res));
    });
    await new Promise<void>(r => server.listen(0, "127.0.0.1", r));
    servers.push(server);
    return { base: `http://127.0.0.1:${(server.address() as { port: number }).port}/v1`, state };
  }
  const answer = (res: ServerResponse, input: string[]) =>
    res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ data: input.map((_, index) => ({ index, embedding: [0.1, 0.2] })) }));
  const llmFor = (baseUrl: string, model: string) => new OpenAiLLM({ baseUrl, embedModel: `openai:${model}`, roles: ["embed"] });

  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    process.env.QMD_EMBED_TIMEOUT_MS = "200";
    process.env.QMD_EMBED_BREAKER_MS = "60000";
    resetEmbedBreakerForTests();
  });
  afterEach(async () => {
    for (const server of servers.splice(0)) {
      server.closeAllConnections();
      await new Promise<void>(r => server.close(() => r()));
    }
    resetEmbedBreakerForTests();
    vi.restoreAllMocks();
    restoreEnv(saved);
  });

  test("a hang at one endpoint leaves the same model at another endpoint asked", async () => {
    const hung = await serve(() => {}); // never answers
    const healthy = await serve((_, body, res) => answer(res, body.input));
    expect((await llmFor(hung.base, "shared").embedBatch(["q"], { isQuery: true }))[0]).toBeNull();
    const fromHealthy = await llmFor(healthy.base, "shared").embedBatch(["q"], { isQuery: true });
    expect(fromHealthy[0]).not.toBeNull();
    expect(healthy.state.calls).toBe(1);
  });

  test("a hang for one model leaves another model at the same endpoint asked", async () => {
    const mixed = await serve((_, body, res) => { if (body.model !== "m-hang") answer(res, body.input); });
    expect((await llmFor(mixed.base, "m-hang").embedBatch(["q"], { isQuery: true }))[0]).toBeNull();
    const fromOk = await llmFor(mixed.base, "m-ok").embedBatch(["q"], { isQuery: true });
    expect(fromOk[0]).not.toBeNull();
    expect(mixed.state.calls).toBe(2);
  });

  test("vectors gathered before a query batch failed are kept", async () => {
    process.env.QMD_OPENAI_EMBED_BATCH_SIZE = "1";
    process.env.QMD_OPENAI_EMBED_CONCURRENCY = "1";
    const flaky = await serve((callNo, body, res) => {
      if (callNo === 2) res.writeHead(429, { "content-type": "application/json" }).end('{"error":"slow down"}');
      else answer(res, body.input);
    });
    const [first, second] = await llmFor(flaky.base, "gathered").embedBatch(["a", "b"], { isQuery: true });
    expect(first).not.toBeNull();
    expect(second).toBeNull();
    expect(flaky.state.calls).toBe(2);
  });
});
