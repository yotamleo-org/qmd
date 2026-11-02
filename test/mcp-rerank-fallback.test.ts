/**
 * HIMMEL-8747: the query tool reranks by default, so a rerank that fails must never fail the
 * search. Real store, real indexed collection, a real HTTP stub for the remote /rerank
 * endpoint (store.search is not stubbed).
 */
import { describe, test, expect, afterEach, beforeEach, vi } from "vitest";
import { createServer, type Server } from "node:http";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// llm.ts reads the rerank-hung marker once at import: point it at an empty dir first, so a real
// marker under the user's cache neither skips the reranker here nor is written by these tests.
vi.hoisted(() => {
  const { mkdtempSync } = require("node:fs") as typeof import("node:fs");
  const { tmpdir } = require("node:os") as typeof import("node:os");
  const { join } = require("node:path") as typeof import("node:path");
  process.env.XDG_CACHE_HOME = mkdtempSync(join(tmpdir(), "qmd-rerank-fb-cache-"));
});
import { createStore } from "../src/index.ts";
import { startMcpHttpServer, type HttpServerHandle } from "../src/mcp/server.ts";
import { _resetProductionModeForTesting, rerank } from "../src/store.ts";
import { markLlamaPoisonedForTests, resetLlamaPoisonedForTests, resetRerankBreakerForTests } from "../src/llm.ts";

type Mode = "ok" | "401" | "500" | "hang";
type Rpc = { result?: Record<string, any> };

const searchArgs = { searches: [{ type: "lex", query: "alpha" }] };

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
          "io.modelcontextprotocol/clientInfo": { name: "rerank-test", version: "1.0.0" },
          "io.modelcontextprotocol/clientCapabilities": {},
        },
      },
    }),
  });
  return (await res.json()) as Rpc;
}

describe("query tool: a failing rerank never fails the search", () => {
  let dir: string;
  let stub: Server;
  let mode: Mode;
  let rerankCalls: number;
  let handle: HttpServerHandle;
  const saved = { ...process.env };

  // rerankModel "remote" points models.rerank at the stub; "local" leaves the default hf: model.
  async function start(rerankModel: "remote" | "local") {
    const base = `http://127.0.0.1:${(stub.address() as { port: number }).port}/v1`;
    const docs = join(dir, "docs");
    mkdirSync(docs, { recursive: true });
    writeFileSync(join(docs, "a.md"), "# Alpha\n\nalpha bravo charlie\n");
    writeFileSync(join(docs, "b.md"), "# Beta\n\nalpha delta echo\n");
    // Every role remote: a poisoned process then only logs, it never records a hang or kills the runner.
    const models = rerankModel === "remote"
      ? `models:\n  embed: openai:stub-embed\n  generate: openai:stub-gen\n  rerank: openai:stub-rerank\n  openai_base_url: ${base}\n`
      : "";
    writeFileSync(join(dir, "index.yml"), `collections:\n  notes:\n    path: ${docs}\n    pattern: "**/*.md"\n${models}`);
    process.env.INDEX_PATH = join(dir, "index.sqlite");
    process.env.QMD_CONFIG_DIR = dir;
    process.env.QMD_OPENAI_API_KEY = "stub-key";
    const store = await createStore({ dbPath: join(dir, "index.sqlite"), configPath: join(dir, "index.yml") });
    await store.update();
    await store.close();
    handle = await startMcpHttpServer(0, { quiet: true, dbPath: join(dir, "index.sqlite") });
  }

  beforeEach(async () => {
    dir = mkdtempSync(join(tmpdir(), "qmd-rerank-fb-"));
    mode = "ok";
    rerankCalls = 0;
    resetRerankBreakerForTests();
    stub = createServer((req, res) => {
      if (!req.url?.endsWith("/rerank")) { res.writeHead(404).end(); return; }
      rerankCalls++;
      let body = "";
      req.on("data", c => { body += c; });
      req.on("end", () => {
        if (mode === "hang") return; // never answers
        if (mode !== "ok") { res.writeHead(Number(mode), { "content-type": "application/json" }).end('{"error":"nope"}'); return; }
        const n = (JSON.parse(body).documents as unknown[]).length;
        const results = Array.from({ length: n }, (_, i) => ({ index: i, relevance_score: 1 - i / 10 }));
        res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ results }));
      });
    });
    await new Promise<void>(r => stub.listen(0, "127.0.0.1", r));
  });
  afterEach(async () => {
    await handle?.stop();
    stub.closeAllConnections();
    await new Promise<void>(r => stub.close(() => r()));
    _resetProductionModeForTesting();
    resetLlamaPoisonedForTests();
    resetRerankBreakerForTests();
    process.env = { ...saved };
    rmSync(dir, { recursive: true, force: true });
  });

  test("a healthy remote reranker is used by default", async () => {
    await start("remote");
    const res = await rpc(handle.port, searchArgs);
    expect(res.result?.isError).toBeUndefined();
    expect(res.result?.structuredContent.results.length).toBe(2);
    expect(rerankCalls).toBe(1);
    expect(res.result?.content[0].text).not.toMatch(/unreranked/);
  });

  for (const status of ["401", "500"] as const) {
    test(`a ${status} from the remote reranker answers unreranked with a note and a flag, not an error`, async () => {
      mode = status;
      await start("remote");
      const res = await rpc(handle.port, searchArgs);
      expect(res.result?.isError).toBeUndefined();
      expect(res.result?.structuredContent.results.length).toBe(2);
      expect(res.result?.structuredContent.results.every((r: any) => r.rerankTimedOut === true)).toBe(true);
      expect(res.result?.content[0].text).toMatch(/unreranked/);
    });
  }

  test("a hung remote reranker waits its short budget once, then the breaker skips it for later searches", async () => {
    mode = "hang";
    process.env.QMD_RERANK_TIMEOUT_MS = "300";
    await start("remote");
    const first = await rpc(handle.port, searchArgs);
    expect(first.result?.isError).toBeUndefined();
    expect(first.result?.content[0].text).toMatch(/unreranked/);
    expect(rerankCalls).toBe(1);

    const t0 = Date.now();
    const second = await rpc(handle.port, searchArgs);
    expect(Date.now() - t0).toBeLessThan(250);
    expect(rerankCalls).toBe(1); // the breaker kept it from asking again
    expect(second.result?.structuredContent.rerankSkipped).toBe(true);
    expect(second.result?.content[0].text).toMatch(/unreranked/);
  });

  test("a remote reranker gets a 15 s default budget, not the native 60 s", async () => {
    const { resolveRerankTimeoutMs } = await import("../src/llm.ts");
    delete process.env.QMD_RERANK_TIMEOUT_MS;
    expect(resolveRerankTimeoutMs(true)).toBe(15_000);
    expect(resolveRerankTimeoutMs(false)).toBe(60_000);
  });

  test("a local reranker is not asked by default: no attempt, no flag", async () => {
    await start("local");
    const res = await rpc(handle.port, searchArgs);
    expect(res.result?.isError).toBeUndefined();
    expect(res.result?.structuredContent.results.length).toBe(2);
    expect(res.result?.structuredContent.results.some((r: any) => r.rerankTimedOut)).toBe(false);
    expect(res.result?.structuredContent.rerankSkipped).toBeUndefined();
  });

  test("a process poisoned by an earlier hang says the results are unreranked", async () => {
    await start("remote");
    markLlamaPoisonedForTests();
    const res = await rpc(handle.port, searchArgs);
    expect(res.result?.structuredContent.rerankSkipped).toBe(true);
    expect(res.result?.content[0].text).toMatch(/unreranked/);
    expect(rerankCalls).toBe(0);
  });
});

describe("rerank(): a local reranker with no context scores nothing", () => {
  test("the 0.5 placeholder scores are refused and never cached", async () => {
    const dir = mkdtempSync(join(tmpdir(), "qmd-rerank-nocache-"));
    const store = await createStore({ dbPath: join(dir, "i.sqlite"), config: { collections: {} } });
    try {
      const docs = [{ file: "a", text: "alpha" }, { file: "b", text: "beta" }];
      const llm = {
        rerankModelName: "hf:local/no-context",
        rerank: async () => ({ results: docs.map((d, index) => ({ ...d, score: 0.5, index })), model: "fallback" }),
      };
      await expect(rerank("q", docs, llm.rerankModelName, store.internal.db, undefined, llm as never)).rejects.toThrow(/no usable context/);
      const rows = store.internal.db.prepare("SELECT COUNT(*) AS c FROM llm_cache").get() as { c: number };
      expect(rows.c).toBe(0);
    } finally {
      await store.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
