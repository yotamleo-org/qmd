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
// Runs between a search's decision to skip the reranker and the tool's answer: lets a row move the
// breaker's clock in that window. Everything else reaches the real store.
const afterSearch = vi.hoisted(() => ({ fn: null as null | (() => void) }));
// oxlint-disable-next-line anti-slop/no-module-mocking -- store.search has no seam between its decision and its return, HIMMEL-5938
vi.mock("../src/index.js", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../src/index.js")>();
  return {
    ...mod,
    createStore: async (...args: Parameters<typeof mod.createStore>) => {
      const store = await mod.createStore(...args);
      const realSearch = store.search.bind(store);
      store.search = (async (opts: Parameters<typeof realSearch>[0]) => {
        const found = await realSearch(opts);
        afterSearch.fn?.();
        return found;
      }) as typeof realSearch;
      return store;
    },
  };
});
import { createStore } from "../src/index.ts";
import { startMcpHttpServer, type HttpServerHandle } from "../src/mcp/server.ts";
import { _resetProductionModeForTesting, rerank } from "../src/store.ts";
import {
  isRerankBreakerOpen, markLlamaPoisonedForTests, resetLlamaPoisonedForTests, resetRerankBreakerForTests, tripRerankBreaker,
} from "../src/llm.ts";

type Mode = "ok" | "400" | "401" | "500" | "hang" | "empty";
const PROVIDER_DETAIL = "provider-detail-must-not-be-logged";
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
        if (mode === "empty") { res.writeHead(200, { "content-type": "application/json" }).end('{"results":[]}'); return; }
        if (mode !== "ok") { res.writeHead(Number(mode), { "content-type": "application/json" }).end(`{"error":"${PROVIDER_DETAIL}"}`); return; }
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
    afterSearch.fn = null;
    vi.restoreAllMocks();
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

  // Both halves of the store: typed searches reach structuredSearch, a plain query reaches hybridQuery.
  const shapes = { "typed searches": searchArgs, "a plain query": { query: "alpha" } };
  for (const [shape, args] of Object.entries(shapes)) {
    for (const status of ["401", "500"] as const) {
      test(`${shape}: a ${status} from the remote reranker answers unreranked with a note and a flag, not an error`, async () => {
        mode = status;
        const logged = vi.spyOn(console, "error").mockImplementation(() => {});
        await start("remote");
        const res = await rpc(handle.port, args);
        expect(res.result?.isError).toBeUndefined();
        expect(res.result?.structuredContent.results.length).toBe(2);
        expect(res.result?.structuredContent.results.every((r: any) => r.rerankTimedOut === true)).toBe(true);
        expect(res.result?.content[0].text).toMatch(/unreranked/);
        // The log line carries the status, never the provider's response text.
        const lines = logged.mock.calls.map(c => c.join(" "));
        expect(lines.some(l => l.includes(`HTTP ${status}`))).toBe(true);
        expect(lines.some(l => l.includes(PROVIDER_DETAIL))).toBe(false);
      });
    }

    test(`${shape}: an open breaker skips the reranker and the answer says so`, async () => {
      await start("remote");
      tripRerankBreaker();
      const res = await rpc(handle.port, args);
      expect(res.result?.isError).toBeUndefined();
      expect(rerankCalls).toBe(0);
      expect(res.result?.structuredContent.rerankSkipped).toBe(true);
      expect(res.result?.content[0].text).toMatch(/unreranked/);
    });
  }

  test("a breaker that expires while the search runs: the answer still says the search skipped the reranker", async () => {
    await start("remote");
    tripRerankBreaker();
    afterSearch.fn = resetRerankBreakerForTests; // the breaker closes after the search decided to skip
    const res = await rpc(handle.port, searchArgs);
    expect(rerankCalls).toBe(0);
    expect(res.result?.structuredContent.rerankSkipped).toBe(true);
    expect(res.result?.content[0].text).toMatch(/unreranked/);
  });

  test("a breaker that opens while the search runs does not call a reranked answer unreranked", async () => {
    await start("remote");
    afterSearch.fn = tripRerankBreaker; // opens after the search already reranked
    const res = await rpc(handle.port, searchArgs);
    expect(rerankCalls).toBe(1);
    expect(res.result?.structuredContent.rerankSkipped).toBeUndefined();
    expect(res.result?.content[0].text).not.toMatch(/unreranked/);
  });

  test("a 400 answers one request and does not open the breaker; a 500 does", async () => {
    mode = "400";
    vi.spyOn(console, "error").mockImplementation(() => {});
    await start("remote");
    const first = await rpc(handle.port, searchArgs);
    expect(first.result?.content[0].text).toMatch(/unreranked/);
    expect(isRerankBreakerOpen()).toBe(false);
    await rpc(handle.port, searchArgs);
    expect(rerankCalls).toBe(2); // asked again
    mode = "500";
    await rpc(handle.port, searchArgs);
    expect(isRerankBreakerOpen()).toBe(true);
  });

  test("a remote 200 with no results is flagged unreranked, not read as reranked", async () => {
    mode = "empty";
    vi.spyOn(console, "error").mockImplementation(() => {});
    await start("remote");
    const res = await rpc(handle.port, searchArgs);
    expect(res.result?.structuredContent.results.every((r: any) => r.rerankTimedOut === true)).toBe(true);
    expect(res.result?.content[0].text).toMatch(/unreranked/);
  });

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

  test("the budget rerank() starts is 15 s for a remote reranker and 60 s for a local one, with no override", async () => {
    delete process.env.QMD_RERANK_TIMEOUT_MS;
    const dir2 = mkdtempSync(join(tmpdir(), "qmd-rerank-budget-"));
    const store = await createStore({ dbPath: join(dir2, "i.sqlite"), config: { collections: {} } });
    try {
      const docs = [{ file: "a", text: "alpha" }];
      const budgetFor = async (model: string) => {
        const timers = vi.spyOn(globalThis, "setTimeout");
        const llm = { rerankModelName: model, rerank: async () => ({ results: [{ file: "a", score: 0.9, index: 0 }], model }) };
        await rerank(`q-${model}`, docs, model, store.internal.db, undefined, llm as never);
        const delays = timers.mock.calls.map(c => c[1]);
        timers.mockRestore();
        return delays;
      };
      expect(await budgetFor("openai:remote-model")).toContain(15_000);
      expect(await budgetFor("hf:local/model")).toContain(60_000);
    } finally {
      await store.close();
      rmSync(dir2, { recursive: true, force: true });
    }
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

describe("the rerank breaker's lifetime", () => {
  const saved = process.env.QMD_RERANK_BREAKER_MS;
  afterEach(() => {
    resetRerankBreakerForTests();
    if (saved !== undefined) process.env.QMD_RERANK_BREAKER_MS = saved; else delete process.env.QMD_RERANK_BREAKER_MS;
  });

  test("it closes by itself once QMD_RERANK_BREAKER_MS has passed", async () => {
    process.env.QMD_RERANK_BREAKER_MS = "60";
    tripRerankBreaker();
    expect(isRerankBreakerOpen()).toBe(true);
    await new Promise(r => setTimeout(r, 120));
    expect(isRerankBreakerOpen()).toBe(false);
  });

  test("unset or empty QMD_RERANK_BREAKER_MS means the 60 s default, only an explicit 0 turns it off", () => {
    for (const value of [undefined, "", "  "]) {
      resetRerankBreakerForTests();
      if (value === undefined) delete process.env.QMD_RERANK_BREAKER_MS; else process.env.QMD_RERANK_BREAKER_MS = value;
      tripRerankBreaker();
      expect(isRerankBreakerOpen()).toBe(true);
    }
    resetRerankBreakerForTests();
    process.env.QMD_RERANK_BREAKER_MS = "0";
    tripRerankBreaker();
    expect(isRerankBreakerOpen()).toBe(false);
  });
});
