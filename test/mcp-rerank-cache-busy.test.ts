/**
 * HIMMEL-8848: after a rerank the daemon caches the scores in SQLite on its one thread. While
 * another process (qmd update, qmd embed) holds the write lock, those writes must not hold the
 * search, or the searches queued behind it, for the 120 s busy_timeout. Real store, real indexed
 * collection, a real HTTP stub for the remote /rerank endpoint, and a second process that holds
 * the write lock with BEGIN IMMEDIATE for longer than the search may wait.
 */
import { describe, test, expect, afterAll, afterEach, beforeEach, vi } from "vitest";
import { spawn, type ChildProcess } from "node:child_process";
import { createServer, type Server } from "node:http";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// llm.ts reads the rerank-hung marker once at import: point it at an empty dir first (as mcp-rerank-fallback does).
const originalCacheHome = vi.hoisted(() => {
  const { mkdtempSync } = require("node:fs") as typeof import("node:fs");
  const { tmpdir } = require("node:os") as typeof import("node:os");
  const { join } = require("node:path") as typeof import("node:path");
  const original = process.env.XDG_CACHE_HOME;
  process.env.XDG_CACHE_HOME = mkdtempSync(join(tmpdir(), "qmd-rerank-busy-cache-"));
  return original;
});
afterAll(() => {
  if (originalCacheHome !== undefined) process.env.XDG_CACHE_HOME = originalCacheHome; else delete process.env.XDG_CACHE_HOME;
});

import { createStore } from "../src/index.ts";
import { isBun, openDatabase } from "../src/db.ts";
import { startMcpHttpServer, type HttpServerHandle } from "../src/mcp/server.ts";
import { _resetProductionModeForTesting } from "../src/store.ts";
import { resetRerankBreakerForTests } from "../src/llm.ts";

type Rpc = { result?: Record<string, any> };

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const HOLD_MS = 6_000;
const BUDGET_MS = 2_000; // what a search may take here; the hold outlasts it threefold

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
          "io.modelcontextprotocol/clientInfo": { name: "rerank-busy-test", version: "1.0.0" },
          "io.modelcontextprotocol/clientCapabilities": {},
        },
      },
    }),
  });
  return (await res.json()) as Rpc;
}

async function timed(port: number, args: Record<string, unknown>): Promise<{ ms: number; res: Rpc }> {
  const t0 = Date.now();
  const res = await rpc(port, args);
  return { ms: Date.now() - t0, res };
}

const alpha = { searches: [{ type: "lex", query: "alpha" }] };
const bravo = { searches: [{ type: "lex", query: "bravo" }] };

describe("query tool: a write lock held by another process never holds a reranked search", () => {
  let dir: string;
  let stub: Server;
  let rerankCalls: number;
  let handle: HttpServerHandle;
  let holder: ChildProcess | undefined;
  const saved = { ...process.env };

  async function start() {
    const base = `http://127.0.0.1:${(stub.address() as { port: number }).port}/v1`;
    const docs = join(dir, "docs");
    mkdirSync(docs, { recursive: true });
    writeFileSync(join(docs, "a.md"), "# Alpha\n\nalpha bravo charlie\n");
    writeFileSync(join(docs, "b.md"), "# Beta\n\nalpha delta echo\n");
    writeFileSync(join(dir, "index.yml"),
      `collections:\n  notes:\n    path: ${docs}\n    pattern: "**/*.md"\nmodels:\n  embed: openai:stub-embed\n  generate: openai:stub-gen\n  rerank: openai:stub-rerank\n  openai_base_url: ${base}\n`);
    process.env.INDEX_PATH = join(dir, "index.sqlite");
    process.env.QMD_CONFIG_DIR = dir;
    process.env.QMD_OPENAI_API_KEY = "stub-key";
    const store = await createStore({ dbPath: join(dir, "index.sqlite"), configPath: join(dir, "index.yml") });
    await store.update();
    await store.close();
    handle = await startMcpHttpServer(0, { quiet: true, dbPath: join(dir, "index.sqlite") });
  }

  // A second process takes the write lock and keeps it for HOLD_MS; resolves once it holds it.
  async function holdWriteLock(): Promise<void> {
    const script = join(dir, "hold.mts"); // .mts: the temp dir has no package.json saying ESM
    writeFileSync(script, [
      `import { openDatabase } from ${JSON.stringify(join(projectRoot, "src", "db.ts"))};`,
      `const db = openDatabase(${JSON.stringify(join(dir, "index.sqlite"))});`,
      `db.exec("BEGIN IMMEDIATE");`,
      `process.stdout.write("held\\n");`,
      `setTimeout(() => { db.exec("ROLLBACK"); db.close(); process.exit(0); }, ${HOLD_MS});`,
    ].join("\n"));
    const args = isBun ? [script] : [join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs"), script];
    holder = spawn(process.execPath, args, { stdio: ["ignore", "pipe", "inherit"] });
    await new Promise<void>((resolve, reject) => {
      holder!.stdout!.on("data", (d: Buffer) => { if (d.toString().includes("held")) resolve(); });
      holder!.on("exit", code => reject(new Error(`lock holder exited early (${code})`)));
    });
  }

  function rerankCacheRows(): number {
    const db = openDatabase(join(dir, "index.sqlite"));
    try {
      return (db.prepare(`SELECT COUNT(*) AS c FROM llm_cache`).get() as { c: number }).c;
    } finally {
      db.close();
    }
  }

  beforeEach(async () => {
    dir = mkdtempSync(join(tmpdir(), "qmd-rerank-busy-"));
    rerankCalls = 0;
    resetRerankBreakerForTests();
    stub = createServer((req, res) => {
      if (!req.url?.endsWith("/rerank")) { res.writeHead(404).end(); return; }
      rerankCalls++;
      let body = "";
      req.on("data", c => { body += c; });
      req.on("end", () => {
        const n = (JSON.parse(body).documents as unknown[]).length;
        const results = Array.from({ length: n }, (_, i) => ({ index: i, relevance_score: 1 - i / 10 }));
        res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ results }));
      });
    });
    await new Promise<void>(r => stub.listen(0, "127.0.0.1", r));
    process.env.QMD_RERANK_CACHE_BUSY_TIMEOUT_MS = "100";
  });
  afterEach(async () => {
    if (holder && holder.exitCode === null) {
      const gone = new Promise(r => holder!.once("exit", r));
      holder.kill();
      await gone;
    }
    holder = undefined;
    await handle?.stop();
    stub.closeAllConnections();
    await new Promise<void>(r => stub.close(() => r()));
    _resetProductionModeForTesting();
    resetRerankBreakerForTests();
    restoreEnv(saved);
    rmSync(dir, { recursive: true, force: true });
  });

  test("a reranked search during the hold answers inside its budget with reranked results", async () => {
    await start();
    await holdWriteLock();
    const { ms, res } = await timed(handle.port, alpha);
    expect(ms).toBeLessThan(BUDGET_MS);
    expect(res.result?.isError).toBeUndefined();
    expect(res.result?.structuredContent.results.length).toBe(2);
    expect(rerankCalls).toBe(1); // it was reranked; only the cache write was skipped
  }, 30_000);

  test("a second search issued during the hold is not held behind the first", async () => {
    await start();
    await holdWriteLock();
    const [first, second] = await Promise.all([timed(handle.port, alpha), timed(handle.port, bravo)]);
    expect(first.ms).toBeLessThan(BUDGET_MS);
    expect(second.ms).toBeLessThan(BUDGET_MS);
    expect(first.res.result?.structuredContent.results.length).toBe(2);
    expect(second.res.result?.structuredContent.results.length).toBe(1);
    expect(rerankCalls).toBe(2);
  }, 30_000);

  test("with no hold the scores are cached as before and a repeat search does not rerank again", async () => {
    await start();
    const before = rerankCacheRows();
    await rpc(handle.port, alpha);
    expect(rerankCalls).toBe(1);
    expect(rerankCacheRows()).toBe(before + 2); // one score per reranked chunk
    await rpc(handle.port, alpha);
    expect(rerankCalls).toBe(1);
  }, 30_000);
});
