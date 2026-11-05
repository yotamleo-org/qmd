/**
 * HIMMEL-9068: a plain query saves its expanded sub-queries in llm_cache, and hybridQuery deletes
 * that row when every sub-query came back empty. Both are synchronous SQLite writes on the daemon's
 * one thread: while another process (qmd update, qmd embed) holds the write lock they must not wait
 * out the connection's 120 s busy_timeout. Real store and database, a second process that holds the
 * write lock with BEGIN IMMEDIATE for longer than the write may wait.
 */
import { describe, test, expect, afterEach, beforeEach } from "vitest";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { isBun } from "../src/db.ts";
import type { LLM } from "../src/llm.ts";
import { createStore, getCacheKey, getCachedResult, setCachedResult, type Store } from "../src/store.ts";

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const HOLD_MS = 6_000;
const BUDGET_MS = 2_000; // what the write may take here; the hold outlasts it threefold
const MODEL = "stub-gen";
const QUERY = "how do i stop a search freezing the daemon";

describe("expansion cache writes: a write lock held by another process never holds a plain query", () => {
  let dir: string;
  let store: Store;
  let expandCalls: number;
  let holder: ChildProcess | undefined;
  const savedEnv = process.env.QMD_RERANK_CACHE_BUSY_TIMEOUT_MS;

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

  const cacheKey = () => getCacheKey("expandQuery", { query: QUERY, model: MODEL });

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "qmd-expansion-busy-"));
    expandCalls = 0;
    store = createStore(join(dir, "index.sqlite"));
    store.llm = {
      generateModelName: MODEL,
      expandQuery: async () => {
        expandCalls++;
        return [{ type: "lex", text: "freeze daemon" }, { type: "vec", text: "daemon stops answering searches" }];
      },
    } satisfies Partial<LLM> as LLM;
    process.env.QMD_RERANK_CACHE_BUSY_TIMEOUT_MS = "100";
  });
  afterEach(async () => {
    if (holder && holder.exitCode === null) {
      const gone = new Promise(r => holder!.once("exit", r));
      holder.kill();
      await gone;
    }
    holder = undefined;
    store.close();
    if (savedEnv !== undefined) process.env.QMD_RERANK_CACHE_BUSY_TIMEOUT_MS = savedEnv; else delete process.env.QMD_RERANK_CACHE_BUSY_TIMEOUT_MS;
    rmSync(dir, { recursive: true, force: true });
  });

  test("expanding a plain query during the hold answers inside its budget and skips the cache write", async () => {
    await holdWriteLock();
    const t0 = Date.now();
    const expanded = await store.expandQuery(QUERY);
    expect(Date.now() - t0).toBeLessThan(BUDGET_MS);
    expect(expanded.map(e => e.type)).toEqual(["lex", "vec"]);
    expect(expandCalls).toBe(1);
    expect(getCachedResult(store.db, cacheKey())).toBeNull(); // skipped, not lost for good
  }, 30_000);

  test("dropping a cached expansion during the hold answers inside its budget and leaves the row", async () => {
    setCachedResult(store.db, cacheKey(), JSON.stringify([{ type: "vec", query: "x" }]));
    await holdWriteLock();
    const t0 = Date.now();
    store.invalidateExpansionCache(QUERY);
    expect(Date.now() - t0).toBeLessThan(BUDGET_MS);
    expect(getCachedResult(store.db, cacheKey())).not.toBeNull();
  }, 30_000);

  test("with no hold the expansion is cached as before and a repeat does not expand again", async () => {
    await store.expandQuery(QUERY);
    expect(expandCalls).toBe(1);
    expect(getCachedResult(store.db, cacheKey())).not.toBeNull();
    await store.expandQuery(QUERY);
    expect(expandCalls).toBe(1);
  }, 30_000);

  test("with no hold dropping a cached expansion deletes its row", async () => {
    await store.expandQuery(QUERY);
    store.invalidateExpansionCache(QUERY);
    expect(getCachedResult(store.db, cacheKey())).toBeNull();
  }, 30_000);
});
