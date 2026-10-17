import { describe, test, expect, beforeAll, afterAll, vi } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { startMcpHttpServer, createQueryQueue, type HttpServerHandle } from "../src/mcp/server.ts";
import { _resetProductionModeForTesting } from "../src/store.ts";

// Lets a test replace store.search; every other call reaches the real store.
const searchStub = vi.hoisted(() => ({ fn: null as null | ((opts: unknown) => Promise<unknown[]>) }));
// oxlint-disable-next-line anti-slop/no-module-mocking -- store.search has no injection seam, replace under HIMMEL-5938
vi.mock("../src/index.js", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../src/index.js")>();
  return {
    ...mod,
    createStore: async (...args: Parameters<typeof mod.createStore>) => {
      const store = await mod.createStore(...args);
      const realSearch = store.search.bind(store);
      store.search = (opts) => (searchStub.fn ? searchStub.fn(opts) : realSearch(opts)) as ReturnType<typeof realSearch>;
      return store;
    },
  };
});

describe("query queue", () => {
  test("N+1 parallel calls: the cap is admitted, the rest get busy at once", async () => {
    const q = createQueryQueue(3);
    const first = q.acquire();
    const rest = Array.from({ length: 9 }, () => q.acquire());
    const results = await Promise.all([first, ...rest].map(p => Promise.race([p, new Promise<"pending">(r => setTimeout(() => r("pending"), 50))])));
    const busy = results.filter(r => r !== "pending" && !r.ok && r.reason === "busy");
    expect(busy).toHaveLength(7);
    expect(q.depth()).toBe(3);
  });

  test("calls run one at a time, in arrival order", async () => {
    const q = createQueryQueue(3);
    const order: number[] = [];
    const run = async (n: number) => {
      const t = await q.acquire();
      if (!t.ok) throw new Error("unexpected " + t.reason);
      order.push(n);
      await new Promise(r => setTimeout(r, 10));
      t.release();
    };
    await Promise.all([run(1), run(2), run(3)]);
    expect(order).toEqual([1, 2, 3]);
    expect(q.depth()).toBe(0);
  });

  test("a call cancelled while queued never starts and frees its slot", async () => {
    const q = createQueryQueue(2);
    const head = await q.acquire();
    if (!head.ok) throw new Error("head not admitted");
    const ac = new AbortController();
    const queued = q.acquire(ac.signal);
    expect(q.depth()).toBe(2);
    ac.abort();
    const res = await queued;
    expect(res).toEqual({ ok: false, reason: "cancelled" });
    expect(q.depth()).toBe(1);
    // the freed slot admits a new call, and the head's release does not hand it to the dead one
    const next = q.acquire();
    head.release();
    const granted = await next;
    expect(granted.ok).toBe(true);
    if (granted.ok) granted.release();
    expect(q.depth()).toBe(0);
  });

  test("a call already cancelled on arrival is refused without taking a slot", async () => {
    const q = createQueryQueue(2);
    const ac = new AbortController();
    ac.abort();
    expect(await q.acquire(ac.signal)).toEqual({ ok: false, reason: "cancelled" });
    expect(q.depth()).toBe(0);
  });
});

describe("daemon: typed vec searches are capped", () => {
  let handle: HttpServerHandle;
  let dir: string;
  let release: () => void;
  const origIndex = process.env.INDEX_PATH;
  const origCfg = process.env.QMD_CONFIG_DIR;

  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "qmd-queue-"));
    process.env.INDEX_PATH = join(dir, "index.sqlite");
    process.env.QMD_CONFIG_DIR = dir;
    writeFileSync(join(dir, "index.yml"), "collections: {}\n");
    const gate = new Promise<void>(r => { release = r; });
    searchStub.fn = async () => { await gate; return []; };
    handle = await startMcpHttpServer(0, { quiet: true, dbPath: join(dir, "index.sqlite") });
  });
  afterAll(async () => {
    release();
    searchStub.fn = null;
    await handle.stop();
    _resetProductionModeForTesting();
    if (origIndex !== undefined) process.env.INDEX_PATH = origIndex; else delete process.env.INDEX_PATH;
    if (origCfg !== undefined) process.env.QMD_CONFIG_DIR = origCfg; else delete process.env.QMD_CONFIG_DIR;
    rmSync(dir, { recursive: true, force: true });
  });

  test("10 parallel vec searches: calls past the cap answer Busy inside 2 s", async () => {
    const call = async (i: number) => {
      const started = Date.now();
      const res = await fetch(`http://localhost:${handle.port}/mcp`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json, text/event-stream" },
        body: JSON.stringify({
          jsonrpc: "2.0", id: i + 1, method: "tools/call",
          params: { name: "query", arguments: { searches: [{ type: "vec", query: `burst probe ${i}` }] } },
        }),
        signal: AbortSignal.timeout(2000),
      }).then(r => r.text(), () => "TIMEOUT");
      return { text: res, ms: Date.now() - started };
    };
    const results = await Promise.all(Array.from({ length: 10 }, (_, i) => call(i)));
    const busy = results.filter(r => r.text.includes("Busy"));
    expect(busy.length).toBe(7);
    for (const b of busy) expect(b.ms).toBeLessThan(2000);
  });
});
