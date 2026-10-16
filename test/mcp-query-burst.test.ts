import { describe, test, expect, beforeAll, afterAll, vi } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { startMcpHttpServer, createExpansionGate, type HttpServerHandle } from "../src/mcp/server";
import { _resetProductionModeForTesting } from "../src/store";

// Lets one test replace store.search; every other call reaches the real store.
const searchStub = vi.hoisted(() => ({ fn: null as null | ((opts: unknown) => Promise<unknown[]>) }));
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

describe("query tool defaults", () => {
  let handle: HttpServerHandle;
  let dir: string;
  const origIndex = process.env.INDEX_PATH;
  const origCfg = process.env.QMD_CONFIG_DIR;

  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "qmd-burst-"));
    process.env.INDEX_PATH = join(dir, "index.sqlite");
    process.env.QMD_CONFIG_DIR = dir;
    writeFileSync(join(dir, "index.yml"), "collections: {}\n");
    handle = await startMcpHttpServer(0, { quiet: true, dbPath: join(dir, "index.sqlite") });
  });
  afterAll(async () => {
    await handle.stop();
    _resetProductionModeForTesting();
    if (origIndex !== undefined) process.env.INDEX_PATH = origIndex; else delete process.env.INDEX_PATH;
    if (origCfg !== undefined) process.env.QMD_CONFIG_DIR = origCfg; else delete process.env.QMD_CONFIG_DIR;
    rmSync(dir, { recursive: true, force: true });
  });

  test("rerank is opt-in: the advertised default is false", async () => {
    const res = await fetch(`http://localhost:${handle.port}/mcp`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json, text/event-stream" },
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/list", params: {} }),
    });
    const text = await res.text();
    const data = text.split("\n").find(l => l.startsWith("data: "));
    const body = JSON.parse(data ? data.slice(6) : text) as { result: { tools: Array<{ name: string; inputSchema: { properties: Record<string, { default?: unknown }> } }> } };
    const query = body.result.tools.find(t => t.name === "query")!;
    expect(query.inputSchema.properties.rerank!.default).toBe(false);
  });
});

describe("expansion admission gate", () => {
  test("10 parallel plain queries: the cap are admitted, the rest are refused at once", () => {
    const gate = createExpansionGate(3);
    const results = Array.from({ length: 10 }, () => gate.tryEnter());
    expect(results.filter(r => r !== null)).toHaveLength(3);
    expect(gate.depth()).toBe(3);
  });

  test("leaving frees a slot", () => {
    const gate = createExpansionGate(1);
    const leave = gate.tryEnter()!;
    expect(gate.tryEnter()).toBeNull();
    leave();
    leave(); // idempotent
    expect(gate.depth()).toBe(0);
    expect(gate.tryEnter()).not.toBeNull();
  });
});

describe("query handler admission", () => {
  let handle: HttpServerHandle;
  let dir: string;
  const origIndex = process.env.INDEX_PATH;
  const origCfg = process.env.QMD_CONFIG_DIR;

  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "qmd-burst-h-"));
    process.env.INDEX_PATH = join(dir, "index.sqlite");
    process.env.QMD_CONFIG_DIR = dir;
    writeFileSync(join(dir, "index.yml"), "collections: {}\n");
    handle = await startMcpHttpServer(0, { quiet: true, dbPath: join(dir, "index.sqlite") });
  });
  afterAll(async () => {
    searchStub.fn = null;
    await handle.stop();
    _resetProductionModeForTesting();
    if (origIndex !== undefined) process.env.INDEX_PATH = origIndex; else delete process.env.INDEX_PATH;
    if (origCfg !== undefined) process.env.QMD_CONFIG_DIR = origCfg; else delete process.env.QMD_CONFIG_DIR;
    rmSync(dir, { recursive: true, force: true });
  });

  async function callQuery(args: Record<string, unknown>, id: number) {
    const res = await fetch(`http://localhost:${handle.port}/mcp`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json, text/event-stream", "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call", "Mcp-Name": "query" },
      body: JSON.stringify({
        jsonrpc: "2.0", id, method: "tools/call",
        params: {
          name: "query", arguments: args,
          _meta: {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientInfo": { name: "burst-test", version: "1.0.0" },
            "io.modelcontextprotocol/clientCapabilities": {},
          },
        },
      }),
    });
    return (await res.json()) as { result: { isError?: boolean; content: Array<{ text: string }> } };
  }
  const isBusy = (r: Awaited<ReturnType<typeof callQuery>>) => r.result.isError === true && r.result.content[0]!.text.startsWith("Busy:");

  test("plain queries past the cap get Busy at once; typed searches are never refused", async () => {
    let release!: () => void;
    const blocked = new Promise<void>(r => { release = r; });
    let entered = 0;
    searchStub.fn = async () => { entered += 1; await blocked; return []; };
    try {
      const plain = Array.from({ length: 6 }, (_, i) => callQuery({ query: `q${i}` }, i + 1));
      const typed = callQuery({ searches: [{ type: "lex", query: "x" }] }, 100);
      await vi.waitFor(() => expect(entered).toBe(4), { timeout: 5000 }); // 3 admitted plain + 1 typed
      release();
      const results = await Promise.all(plain);
      expect(results.filter(isBusy)).toHaveLength(3);
      expect(isBusy(await typed)).toBe(false);
    } finally {
      release();
    }
  });

  test("a throwing search returns its slot", async () => {
    searchStub.fn = async () => { throw new Error("boom"); };
    for (let i = 0; i < 6; i++) {
      const r = await callQuery({ query: `t${i}` }, 200 + i);
      expect(isBusy(r)).toBe(false);
    }
    searchStub.fn = null;
  });
});
