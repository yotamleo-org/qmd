import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { startMcpHttpServer, createExpansionGate, type HttpServerHandle } from "../src/mcp/server";
import { _resetProductionModeForTesting } from "../src/store";

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
