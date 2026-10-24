import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { startMcpHttpServer, type HttpServerHandle } from "../src/mcp/server.ts";
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

const TASKS_EXT = "io.modelcontextprotocol/tasks";

function envelope(withTasks: boolean) {
  return {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientInfo": { name: "task-test", version: "1.0.0" },
    "io.modelcontextprotocol/clientCapabilities": withTasks ? { extensions: { [TASKS_EXT]: {} } } : {},
  };
}

type Rpc = { result?: Record<string, any>; error?: { code: number; message: string; data?: Record<string, any> } };

const queryArgs = { searches: [{ type: "lex", query: "alpha" }] };

describe("query tool as an MCP task — HTTP", () => {
  let handle: HttpServerHandle;
  let dir: string;
  const origIndex = process.env.INDEX_PATH;
  const origCfg = process.env.QMD_CONFIG_DIR;
  let gate: Promise<void>;
  let release!: () => void;

  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "qmd-task-"));
    process.env.INDEX_PATH = join(dir, "index.sqlite");
    process.env.QMD_CONFIG_DIR = dir;
    writeFileSync(join(dir, "index.yml"), "collections: {}\n");
    searchStub.fn = async () => {
      await gate;
      return [{
        docid: "abc123", displayPath: "notes/alpha.md", title: "Alpha", score: 0.9, context: "ctx",
        metadata: {}, body: "alpha line one\nalpha line two\n", bestChunkPos: 0, bestChunk: "alpha line one",
      }];
    };
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

  async function rpc(method: string, params: Record<string, unknown>, withTasks: boolean, id = 1): Promise<Rpc> {
    const res = await fetch(`http://localhost:${handle.port}/mcp`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json", Accept: "application/json, text/event-stream",
        "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": method,
        ...(method === "tools/call" ? { "Mcp-Name": "query" } : {}),
        ...(method === "tasks/get" || method === "tasks/cancel" ? { "Mcp-Name": String(params.taskId) } : {}),
      },
      body: JSON.stringify({ jsonrpc: "2.0", id, method, params: { ...params, _meta: envelope(withTasks) } }),
    });
    return (await res.json()) as Rpc;
  }

  const open = () => { gate = new Promise<void>(r => { release = r; }); };

  test("without the extension the call is synchronous and unchanged", async () => {
    open(); release();
    const res = await rpc("tools/call", { name: "query", arguments: queryArgs }, false);
    expect(res.result?.resultType).toBe("complete");
    expect(res.result?.taskId).toBeUndefined();
    expect(res.result?.structuredContent.results).toHaveLength(1);
    expect(res.result?.content[0].text).toContain("notes/alpha.md");
  });

  test("with the extension the call returns a task at once and tasks/get reaches complete with the same result", async () => {
    open(); release();
    const sync = await rpc("tools/call", { name: "query", arguments: queryArgs }, false);
    open();
    const created = await rpc("tools/call", { name: "query", arguments: queryArgs }, true);
    expect(created.result?.resultType).toBe("task");
    const task = created.result!;
    expect(task.status).toBe("working");
    expect(typeof task.taskId).toBe("string");
    expect(task.taskId.length).toBeLessThanOrEqual(128);
    expect(typeof task.createdAt).toBe("string");
    expect(typeof task.lastUpdatedAt).toBe("string");
    expect("ttlMs" in task).toBe(true);

    const working = await rpc("tasks/get", { taskId: task.taskId }, true);
    expect(working.result?.resultType).toBe("complete");
    expect(working.result?.status).toBe("working");

    release();
    let done: Rpc = working;
    await vi.waitFor(async () => {
      done = await rpc("tasks/get", { taskId: task.taskId }, true);
      expect(done.result?.status).toBe("completed");
    }, { timeout: 5000 });
    expect(done.result?.resultType).toBe("complete");
    expect(done.result?.taskId).toBe(task.taskId);
    expect(done.result?.result.structuredContent).toEqual(sync.result?.structuredContent);
    expect(done.result?.result.content).toEqual(sync.result?.content);
  });

  test("a search that throws ends the task completed with the same error result as a synchronous call", async () => {
    const ok = searchStub.fn;
    searchStub.fn = async () => { throw new Error("rerank exploded"); };
    try {
      const sync = await rpc("tools/call", { name: "query", arguments: queryArgs }, false);
      expect(sync.result?.isError).toBe(true);
      const created = await rpc("tools/call", { name: "query", arguments: queryArgs }, true);
      expect(created.result?.resultType).toBe("task");
      let done: Rpc = created;
      await vi.waitFor(async () => {
        done = await rpc("tasks/get", { taskId: created.result!.taskId }, true);
        expect(done.result?.status).toBe("completed");
      }, { timeout: 5000 });
      expect(done.result?.result.isError).toBe(true);
      expect(done.result?.result.content).toEqual(sync.result?.content);
      expect(done.result?.result.content[0].text).toContain("rerank exploded");
    } finally {
      searchStub.fn = ok;
    }
  });

  test("an unknown task id is an invalid-params error for get and cancel", async () => {
    expect((await rpc("tasks/get", { taskId: "nope" }, true)).error?.code).toBe(-32602);
    expect((await rpc("tasks/cancel", { taskId: "nope" }, true)).error?.code).toBe(-32602);
  });

  test("tasks/get and tasks/cancel refuse a client that did not declare the extension", async () => {
    open();
    const created = await rpc("tools/call", { name: "query", arguments: queryArgs }, true);
    const taskId = created.result!.taskId;
    for (const method of ["tasks/get", "tasks/cancel"]) {
      const res = await rpc(method, { taskId }, false);
      expect(res.error?.code).toBe(-32021);
      expect(res.error?.data?.requiredCapabilities).toEqual({ extensions: { [TASKS_EXT]: {} } });
    }
    release();
  });

  test("tasks/cancel acks with an empty result and the task stays cancelled when the search finishes", async () => {
    open();
    const created = await rpc("tools/call", { name: "query", arguments: queryArgs }, true);
    const taskId = created.result!.taskId;
    const ack = await rpc("tasks/cancel", { taskId }, true);
    expect(ack.error).toBeUndefined();
    expect(ack.result?.resultType).toBe("complete");
    release();
    await new Promise(r => setTimeout(r, 100));
    const after = await rpc("tasks/get", { taskId }, true);
    expect(after.result?.status).toBe("cancelled");
    expect(after.result?.result).toBeUndefined();
  });
});

describe("query tool as an MCP task — stdio", () => {
  const repoRoot = fileURLToPath(new URL("..", import.meta.url));
  const cliPath = join(repoRoot, "src", "cli", "qmd.ts");
  let workDir: string;
  let server: ChildProcessWithoutNullStreams;
  let buffer = "";
  const waiting = new Map<number, (m: Rpc) => void>();

  // A fresh process per test: an initialize after stateless 2026 requests is not a clean 2025 session.
  beforeEach(async () => {
    buffer = "";
    waiting.clear();
    workDir = await mkdtemp(join(tmpdir(), "qmd-task-stdio-"));
    await writeFile(join(workDir, "index.yml"), "collections: {}\n");
    const runtimeArgs = process.versions.bun ? [cliPath, "mcp"] : ["--import", "tsx", cliPath, "mcp"];
    server = spawn(process.execPath, runtimeArgs, {
      cwd: repoRoot,
      env: { ...process.env, INDEX_PATH: join(workDir, "task.sqlite"), QMD_CONFIG_DIR: workDir },
      stdio: ["pipe", "pipe", "pipe"],
    });
    server.stdout.on("data", (chunk) => {
      buffer += String(chunk);
      let nl: number;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl);
        buffer = buffer.slice(nl + 1);
        if (!line.trim()) continue;
        const msg = JSON.parse(line) as Rpc & { id?: number };
        if (typeof msg.id === "number") waiting.get(msg.id)?.(msg);
      }
    });
  });
  afterEach(async () => {
    server.kill("SIGKILL");
    await rm(workDir, { recursive: true, force: true });
  });

  let nextId = 1;
  // withTasks null sends no envelope, as a pre-2026 client does.
  function rpc(method: string, params: Record<string, unknown>, withTasks: boolean | null): Promise<Rpc> {
    const id = nextId++;
    return new Promise<Rpc>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`no reply to ${method}`)), 45_000);
      waiting.set(id, (m) => { clearTimeout(timer); resolve(m); });
      const body = withTasks === null ? params : { ...params, _meta: envelope(withTasks) };
      server.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params: body }) + "\n");
    });
  }

  test("tasks extension: task at once, tasks/get completes with the synchronous result; without it unchanged", async () => {
    const sync = await rpc("tools/call", { name: "query", arguments: queryArgs }, false);
    expect(sync.result?.resultType).toBe("complete");
    expect(sync.result?.structuredContent).toEqual({ results: [] });

    const created = await rpc("tools/call", { name: "query", arguments: queryArgs }, true);
    expect(created.result?.resultType).toBe("task");
    let done: Rpc = created;
    await vi.waitFor(async () => {
      done = await rpc("tasks/get", { taskId: created.result!.taskId }, true);
      expect(done.result?.status).toBe("completed");
    }, { timeout: 30_000 });
    expect(done.result?.result.structuredContent).toEqual(sync.result?.structuredContent);
    expect(done.result?.result.content).toEqual(sync.result?.content);
  }, 60_000);

  test("a 2025-era initialize session gets the synchronous result", async () => {
    const init = await rpc("initialize", {
      protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "legacy", version: "1.0.0" },
    }, null);
    expect(init.result?.protocolVersion).toBe("2025-06-18");
    const res = await rpc("tools/call", { name: "query", arguments: queryArgs }, null);
    expect(res.result?.taskId).toBeUndefined();
    expect(res.result?.resultType).not.toBe("task");
    expect(res.result?.structuredContent).toEqual({ results: [] });
  }, 60_000);
});
