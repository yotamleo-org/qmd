import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, rmSync, statSync, readFileSync, writeFileSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadOrCreateToken, checkBearer } from "../src/mcp/auth";
import { startMcpHttpServer, type HttpServerHandle } from "../src/mcp/server";
import { _resetProductionModeForTesting } from "../src/store";

describe("http bearer token file", () => {
  test("is generated on first use with mode 0600 and reused afterwards", () => {
    const dir = mkdtempSync(join(tmpdir(), "qmd-auth-"));
    try {
      const f = join(dir, "sub", "http-token");
      const t1 = loadOrCreateToken(f);
      expect(t1).toMatch(/^[0-9a-f]{64}$/);
      expect(statSync(f).mode & 0o777).toBe(0o600);
      expect(loadOrCreateToken(f)).toBe(t1);
      expect(readFileSync(f, "utf8").trim()).toBe(t1);
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  test("refuses a group/world-readable token file", () => {
    const dir = mkdtempSync(join(tmpdir(), "qmd-auth-"));
    try {
      const f = join(dir, "http-token");
      writeFileSync(f, "a".repeat(64));
      chmodSync(f, 0o644);
      expect(() => loadOrCreateToken(f)).toThrow(/0600/);
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  test("checkBearer accepts only the exact token", () => {
    expect(checkBearer("Bearer abc", "abc")).toBe(true);
    expect(checkBearer("Bearer abd", "abc")).toBe(false);
    expect(checkBearer("Bearer abcd", "abc")).toBe(false);
    expect(checkBearer("abc", "abc")).toBe(false);
    expect(checkBearer(undefined, "abc")).toBe(false);
  });
});

describe("HTTP daemon requires the bearer token", () => {
  let handle: HttpServerHandle;
  let dir: string;
  let base: string;
  const TOKEN = "t".repeat(64);
  const origIndex = process.env.INDEX_PATH;
  const origCfg = process.env.QMD_CONFIG_DIR;

  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "qmd-auth-http-"));
    process.env.INDEX_PATH = join(dir, "index.sqlite");
    process.env.QMD_CONFIG_DIR = dir;
    writeFileSync(join(dir, "index.yml"), "collections: {}\n");
    handle = await startMcpHttpServer(0, { quiet: true, authToken: TOKEN, dbPath: join(dir, "index.sqlite") });
    base = `http://localhost:${handle.port}`;
  });
  afterAll(async () => {
    await handle.stop();
    _resetProductionModeForTesting();
    if (origIndex !== undefined) process.env.INDEX_PATH = origIndex; else delete process.env.INDEX_PATH;
    if (origCfg !== undefined) process.env.QMD_CONFIG_DIR = origCfg; else delete process.env.QMD_CONFIG_DIR;
    rmSync(dir, { recursive: true, force: true });
  });

  const post = (path: string, headers: Record<string, string> = {}) =>
    fetch(`${base}${path}`, { method: "POST", headers: { "Content-Type": "application/json", ...headers }, body: "not json" });

  test.each(["/query", "/search", "/mcp"])("POST %s without a token is 401", async (p) => {
    expect((await post(p)).status).toBe(401);
  });

  test("a wrong token is 401", async () => {
    expect((await post("/query", { Authorization: "Bearer nope" })).status).toBe(401);
  });

  test("the right token passes auth (reaches the body parser, not 401)", async () => {
    expect((await post("/query", { Authorization: `Bearer ${TOKEN}` })).status).not.toBe(401);
  });

  test("/health stays open for liveness probes", async () => {
    expect((await fetch(`${base}/health`)).status).toBe(200);
  });
});
