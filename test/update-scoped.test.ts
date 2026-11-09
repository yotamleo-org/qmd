/**
 * HIMMEL-9517: `qmd update -c <collection>` indexes one collection, and `update` (scoped or not)
 * keeps the LLM cache; only `qmd cleanup` clears it. Real qmd processes on a scratch index.
 */
import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { openDatabase } from "../src/db.ts";

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");

let root: string;
let dbPath: string;
let configDir: string;
let dirA: string;
let dirB: string;

async function runQmd(args: string[]): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const argv = isBunRuntime ? [qmdScript, ...args] : [tsxCli, qmdScript, ...args];
  const proc = spawn(process.execPath, argv, {
    cwd: root,
    env: {
      ...process.env,
      INDEX_PATH: dbPath,
      QMD_CONFIG_DIR: configDir,
      PWD: root,
      QMD_DOCTOR_DEVICE_PROBE: "0",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  proc.stdout.on("data", (d: Buffer) => { stdout += d.toString(); });
  proc.stderr.on("data", (d: Buffer) => { stderr += d.toString(); });
  const exitCode = await new Promise<number>((resolve, reject) => {
    proc.once("error", reject);
    proc.once("close", code => resolve(code ?? 1));
  });
  return { stdout, stderr, exitCode };
}

function query<T>(sql: string, ...params: string[]): T[] {
  const db = openDatabase(dbPath);
  try {
    return db.prepare(sql).all(...params) as T[];
  } finally {
    db.close();
  }
}

const hashOf = (collection: string) =>
  query<{ hash: string }>(`SELECT hash FROM documents WHERE collection = ? AND active = 1`, collection)[0]?.hash;
const cacheRows = () => query<{ hash: string }>(`SELECT hash FROM llm_cache WHERE hash = 'seeded-row'`).length;

function seedCache(): void {
  const db = openDatabase(dbPath);
  try {
    db.prepare(`INSERT OR REPLACE INTO llm_cache (hash, result, created_at) VALUES ('seeded-row', '{}', ?)`)
      .run(new Date().toISOString());
  } finally {
    db.close();
  }
}

let round = 0;
function changeBothFiles(): void {
  round++;
  writeFileSync(join(dirA, "note.md"), `# A\n\nchanged in round ${round}\n`);
  writeFileSync(join(dirB, "note.md"), `# B\n\nchanged in round ${round}\n`);
}

describe("qmd update, scoped and cache-keeping (HIMMEL-9517)", () => {
  beforeAll(async () => {
    root = mkdtempSync(join(tmpdir(), "qmd-update-scoped-"));
    dbPath = join(root, "index.sqlite");
    configDir = join(root, "config");
    dirA = join(root, "a");
    dirB = join(root, "b");
    mkdirSync(configDir, { recursive: true });
    mkdirSync(dirA);
    mkdirSync(dirB);
    writeFileSync(join(configDir, "index.yml"), "collections: {}\n");
    writeFileSync(join(dirA, "note.md"), "# A\n\noriginal\n");
    writeFileSync(join(dirB, "note.md"), "# B\n\noriginal\n");
    for (const [name, dir] of [["a", dirA], ["b", dirB]] as const) {
      const added = await runQmd(["collection", "add", dir, "--name", name]);
      expect(added.exitCode).toBe(0);
    }
  });

  afterAll(() => {
    rmSync(root, { recursive: true, force: true });
  });

  test("scoped_indexes_one_collection: update -c a re-indexes only a", async () => {
    const beforeA = hashOf("a");
    const beforeB = hashOf("b");
    changeBothFiles();

    const { stdout, exitCode } = await runQmd(["update", "-c", "a"]);

    expect(exitCode).toBe(0);
    expect(stdout).toContain("Updating 1 collection(s)");
    expect(hashOf("a")).not.toBe(beforeA);
    expect(hashOf("b")).toBe(beforeB);

    // an unscoped update still walks every collection and catches up b
    const all = await runQmd(["update"]);
    expect(all.exitCode).toBe(0);
    expect(all.stdout).toContain("Updating 2 collection(s)");
    expect(hashOf("b")).not.toBe(beforeB);
  });

  test("update_keeps_llm_cache: a seeded llm_cache row survives update and update -c a", async () => {
    seedCache();
    expect(cacheRows()).toBe(1);

    expect((await runQmd(["update"])).exitCode).toBe(0);
    expect(cacheRows()).toBe(1);

    expect((await runQmd(["update", "-c", "a"])).exitCode).toBe(0);
    expect(cacheRows()).toBe(1);
  });

  test("cleanup_still_clears: the same row is gone after cleanup", async () => {
    seedCache();
    expect(cacheRows()).toBe(1);

    const { exitCode } = await runQmd(["cleanup"]);

    expect(exitCode).toBe(0);
    expect(cacheRows()).toBe(0);
  });

  test("unknown_collection_errors: update -c nope exits 1 as embed -c does", async () => {
    const { stderr, exitCode } = await runQmd(["update", "-c", "nope"]);

    expect(exitCode).toBe(1);
    expect(stderr).toContain("Collection not found: nope");
  });
});
