/**
 * Opening an existing index and syncing an unchanged config must not take the
 * write lock. Both ran a BEGIN IMMEDIATE on every open, so a read-only `qmd
 * query` waited out its busy_timeout behind a long `update` or `embed` and died
 * with SQLITE_BUSY, printing the processor schema DDL as its source excerpt
 * (HIMMEL-11508).
 */
import { describe, test, expect, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, syncConfigToDb, type Store } from "../src/store.js";
import { openDatabase, type Database } from "../src/db.js";
import type { CollectionConfig } from "../src/collections.js";

const savedBusy = process.env.QMD_SQLITE_BUSY_TIMEOUT;
let store: Store | null = null;
let holder: Database | null = null;
let dir = "";

afterEach(async () => {
  holder?.exec("ROLLBACK");
  holder?.close();
  holder = null;
  store?.close();
  store = null;
  if (savedBusy === undefined) delete process.env.QMD_SQLITE_BUSY_TIMEOUT;
  else process.env.QMD_SQLITE_BUSY_TIMEOUT = savedBusy;
  if (dir) await rm(dir, { recursive: true, force: true });
  dir = "";
});

function config(pattern: string): CollectionConfig {
  return { collections: { notes: { path: dir, pattern } } };
}

async function seededIndex(): Promise<string> {
  dir = await mkdtemp(join(tmpdir(), "qmd-open-lock-"));
  const path = join(dir, "index.sqlite");
  const seed = createStore(path);
  syncConfigToDb(seed.db, config("**/*.md"));
  seed.close();
  return path;
}

function holdWriteLock(path: string): void {
  holder = openDatabase(path);
  holder.exec("BEGIN IMMEDIATE");
  process.env.QMD_SQLITE_BUSY_TIMEOUT = "0";
}

describe("opening an existing store and the write lock", () => {
  test("opening an existing store takes no write lock", async () => {
    const path = await seededIndex();
    holdWriteLock(path);
    store = createStore(path);
    expect(store.db.prepare("SELECT count(*) AS n FROM processor_outputs").get()).toEqual({ n: 0 });
  });

  test("syncing an unchanged config takes no write lock", async () => {
    const path = await seededIndex();
    store = createStore(path);
    holdWriteLock(path);
    store.db.exec("PRAGMA busy_timeout = 0");
    syncConfigToDb(store.db, config("**/*.md"));
    expect(store.db.prepare("SELECT pattern FROM store_collections WHERE name = 'notes'").get()).toEqual({ pattern: "**/*.md" });
  });

  test("a changed config is still written", async () => {
    const path = await seededIndex();
    store = createStore(path);
    syncConfigToDb(store.db, config("**/*.txt"));
    expect(store.db.prepare("SELECT pattern FROM store_collections WHERE name = 'notes'").get()).toEqual({ pattern: "**/*.txt" });
  });

  test("a fresh index still gets the processor schema", async () => {
    dir = await mkdtemp(join(tmpdir(), "qmd-open-lock-"));
    store = createStore(join(dir, "index.sqlite"));
    const names = (store.db.prepare("SELECT name FROM sqlite_master WHERE name LIKE '%processor%' AND name NOT LIKE 'sqlite_%' ORDER BY name").all() as { name: string }[]).map((r) => r.name);
    expect(names).toEqual(["idx_processor_outputs_source", "processor_modes", "processor_outputs", "processor_sources", "processor_specs"]);
  });
});
