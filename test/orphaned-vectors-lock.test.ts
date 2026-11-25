/**
 * cleanupOrphanedVectors runs at the end of every `qmd update`. It must not
 * hold the index write lock while it searches for orphans: its orphan query ran
 * for 10+ minutes on a 7 GB index when SQLite scanned `documents` through
 * idx_documents_collection once per vector row, and every other writer
 * (the post-merge refresh, agents' queries, the daemon's first open) waited
 * out its busy_timeout and failed with SQLITE_BUSY (HIMMEL-11112).
 */
import { describe, test, expect, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore,
  cleanupOrphanedVectors,
  insertContent,
  insertDocument,
  deactivateDocument,
  ORPHANED_VECTOR_ROWS_SQL,
  type Store,
} from "../src/store.js";
import { openDatabase } from "../src/db.js";

let store: Store | null = null;
let dir = "";

async function openStore(): Promise<Store> {
  dir = await mkdtemp(join(tmpdir(), "qmd-orphan-lock-"));
  store = createStore(join(dir, "index.sqlite"));
  return store;
}

afterEach(async () => {
  store?.close();
  store = null;
  if (dir) await rm(dir, { recursive: true, force: true });
  dir = "";
});

function seed(s: Store, collection: string, count: number): void {
  const now = new Date().toISOString();
  s.ensureVecTable(3);
  s.db.transaction(() => {
    for (let i = 0; i < count; i++) {
      const hash = `${collection}${String(i).padStart(5, "0")}`;
      insertContent(s.db, hash, `body ${hash}`, now);
      insertDocument(s.db, collection, `${hash}.md`, hash, hash, now, now);
      s.insertEmbedding(hash, 0, 0, new Float32Array([1, i * 0.001, 0]), "test-model", now, 1);
    }
  })();
}

describe("cleanupOrphanedVectors and the write lock", () => {
  test("with nothing to remove it takes no write lock", async () => {
    const s = await openStore();
    seed(s, "docs", 5);
    const other = openDatabase(s.dbPath);
    other.exec("BEGIN IMMEDIATE");
    try {
      s.db.exec("PRAGMA busy_timeout = 0");
      expect(cleanupOrphanedVectors(s.db)).toBe(0);
    } finally {
      other.exec("ROLLBACK");
      other.close();
    }
  });

  test("it still removes the vector rows of a deactivated document", async () => {
    const s = await openStore();
    seed(s, "docs", 5);
    deactivateDocument(s.db, "docs", "docs00002.md");
    expect(cleanupOrphanedVectors(s.db)).toBe(1);
    expect(cleanupOrphanedVectors(s.db)).toBe(0);
  });

  test("the orphan query looks documents up by hash, not by scanning a collection", async () => {
    const s = await openStore();
    seed(s, "docs", 50);
    seed(s, "notes", 50);
    const plan = (s.db.prepare(`EXPLAIN QUERY PLAN ${ORPHANED_VECTOR_ROWS_SQL}`).all() as { detail: string }[])
      .map((r) => r.detail)
      .join(" | ");
    expect(plan).toContain("idx_documents_hash");
    expect(plan).not.toContain("idx_documents_collection");
  });
});
