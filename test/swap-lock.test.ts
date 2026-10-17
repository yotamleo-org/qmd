/**
 * The HTTP daemon start path honours the embed-model swap lock (HIMMEL-4324).
 * The lock is a directory written by himmel's scripts/luna/qmd-embed-model.sh:
 * <cache>/qmd/embed-swap.lock/{role,pid}. A live `swap` holder blocks a start;
 * a dead holder, another role or no lock never does (fail open).
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { swapLockDir, swapLockHolder } from "../src/cli/swap-lock.ts";

let cache: string;
const lock = () => join(cache, "qmd", "embed-swap.lock");

function plant(role: string | null, pid: string | null): void {
  mkdirSync(lock(), { recursive: true });
  if (role !== null) writeFileSync(join(lock(), "role"), role + "\n");
  if (pid !== null) writeFileSync(join(lock(), "pid"), pid + "\n");
}

beforeEach(() => { cache = mkdtempSync(join(tmpdir(), "qmd-swaplock-")); });
afterEach(() => { rmSync(cache, { recursive: true, force: true }); });

describe("swapLockDir", () => {
  test("honours XDG_CACHE_HOME, same path as qmd-embed-model.sh", () => {
    expect(swapLockDir({ XDG_CACHE_HOME: "/x/cache" })).toBe("/x/cache/qmd/embed-swap.lock");
  });
});

describe("swapLockHolder", () => {
  test("no lock directory: nothing holds the swap", () => {
    expect(swapLockHolder(lock())).toBeNull();
  });

  test("a live swap holder is reported with its pid", () => {
    plant("swap", String(process.pid));
    expect(swapLockHolder(lock())).toBe(process.pid);
  });

  test("a dead-pid lock is stale and lets the start proceed", () => {
    plant("swap", "2147483646");
    expect(swapLockHolder(lock())).toBeNull();
  });

  test("a live holder with another role does not block a start", () => {
    plant("ensure", String(process.pid));
    expect(swapLockHolder(lock())).toBeNull();
  });

  test("no pid file: held while fresh, stale once a minute old", () => {
    plant("swap", null);
    expect(swapLockHolder(lock())).toBe(0);
    const old = new Date(Date.now() - 120_000);
    utimesSync(lock(), old, old);
    expect(swapLockHolder(lock())).toBeNull();
  });

  test("a garbled pid file is treated like a missing one", () => {
    plant("swap", "not-a-pid");
    const old = new Date(Date.now() - 120_000);
    utimesSync(lock(), old, old);
    expect(swapLockHolder(lock())).toBeNull();
  });
});
