/**
 * Embed-model swap lock check for the HTTP daemon start path (HIMMEL-4324).
 *
 * himmel's scripts/luna/qmd-embed-model.sh `swap` holds the directory
 * <cache>/qmd/embed-swap.lock (files `role` and `pid`) while it replaces the
 * index and flips models.embed. A daemon started inside that window embeds
 * queries with the OLD model against the NEW vectors. The directory path and
 * stale rule mirror that script: a dead pid is stale; no readable pid is stale
 * once the directory is a minute old (the holder died between mkdir and the
 * pid write). Only role `swap` blocks a start. Every other state, and every
 * read error, lets the start proceed: this check must never wedge the daemon.
 */

import { readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const STALE_AFTER_MS = 60_000;

export function swapLockDir(env: NodeJS.ProcessEnv = process.env): string {
  const cache = env.XDG_CACHE_HOME || join(homedir(), ".cache");
  return join(cache, "qmd", "embed-swap.lock");
}

function readTrimmed(path: string): string | null {
  try {
    return readFileSync(path, "utf-8").trim();
  } catch {
    return null;
  }
}

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err: unknown) {
    // EPERM: the process exists but belongs to another user.
    return typeof err === "object" && err !== null && (err as NodeJS.ErrnoException).code === "EPERM";
  }
}

/**
 * The pid of a live `swap` holding the lock, 0 when it holds the lock but has
 * not written its pid yet, or null when nothing blocks a daemon start.
 */
export function swapLockHolder(lockDir: string = swapLockDir()): number | null {
  if (readTrimmed(join(lockDir, "role")) !== "swap") return null;
  const raw = readTrimmed(join(lockDir, "pid"));
  if (raw !== null && /^[0-9]+$/.test(raw) && Number(raw) > 0) {
    return pidAlive(Number(raw)) ? Number(raw) : null;
  }
  try {
    return Date.now() - statSync(lockDir).mtimeMs > STALE_AFTER_MS ? null : 0;
  } catch {
    return null;
  }
}
