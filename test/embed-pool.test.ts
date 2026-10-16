/**
 * embed-pool.test.ts - The pieces of the remote embed pipeline that need no
 * network: env knobs, the barrier-free worker pool, retry with backoff, the
 * adaptive concurrency limit and request-latency stats.
 */

import { describe, test, expect, afterEach } from "vitest";
import {
  readPositiveIntEnv,
  HttpStatusError,
  parseRetryAfterMs,
  withRetry,
  AdaptiveLimit,
  LatencyStats,
  runPool,
} from "../src/embed-pool.js";

const sleepFor = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

describe("readPositiveIntEnv", () => {
  afterEach(() => { delete process.env.QMD_TEST_KNOB; });

  test("unset, junk, zero and negative fall back; a good value wins; the cap clamps", () => {
    expect(readPositiveIntEnv("QMD_TEST_KNOB", 10, 100)).toBe(10);
    for (const bad of ["abc", "0", "-3", "", "2.5"]) {
      process.env.QMD_TEST_KNOB = bad;
      expect(readPositiveIntEnv("QMD_TEST_KNOB", 10, 100)).toBe(10);
    }
    process.env.QMD_TEST_KNOB = "32";
    expect(readPositiveIntEnv("QMD_TEST_KNOB", 10, 100)).toBe(32);
    process.env.QMD_TEST_KNOB = "5000";
    expect(readPositiveIntEnv("QMD_TEST_KNOB", 10, 100)).toBe(100);
  });
});

describe("parseRetryAfterMs", () => {
  test("seconds and HTTP dates; junk is undefined", () => {
    expect(parseRetryAfterMs("2")).toBe(2000);
    expect(parseRetryAfterMs("0")).toBe(0);
    expect(parseRetryAfterMs(null)).toBeUndefined();
    expect(parseRetryAfterMs("soon")).toBeUndefined();
    const now = Date.parse("2026-10-16T10:00:00Z");
    expect(parseRetryAfterMs("Fri, 16 Oct 2026 10:00:05 GMT", now)).toBe(5000);
  });
});

describe("withRetry", () => {
  test("retries 429 and 5xx, waits Retry-After, and returns the eventual result", async () => {
    const waits: number[] = [];
    let calls = 0;
    const result = await withRetry(async () => {
      calls++;
      if (calls === 1) throw new HttpStatusError(429, "slow down", 1500);
      if (calls === 2) throw new HttpStatusError(503, "busy");
      return "ok";
    }, { sleep: async ms => { waits.push(ms); }, random: () => 0.5, baseMs: 100, maxMs: 10_000 });
    expect(result).toBe("ok");
    expect(calls).toBe(3);
    expect(waits[0]).toBe(1500);
    // Second wait is exponential backoff (100 * 2^1) with jitter in [0.5x, 1.5x].
    expect(waits[1]).toBeGreaterThanOrEqual(100);
    expect(waits[1]).toBeLessThanOrEqual(300);
  });

  test("does not retry other 4xx; gives up after the retry budget with the last error", async () => {
    let calls = 0;
    await expect(withRetry(async () => { calls++; throw new HttpStatusError(401, "no key"); }, { sleep: async () => {} }))
      .rejects.toThrow(/no key/);
    expect(calls).toBe(1);

    calls = 0;
    await expect(withRetry(async () => { calls++; throw new HttpStatusError(500, "boom"); }, { retries: 2, sleep: async () => {} }))
      .rejects.toThrow(/boom/);
    expect(calls).toBe(3);
  });

  test("retries network errors and timeouts (status 0) through the same bounded budget", async () => {
    const seen: number[] = [];
    let calls = 0;
    const result = await withRetry(async () => {
      calls++;
      if (calls === 1) throw new TypeError("fetch failed", { cause: Object.assign(new Error("reset"), { code: "ECONNRESET" }) });
      if (calls === 2) throw new DOMException("The operation timed out", "TimeoutError");
      return "ok";
    }, { sleep: async () => {}, onRetry: status => { seen.push(status); } });
    expect(result).toBe("ok");
    expect(calls).toBe(3);
    expect(seen).toEqual([0, 0]);

    calls = 0;
    await expect(withRetry(async () => { calls++; throw new TypeError("fetch failed"); }, { sleep: async () => {} }))
      .rejects.toThrow(/fetch failed/);
    expect(calls).toBe(6);

    calls = 0;
    await expect(withRetry(async () => { calls++; throw new SyntaxError("bad json"); }, { sleep: async () => {} }))
      .rejects.toThrow(/bad json/);
    expect(calls).toBe(1);
  });

  test("backoff is capped at maxMs and onRetry sees each retried status", async () => {
    const waits: number[] = [];
    const seen: number[] = [];
    await expect(withRetry(async () => { throw new HttpStatusError(502, "bad gateway"); }, {
      retries: 6, baseMs: 1000, maxMs: 2000, random: () => 1,
      sleep: async ms => { waits.push(ms); },
      onRetry: status => { seen.push(status); },
    })).rejects.toThrow();
    expect(Math.max(...waits)).toBeLessThanOrEqual(2000);
    expect(seen).toEqual([502, 502, 502, 502, 502, 502]);
  });
});

describe("LatencyStats", () => {
  test("p50, p95 and max over recorded request latencies", () => {
    const stats = new LatencyStats();
    expect(stats.summary()).toEqual({ count: 0, p50: 0, p95: 0, max: 0 });
    for (let i = 1; i <= 100; i++) stats.record(i);
    const s = stats.summary();
    expect(s.count).toBe(100);
    expect(s.p50).toBe(50);
    expect(s.p95).toBe(95);
    expect(s.max).toBe(100);
  });
});

describe("AdaptiveLimit", () => {
  test("starts at the configured ceiling, halves on throttle, never below 1, climbs back on success", () => {
    const limit = new AdaptiveLimit(8);
    expect(limit.value).toBe(8);
    limit.recordThrottle();
    expect(limit.value).toBe(4);
    limit.recordThrottle();
    limit.recordThrottle();
    limit.recordThrottle();
    expect(limit.value).toBe(1);
    for (let i = 0; i < 200; i++) limit.recordSuccess(100);
    expect(limit.value).toBe(8);
  });

  test("a latency spike against the running baseline backs off", () => {
    const limit = new AdaptiveLimit(8);
    for (let i = 0; i < 20; i++) limit.recordSuccess(100);
    expect(limit.value).toBe(8);
    limit.recordSuccess(5000);
    expect(limit.value).toBeLessThan(8);
  });
});

describe("runPool", () => {
  test("no barrier: a slow job does not hold up the next ones, and the cap is honoured", async () => {
    const started: number[] = [];
    let inFlight = 0;
    let peak = 0;
    let n = 0;
    const next = async () => (n < 6 ? n++ : undefined);
    const finished: number[] = [];
    await runPool(next, async job => {
      started.push(job);
      inFlight++; peak = Math.max(peak, inFlight);
      await sleepFor(job === 0 ? 120 : 10);
      inFlight--;
      finished.push(job);
    }, { value: 3 });
    expect(peak).toBe(3);
    // Job 0 is slow: jobs 1..5 all finish before it does.
    expect(finished[finished.length - 1]).toBe(0);
    expect(finished).toHaveLength(6);
  });

  test("the next job is fetched while earlier ones are still in flight (pipelining)", async () => {
    let inFlightWhenFetched = 0;
    let inFlight = 0;
    let n = 0;
    const next = async () => {
      if (n > 0) inFlightWhenFetched = Math.max(inFlightWhenFetched, inFlight);
      await sleepFor(5);
      return n < 4 ? n++ : undefined;
    };
    await runPool(next, async () => { inFlight++; await sleepFor(40); inFlight--; }, { value: 4 });
    expect(inFlightWhenFetched).toBeGreaterThan(0);
  });

  test("a live limit change applies to jobs not yet started; shouldStop stops dispatch but drains in-flight", async () => {
    const limit = new AdaptiveLimit(4);
    let n = 0;
    let done = 0;
    let stop = false;
    await runPool(async () => (n < 20 ? n++ : undefined), async job => {
      await sleepFor(5);
      done++;
      if (job === 2) stop = true;
    }, limit, () => stop);
    expect(done).toBeLessThan(20);
    expect(done).toBeGreaterThanOrEqual(3);
  });

  test("a failing job rejects the run after in-flight jobs drain", async () => {
    let n = 0;
    let finished = 0;
    await expect(runPool(async () => (n < 4 ? n++ : undefined), async job => {
      await sleepFor(10);
      finished++;
      if (job === 1) throw new Error("job 1 failed");
    }, { value: 2 })).rejects.toThrow(/job 1 failed/);
    expect(finished).toBeGreaterThanOrEqual(2);
  });

  test("a throwing next() rejects only after in-flight jobs have finished", async () => {
    let n = 0;
    let finished = 0;
    await expect(runPool(async () => {
      if (n === 2) throw new Error("chunking failed");
      return n++;
    }, async () => {
      await sleepFor(30);
      finished++;
    }, { value: 4 })).rejects.toThrow(/chunking failed/);
    expect(finished).toBe(2);
  });
});
