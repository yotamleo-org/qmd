/**
 * embed-pool.ts - Building blocks for fast remote embedding: env knobs, retry
 * with backoff on 429/5xx, request-latency stats, an adaptive concurrency
 * limit and a barrier-free worker pool. Nothing here knows about SQLite or any
 * provider; store.ts and llm.ts wire it up.
 */

/** A positive integer from the environment, clamped to `max`; unset or junk gives `fallback`. */
export function readPositiveIntEnv(name: string, fallback: number, max: number): number {
  const raw = process.env[name];
  if (raw === undefined || !/^\d+$/.test(raw.trim())) return fallback;
  const n = Number(raw.trim());
  if (!Number.isSafeInteger(n) || n < 1) return fallback;
  return Math.min(n, max);
}

/** An HTTP error that keeps the status, so retry and throttle logic can branch on it. */
export class HttpStatusError extends Error {
  constructor(readonly status: number, message: string, readonly retryAfterMs?: number) {
    super(message);
    this.name = "HttpStatusError";
  }
}

/** `Retry-After` as milliseconds: delay-seconds or an HTTP date. Undefined when absent or unreadable. */
export function parseRetryAfterMs(value: string | null | undefined, now: number = Date.now()): number | undefined {
  if (value === null || value === undefined) return undefined;
  const trimmed = value.trim();
  if (/^\d+$/.test(trimmed)) return Number(trimmed) * 1000;
  const date = Date.parse(trimmed);
  return Number.isNaN(date) ? undefined : Math.max(0, date - now);
}

export function isRetryableStatus(status: number): boolean {
  return status === 429 || status >= 500;
}

const NETWORK_ERROR_CODES = new Set([
  "ECONNRESET", "ECONNREFUSED", "ETIMEDOUT", "EPIPE", "ENOTFOUND", "EAI_AGAIN", "ENETUNREACH",
  "UND_ERR_SOCKET", "UND_ERR_CONNECT_TIMEOUT", "UND_ERR_HEADERS_TIMEOUT", "UND_ERR_BODY_TIMEOUT",
]);

/** A failure with no HTTP answer: fetch rejected, the socket dropped or the request timed out. */
export function isNetworkError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  if (error.name === "TimeoutError" || error.name === "AbortError") return true;
  if (error instanceof TypeError && /fetch failed|network|terminated/i.test(error.message)) return true;
  const code = (error as { code?: unknown }).code ?? (error.cause as { code?: unknown } | undefined)?.code;
  return typeof code === "string" && NETWORK_ERROR_CODES.has(code);
}

export type RetryOptions = {
  /** Retries after the first attempt. */
  retries?: number;
  baseMs?: number;
  maxMs?: number;
  sleep?: (ms: number) => Promise<void>;
  /** Uniform [0, 1) source for jitter. */
  random?: () => number;
  /** Called with the status of each failure about to be retried (0 for a network error or timeout). */
  onRetry?: (status: number) => void;
};

const defaultSleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

/**
 * Run `fn`, retrying 429 and 5xx answers with exponential backoff and jitter
 * (honouring `Retry-After`). Network errors and timeouts (status 0) share the
 * same budget. Every other error is thrown at once.
 */
export async function withRetry<T>(fn: () => Promise<T>, options: RetryOptions = {}): Promise<T> {
  const retries = options.retries ?? 5;
  const baseMs = options.baseMs ?? 500;
  const maxMs = options.maxMs ?? 30_000;
  const sleep = options.sleep ?? defaultSleep;
  const random = options.random ?? Math.random;
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (error) {
      const http = error instanceof HttpStatusError ? error : undefined;
      const retryable = http ? isRetryableStatus(http.status) : isNetworkError(error);
      if (!retryable || attempt >= retries) throw error;
      options.onRetry?.(http?.status ?? 0);
      const backoff = Math.min(maxMs, baseMs * 2 ** attempt) * (0.5 + random());
      await sleep(http?.retryAfterMs !== undefined ? Math.min(http.retryAfterMs, maxMs) : Math.min(maxMs, backoff));
    }
  }
}

export type LatencySummary = { count: number; p50: number; p95: number; max: number };

/** Request latencies in ms, summarised as nearest-rank percentiles. */
export class LatencyStats {
  private samples: number[] = [];

  record(ms: number): void {
    this.samples.push(ms);
  }

  summary(): LatencySummary {
    if (this.samples.length === 0) return { count: 0, p50: 0, p95: 0, max: 0 };
    const sorted = [...this.samples].sort((a, b) => a - b);
    const rank = (p: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1))]!;
    return { count: sorted.length, p50: rank(0.5), p95: rank(0.95), max: sorted[sorted.length - 1]! };
  }
}

/**
 * Requests allowed in flight. Starts at the configured ceiling, halves on a
 * throttle (429/5xx) or a latency spike against the running baseline, and
 * climbs back one step per `value` clean answers.
 */
export class AdaptiveLimit {
  private current: number;
  private baseline = 0;
  private samples = 0;
  private cleanSincePenalty = 0;

  constructor(private readonly ceiling: number) {
    this.current = Math.max(1, ceiling);
  }

  get value(): number {
    return this.current;
  }

  recordThrottle(): void {
    this.current = Math.max(1, Math.floor(this.current / 2));
    this.cleanSincePenalty = 0;
  }

  recordSuccess(latencyMs: number): void {
    const spike = this.samples >= 5 && latencyMs > this.baseline * 5;
    if (!spike) {
      this.baseline = this.samples === 0 ? latencyMs : this.baseline * 0.9 + latencyMs * 0.1;
      this.samples++;
    }
    if (spike) {
      this.current = Math.max(1, Math.floor(this.current * 0.75));
      this.cleanSincePenalty = 0;
      return;
    }
    if (this.current < this.ceiling && ++this.cleanSincePenalty >= this.current) {
      this.current++;
      this.cleanSincePenalty = 0;
    }
  }
}

/**
 * Run jobs from `next` with at most `limit.value` in flight at once. There is
 * no barrier: a slot is refilled the moment any job ends, and `next` is
 * awaited while earlier jobs are still running, so preparing the following
 * job overlaps their requests. `shouldStop` ends dispatch; in-flight jobs
 * drain. The first job error is thrown once everything in flight has ended.
 */
export async function runPool<T>(
  next: () => Promise<T | undefined>,
  run: (job: T) => Promise<void>,
  limit: { readonly value: number },
  shouldStop: () => boolean = () => false,
): Promise<void> {
  let active = 0;
  let wake: (() => void) | undefined;
  let failure: { error: unknown } | undefined;
  const running = new Set<Promise<void>>();

  while (!shouldStop() && !failure) {
    while (active >= limit.value) {
      await new Promise<void>(resolve => { wake = resolve; });
    }
    if (shouldStop() || failure) break;
    let job: T | undefined;
    try {
      job = await next();
    } catch (error) {
      // Let requests already in flight finish (and commit) before rejecting.
      await Promise.all(running);
      throw error;
    }
    if (job === undefined) break;
    active++;
    const task: Promise<void> = run(job)
      .catch(error => { failure ??= { error }; })
      .finally(() => {
        active--;
        running.delete(task);
        const resolve = wake;
        wake = undefined;
        resolve?.();
      });
    running.add(task);
  }
  await Promise.all(running);
  if (failure) throw failure.error;
}
