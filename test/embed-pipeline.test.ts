/**
 * embed-pipeline.test.ts - generateEmbeddings with a remote embedder: requests
 * of the configured size stay in flight across documents, every chunk is
 * committed as it lands, and an interrupted run resumes. A stub LLM stands in
 * for the network.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore,
  insertContent,
  insertDocument,
  hashContent,
  generateEmbeddings,
  type Store,
} from "../src/store.js";
import type { LLM } from "../src/llm.js";

const REMOTE = "openai:test-embed";
const LOCAL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const sleepFor = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

let dir: string;
let store: Store;

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "qmd-embed-pipeline-"));
  store = createStore(join(dir, "i.sqlite"));
});

afterEach(async () => {
  store.close();
  await rm(dir, { recursive: true, force: true });
});

async function addDocs(count: number): Promise<void> {
  const now = new Date().toISOString();
  for (let i = 0; i < count; i++) {
    const body = `# Doc ${i}\n\nbody of document number ${i}`;
    const hash = await hashContent(body);
    insertContent(store.db, hash, body, now);
    insertDocument(store.db, "notes", `d${i}.md`, `d${i}.md`, hash, now, now);
  }
}

function vectorCount(model: string): number {
  return (store.db.prepare(`SELECT COUNT(*) AS n FROM content_vectors WHERE model = ?`).get(model) as { n: number }).n;
}

function poolLlm(model: string, tuning: { batchSize: number; concurrency: number } | undefined, behaviour: (texts: string[], call: number) => Promise<void> | void = () => {}) {
  const state = { calls: 0, inFlight: 0, peak: 0, sizes: [] as number[] };
  const llm = {
    embedModelName: model,
    ...(tuning ? { embedTuning: tuning } : {}),
    embedBatch: async (texts: string[]) => {
      const call = state.calls++;
      state.sizes.push(texts.length);
      state.inFlight++; state.peak = Math.max(state.peak, state.inFlight);
      try { await behaviour(texts, call); } finally { state.inFlight--; }
      return texts.map(() => ({ embedding: [0.1, 0.2, 0.3], model }));
    },
    embed: async (text: string) => {
      await behaviour([text], state.calls++);
      return { embedding: [0.1, 0.2, 0.3], model };
    },
    dispose: async () => {},
  } satisfies Partial<LLM> as LLM;
  return { llm, state };
}

describe("pipelined remote embed", () => {
  test("requests of the configured size run concurrently across documents, and every chunk is committed", async () => {
    await addDocs(20);
    const { llm, state } = poolLlm(REMOTE, { batchSize: 3, concurrency: 4 }, async () => { await sleepFor(15); });
    store.llm = llm;
    const progress: { chunksEmbedded: number; latency?: { count: number; p95: number } }[] = [];
    const result = await generateEmbeddings(store, { model: REMOTE, onProgress: p => progress.push(p) });
    expect(result.errors).toBe(0);
    expect(result.chunksEmbedded).toBe(20);
    expect(state.peak).toBeGreaterThan(1);
    expect(state.peak).toBeLessThanOrEqual(4);
    expect(Math.max(...state.sizes)).toBeLessThanOrEqual(3);
    expect(vectorCount(REMOTE)).toBe(20);
    expect(progress.at(-1)?.chunksEmbedded).toBe(20);
    expect(progress.at(-1)?.latency?.count).toBeGreaterThan(0);
  });

  test("a local embedder (no embedTuning) keeps one request in flight", async () => {
    await addDocs(6);
    const { llm, state } = poolLlm(LOCAL, undefined, async () => { await sleepFor(5); });
    store.llm = llm;
    await generateEmbeddings(store, { model: LOCAL });
    expect(state.peak).toBe(1);
    expect(vectorCount(LOCAL)).toBe(6);
  });

  test("an interrupted run keeps the chunks already committed, and the next run embeds only the rest", async () => {
    await addDocs(12);
    const first = poolLlm(REMOTE, { batchSize: 2, concurrency: 1 }, (_texts, call) => {
      if (call >= 3) throw new Error("connection reset");
    });
    store.llm = first.llm;
    const r1 = await generateEmbeddings(store, { model: REMOTE });
    const committed = vectorCount(REMOTE);
    expect(committed).toBeGreaterThan(0);
    expect(committed).toBeLessThan(12);
    expect(r1.errors).toBeGreaterThan(0);

    const second = poolLlm(REMOTE, { batchSize: 2, concurrency: 2 });
    store.llm = second.llm;
    const r2 = await generateEmbeddings(store, { model: REMOTE });
    expect(r2.errors).toBe(0);
    expect(second.state.sizes.reduce((a, b) => a + b, 0)).toBe(12 - committed);
    expect(vectorCount(REMOTE)).toBe(12);
  });
});
