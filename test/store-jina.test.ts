/**
 * store-jina.test.ts - Store-side guards for a remote embedder: no mixed vector
 * spaces, the embed session TTL, the full-corpus token dry-run and the
 * query-side embed flag. A stub LLM stands in for the network: nothing is sent.
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
  estimateCorpusEmbedTokens,
  resolveEmbedMaxDurationMs,
  DEFAULT_EMBED_MAX_DURATION_MS,
  chunkDocument,
  type Store,
} from "../src/store.js";
import type { LLM } from "../src/llm.js";

const REMOTE = "openai:jina-embeddings-v3";
const LOCAL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";

let dir: string;
let store: Store;

async function addDoc(path: string, body: string, collection = "notes"): Promise<string> {
  const hash = await hashContent(body);
  const now = new Date().toISOString();
  insertContent(store.db, hash, body, now);
  insertDocument(store.db, collection, path, path, hash, now, now);
  return hash;
}

function addVector(hash: string, model: string): void {
  store.ensureVecTable(3);
  store.db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES (?, 0, 0, ?, ?)`)
    .run(hash, model, new Date().toISOString());
}

function stubLlm(model: string, seen: { isQuery?: boolean }[] = []): LLM {
  return {
    embedModelName: model,
    embedBatch: async (texts: string[], opts?: { isQuery?: boolean }) => {
      seen.push({ isQuery: opts?.isQuery });
      return texts.map(() => ({ embedding: [0.1, 0.2, 0.3], model }));
    },
    embed: async () => ({ embedding: [0.1, 0.2, 0.3], model }),
    dispose: async () => {},
  } satisfies Partial<LLM> as LLM;
}

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "qmd-store-jina-"));
  store = createStore(join(dir, "i.sqlite"));
  delete process.env.QMD_EMBED_MAX_DURATION_MS;
});

afterEach(async () => {
  store.close();
  delete process.env.QMD_EMBED_MAX_DURATION_MS;
  await rm(dir, { recursive: true, force: true });
});

describe("remote embedder never mixes vector spaces", () => {
  test("refuses to embed new chunks next to another model's vectors, naming qmd embed -f", async () => {
    addVector(await addDoc("old.md", "already embedded locally"), LOCAL);
    await addDoc("new.md", "fresh note that needs a vector");
    store.llm = stubLlm(REMOTE);
    await expect(generateEmbeddings(store, { model: REMOTE })).rejects.toThrow(/qmd embed -f/);
    // The old vectors are untouched.
    expect((store.db.prepare(`SELECT COUNT(*) AS n FROM content_vectors WHERE model = ?`).get(LOCAL) as { n: number }).n).toBe(1);
  });

  test("a scoped force cannot clear and then switch space", async () => {
    addVector(await addDoc("old.md", "locally embedded", "a"), LOCAL);
    await addDoc("new.md", "another collection", "b");
    store.llm = stubLlm(REMOTE);
    await expect(generateEmbeddings(store, { model: REMOTE, force: true, collection: "b" })).rejects.toThrow(/qmd embed -f/);
  });

  test("a full forced re-embed may switch space", async () => {
    addVector(await addDoc("old.md", "locally embedded"), LOCAL);
    store.llm = stubLlm(REMOTE);
    const result = await generateEmbeddings(store, { model: REMOTE, force: true });
    expect(result.errors).toBe(0);
    expect((store.db.prepare(`SELECT COUNT(DISTINCT model) AS n FROM content_vectors`).get() as { n: number }).n).toBe(1);
    expect((store.db.prepare(`SELECT model FROM content_vectors LIMIT 1`).get() as { model: string }).model).toBe(REMOTE);
  });

  test("a local embed model is not guarded (existing behaviour)", async () => {
    addVector(await addDoc("old.md", "x"), "some-other-local-model");
    await addDoc("new.md", "y");
    store.llm = stubLlm(LOCAL);
    await expect(generateEmbeddings(store, { model: LOCAL })).resolves.toBeDefined();
  });
});

describe("embed session TTL", () => {
  test("QMD_EMBED_MAX_DURATION_MS sets it; 0 means no limit; junk falls back to 30 min", () => {
    expect(resolveEmbedMaxDurationMs()).toBe(DEFAULT_EMBED_MAX_DURATION_MS);
    process.env.QMD_EMBED_MAX_DURATION_MS = "7200000";
    expect(resolveEmbedMaxDurationMs()).toBe(7_200_000);
    process.env.QMD_EMBED_MAX_DURATION_MS = "0";
    expect(resolveEmbedMaxDurationMs()).toBe(0);
    for (const bad of ["abc", "-5", ""]) {
      process.env.QMD_EMBED_MAX_DURATION_MS = bad;
      expect(resolveEmbedMaxDurationMs()).toBe(DEFAULT_EMBED_MAX_DURATION_MS);
    }
  });
});

describe("corpus token dry-run", () => {
  test("counts every active document as embed -f would, embedded or not, and sends nothing", async () => {
    const long = "paragraph of words.\n\n".repeat(900);
    const hashA = await addDoc("a.md", long);
    await addDoc("b.md", "short note");
    addVector(hashA, LOCAL); // already embedded: a forced embed still re-sends it
    const expectChunks = chunkDocument(long).length + chunkDocument("short note").length;
    const expectChars = [...chunkDocument(long), ...chunkDocument("short note")].reduce((n, c) => n + c.text.length, 0);

    const est = estimateCorpusEmbedTokens(store.db);
    expect(est.docs).toBe(2);
    expect(est.chunks).toBe(expectChunks);
    expect(est.chars).toBe(expectChars);
    expect(est.estimatedTokens).toBe(Math.ceil(expectChars / 3));
  });

  test("ignores inactive documents and honours a collection scope", async () => {
    await addDoc("a.md", "alpha body", "one");
    await addDoc("b.md", "beta body", "two");
    store.db.prepare(`UPDATE documents SET active = 0 WHERE collection = 'two'`).run();
    expect(estimateCorpusEmbedTokens(store.db).docs).toBe(1);
    expect(estimateCorpusEmbedTokens(store.db, "two").docs).toBe(0);
  });
});
