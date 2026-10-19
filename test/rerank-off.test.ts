/**
 * rerank-off.test.ts - QMD_RERANK=off never loads the reranker, and a hung
 * rerank is remembered per process instead of killing the daemon each time.
 * Mocks only: no model is loaded and no native code runs.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as llmModule from "../src/llm.js";
import { resetLlamaPoisonedForTests } from "../src/llm.js";
import {
  createStore,
  structuredSearch,
  insertContent,
  insertDocument,
  hashContent,
  type Store,
} from "../src/store.js";
import { exitIfRerankHung } from "../src/mcp/server.js";

let testDir: string;
let store: Store;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-rerank-off-"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

beforeEach(() => {
  store = createStore(join(testDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`));
  process.env.QMD_RERANK_TIMEOUT_MS = "50";
});

afterEach(() => {
  delete process.env.QMD_RERANK;
  delete process.env.QMD_RERANK_TIMEOUT_MS;
  resetLlamaPoisonedForTests();
  vi.restoreAllMocks();
  store.close();
});

async function insertDoc(path: string, body: string): Promise<void> {
  const now = new Date().toISOString();
  const hash = await hashContent(body);
  insertContent(store.db, hash, body, now);
  insertDocument(store.db, "notes", path, path, hash, now, now);
}

function hangingLlm() {
  return {
    rerank: vi.fn(() => new Promise<never>(() => {})),
    rerankModelName: "hf:example/rerank/hang.gguf",
    poison: vi.fn(() => { /* the real poison() sets the process flag */ }),
  };
}

describe("QMD_RERANK=off", () => {
  for (const value of ["off", "OFF", "0", "false"]) {
    test(`QMD_RERANK=${value}: structuredSearch never touches the reranker`, async () => {
      await insertDoc("a.md", "# A\n\noff keyword");
      await insertDoc("b.md", "# B\n\noff keyword");
      const llm = hangingLlm();
      const getLlm = vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(llm as any);
      process.env.QMD_RERANK = value;

      const results = await structuredSearch(store, [{ type: "lex", query: "off keyword" }], {});

      expect(results.length).toBe(2);
      expect(llm.rerank).not.toHaveBeenCalled();
      expect(getLlm).not.toHaveBeenCalled();
      expect(results.every(r => r.rerankTimedOut === undefined)).toBe(true);
    });
  }

  test("QMD_RERANK unset or on still reranks", async () => {
    await insertDoc("a.md", "# A\n\non keyword");
    const llm = hangingLlm();
    vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(llm as any);
    process.env.QMD_RERANK = "on";

    await structuredSearch(store, [{ type: "lex", query: "on keyword" }], {});

    expect(llm.rerank).toHaveBeenCalledTimes(1);
  });
});

describe("hung rerank is remembered per process", () => {
  test("after one hang, later searches skip rerank instead of calling it again", async () => {
    await insertDoc("a.md", "# A\n\nhang keyword");
    const llm = hangingLlm();
    // poison() flips the process-wide flag exactly as LlamaCpp.poison does.
    llm.poison.mockImplementation(() => { llmModule.markLlamaPoisonedForTests(); });
    vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(llm as any);

    const first = await structuredSearch(store, [{ type: "lex", query: "hang keyword" }], {});
    expect(first.every(r => r.rerankTimedOut === true)).toBe(true);
    expect(llm.rerank).toHaveBeenCalledTimes(1);

    const second = await structuredSearch(store, [{ type: "lex", query: "hang keyword" }], {});
    expect(second.length).toBe(1);
    expect(llm.rerank).toHaveBeenCalledTimes(1);
  });

  const MODEL_ENV = ["QMD_CONFIG_DIR", "QMD_EMBED_MODEL", "QMD_GENERATE_MODEL"] as const;

  /** Runs exitIfRerankHung twice with the given model env and returns how many kill timers it armed. */
  function armedKills(embed: string, generate: string): number {
    const saved = MODEL_ENV.map(k => process.env[k]);
    process.env.QMD_CONFIG_DIR = testDir;
    process.env.QMD_EMBED_MODEL = embed;
    process.env.QMD_GENERATE_MODEL = generate;
    const armed = vi.spyOn(globalThis, "setTimeout").mockImplementation((() => 0) as any);
    vi.spyOn(console, "error").mockImplementation(() => {});
    llmModule.markLlamaPoisonedForTests();
    try {
      exitIfRerankHung();
      exitIfRerankHung();
      return armed.mock.calls.length;
    } finally {
      MODEL_ENV.forEach((k, i) => {
        if (saved[i] === undefined) delete process.env[k];
        else process.env[k] = saved[i];
      });
    }
  }

  test("exitIfRerankHung does not kill a daemon whose embed and generate models are remote", () => {
    expect(armedKills("openai:embed", "openai:generate")).toBe(0);
  });

  test("exitIfRerankHung trusts the store's effective models over the raw config", () => {
    const local = { internal: { llm: { embedModelName: "hf:example/embed/local.gguf", generateModelName: "openai:generate" } } } as any;
    const remote = { internal: { llm: { embedModelName: "openai:embed", generateModelName: "openai:generate" } } } as any;
    const saved = process.env.QMD_EMBED_MODEL;
    process.env.QMD_EMBED_MODEL = "openai:embed";
    const armed = vi.spyOn(globalThis, "setTimeout").mockImplementation((() => 0) as any);
    vi.spyOn(console, "error").mockImplementation(() => {});
    llmModule.markLlamaPoisonedForTests();
    try {
      exitIfRerankHung(remote);
      expect(armed).not.toHaveBeenCalled();
      exitIfRerankHung(local);
      expect(armed).toHaveBeenCalledTimes(1);
    } finally {
      if (saved === undefined) delete process.env.QMD_EMBED_MODEL;
      else process.env.QMD_EMBED_MODEL = saved;
    }
  });

  test("exitIfRerankHung still kills a daemon with a local embed model, which the poisoned instance refuses", () => {
    expect(armedKills("hf:example/embed/local.gguf", "openai:generate")).toBe(2);
  });
});
