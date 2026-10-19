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
  vi.useRealTimers();
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

  test("exitIfRerankHung does not kill a daemon whose embed and generate models are remote", () => {
    vi.useFakeTimers();
    vi.stubEnv("QMD_CONFIG_DIR", testDir);
    vi.stubEnv("QMD_EMBED_MODEL", "openai:embed");
    vi.stubEnv("QMD_GENERATE_MODEL", "openai:generate");
    const kill = vi.spyOn(process, "kill").mockImplementation((() => true) as any);
    vi.spyOn(console, "error").mockImplementation(() => {});
    llmModule.markLlamaPoisonedForTests();

    exitIfRerankHung();
    exitIfRerankHung();
    vi.advanceTimersByTime(5000);

    expect(kill).not.toHaveBeenCalled();
    vi.unstubAllEnvs();
  });

  test("exitIfRerankHung still kills a daemon with a local embed model, which the poisoned instance refuses", () => {
    vi.useFakeTimers();
    vi.stubEnv("QMD_CONFIG_DIR", testDir);
    vi.stubEnv("QMD_EMBED_MODEL", "hf:example/embed/local.gguf");
    vi.stubEnv("QMD_GENERATE_MODEL", "openai:generate");
    const kill = vi.spyOn(process, "kill").mockImplementation((() => true) as any);
    vi.spyOn(process, "exit").mockImplementation((() => undefined) as any);
    vi.spyOn(console, "error").mockImplementation(() => {});
    llmModule.markLlamaPoisonedForTests();

    exitIfRerankHung();
    vi.advanceTimersByTime(5000);

    expect(kill).toHaveBeenCalledTimes(1);
    vi.unstubAllEnvs();
  });
});
