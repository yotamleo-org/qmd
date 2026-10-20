/**
 * rerank-off.test.ts - QMD_RERANK=off never loads the reranker, and a hung
 * rerank is remembered per process instead of killing the daemon each time.
 * Mocks only: no model is loaded and no native code runs.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { existsSync, mkdirSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as llmModule from "../src/llm.js";
import { resetLlamaPoisonedForTests } from "../src/llm.js";
import {
  createStore,
  structuredSearch,
  hybridQuery,
  insertContent,
  insertDocument,
  hashContent,
  type Store,
} from "../src/store.js";
import { exitIfRerankHung } from "../src/mcp/server.js";

let testDir: string;
let store: Store;
let savedCacheHome: string | undefined;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-rerank-off-"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

beforeEach(() => {
  store = createStore(join(testDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`));
  process.env.QMD_RERANK_TIMEOUT_MS = "50";
  savedCacheHome = process.env.XDG_CACHE_HOME;
  process.env.XDG_CACHE_HOME = join(testDir, `cache-${Math.random().toString(36).slice(2)}`);
  llmModule.restartForTests();
});

afterEach(() => {
  delete process.env.QMD_RERANK;
  delete process.env.QMD_RERANK_TIMEOUT_MS;
  // The snapshot is module state: leave it empty for whatever test file runs next in this process.
  rmSync(process.env.XDG_CACHE_HOME!, { recursive: true, force: true });
  llmModule.restartForTests();
  if (savedCacheHome === undefined) delete process.env.XDG_CACHE_HOME;
  else process.env.XDG_CACHE_HOME = savedCacheHome;
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

describe("hybridQuery", () => {
  // Expansion is stubbed (it would load a model); no vector index skips the vector leg.
  async function seed(): Promise<void> {
    vi.spyOn(store, "expandQuery").mockResolvedValue([]);
    await insertDoc("a.md", "# Alpha\n\nzebrafish keyword zebrafish zebrafish");
    await insertDoc("b.md", "# Beta\n\nzebrafish keyword");
  }

  test("QMD_RERANK=off: hybridQuery never calls store.rerank", async () => {
    await seed();
    const rerank = vi.spyOn(store, "rerank");
    process.env.QMD_RERANK = "off";

    const results = await hybridQuery(store, "zebrafish keyword", {});

    expect(results.length).toBe(2);
    expect(rerank).not.toHaveBeenCalled();
    expect(results.every(r => r.rerankTimedOut === undefined)).toBe(true);
  });

  test("hybridQuery calls store.rerank when reranking is on", async () => {
    await seed();
    const rerank = vi.spyOn(store, "rerank").mockResolvedValue([{ file: "a.md", score: 1 }, { file: "b.md", score: 0.5 }]);

    await hybridQuery(store, "zebrafish keyword", {});

    expect(rerank).toHaveBeenCalledTimes(1);
  });

  test("hybridQuery skips store.rerank once the process is poisoned", async () => {
    await seed();
    const rerank = vi.spyOn(store, "rerank");
    llmModule.markLlamaPoisonedForTests();

    const results = await hybridQuery(store, "zebrafish keyword", {});

    expect(results.length).toBe(2);
    expect(rerank).not.toHaveBeenCalled();
  });

  test("a rerank refused because a concurrent search poisoned the process falls back to RRF", async () => {
    await seed();
    vi.spyOn(store, "rerank").mockImplementation(async () => {
      llmModule.markLlamaPoisonedForTests();
      throw new Error("LlamaCpp instance is poisoned (rerank timed out); restart the process");
    });

    const results = await hybridQuery(store, "zebrafish keyword", {});

    expect(results.length).toBe(2);
    expect(results.every(r => r.rerankTimedOut === true)).toBe(true);
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
    const saved = MODEL_ENV.map(k => process.env[k]);
    process.env.QMD_CONFIG_DIR = testDir;
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
      MODEL_ENV.forEach((k, i) => {
        if (saved[i] === undefined) delete process.env[k];
        else process.env[k] = saved[i];
      });
    }
  });

  test("exitIfRerankHung still kills a daemon with a local embed model, which the poisoned instance refuses", () => {
    expect(armedKills("hf:example/embed/local.gguf", "openai:generate")).toBe(2);
  });
});

describe("a daemon with local models restarts at most once for a hung rerank", () => {
  const MODEL_ENV = ["QMD_CONFIG_DIR", "QMD_EMBED_MODEL", "QMD_GENERATE_MODEL"] as const;
  const markerPath = () => join(process.env.XDG_CACHE_HOME!, "qmd", "rerank-hung");

  /** Kills the poisoned local-model daemon once (kill timer mocked), then simulates the supervisor's restart. */
  function hangAndRestart(): void {
    const saved = MODEL_ENV.map(k => process.env[k]);
    process.env.QMD_CONFIG_DIR = testDir;
    process.env.QMD_EMBED_MODEL = "hf:example/embed/local.gguf";
    process.env.QMD_GENERATE_MODEL = "openai:generate";
    vi.spyOn(globalThis, "setTimeout").mockImplementation((() => 0) as any);
    vi.spyOn(console, "error").mockImplementation(() => {});
    llmModule.markLlamaPoisonedForTests();
    try {
      exitIfRerankHung();
    } finally {
      MODEL_ENV.forEach((k, i) => {
        if (saved[i] === undefined) delete process.env[k];
        else process.env[k] = saved[i];
      });
    }
    vi.restoreAllMocks();
    resetLlamaPoisonedForTests();
    llmModule.restartForTests();
  }

  test("the restarted process serves unreranked instead of hanging and being killed again", async () => {
    await insertDoc("a.md", "# A\n\nrestart keyword");
    hangAndRestart();
    const llm = hangingLlm();
    vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(llm as any);

    const results = await structuredSearch(store, [{ type: "lex", query: "restart keyword" }], {});

    expect(results.length).toBe(1);
    expect(llm.rerank).not.toHaveBeenCalled();
  });

  test("a hang recorded long ago is forgotten, so rerank is tried again", async () => {
    await insertDoc("a.md", "# A\n\nstale keyword");
    hangAndRestart();
    expect(existsSync(markerPath())).toBe(true);
    const old = new Date(Date.now() - 2 * 60 * 60 * 1000);
    utimesSync(markerPath(), old, old);
    llmModule.restartForTests();
    const llm = hangingLlm();
    vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(llm as any);

    await structuredSearch(store, [{ type: "lex", query: "stale keyword" }], {});

    expect(llm.rerank).toHaveBeenCalledTimes(1);
  });

  test("a marker dated in the future is ignored, so a clock step back cannot keep rerank off", async () => {
    await insertDoc("a.md", "# A\n\nfuture keyword");
    hangAndRestart();
    const future = new Date(Date.now() + 10 * 60 * 60 * 1000);
    utimesSync(markerPath(), future, future);
    llmModule.restartForTests();
    const llm = hangingLlm();
    vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(llm as any);

    await structuredSearch(store, [{ type: "lex", query: "future keyword" }], {});

    expect(llm.rerank).toHaveBeenCalledTimes(1);
  });

  test("a marker another process writes while this one runs does not switch its rerank off", async () => {
    await insertDoc("a.md", "# A\n\nrunning keyword");
    mkdirSync(join(process.env.XDG_CACHE_HOME!, "qmd"), { recursive: true });
    writeFileSync(markerPath(), String(Date.now()));
    const llm = hangingLlm();
    vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(llm as any);

    await structuredSearch(store, [{ type: "lex", query: "running keyword" }], {});

    expect(llm.rerank).toHaveBeenCalledTimes(1);
  });

  test("an unwritable cache dir still kills the daemon and the next search reranks as before", async () => {
    await insertDoc("a.md", "# A\n\nunwritable keyword");
    process.env.XDG_CACHE_HOME = join(testDir, "not-a-dir");
    writeFileSync(process.env.XDG_CACHE_HOME, "a file where the cache dir should be");
    expect(() => hangAndRestart()).not.toThrow();
    const llm = hangingLlm();
    vi.spyOn(llmModule, "getDefaultLlamaCpp").mockReturnValue(llm as any);

    await structuredSearch(store, [{ type: "lex", query: "unwritable keyword" }], {});

    expect(llm.rerank).toHaveBeenCalledTimes(1);
  });

  test("a daemon with remote models records no hang", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    llmModule.markLlamaPoisonedForTests();
    exitIfRerankHung({ internal: { llm: { embedModelName: "openai:embed", generateModelName: "openai:generate" } } } as any);
    expect(existsSync(markerPath())).toBe(false);
  });
});
