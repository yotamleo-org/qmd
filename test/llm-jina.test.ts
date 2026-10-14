/**
 * llm-jina.test.ts - Jina on top of the OpenAI-compatible backend: key names,
 * the embedding `task`, a remote-rerank-only mix and the hang/poison paths.
 * Global fetch is stubbed: no network, no model load.
 */

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CompositeLLM, LlamaCpp, OpenAiLLM, RerankTimeoutError, createLlm, isJinaEndpoint, isLlamaPoisoned } from "../src/llm.js";
import { createStore, rerank } from "../src/store.js";

const SECRET = "jina_SECRET_KEY_FOR_TESTS";
const JINA = "https://api.jina.ai/v1";

type Call = { url: string; init: RequestInit; body: any };

function stubFetch(handler: (call: Call) => unknown): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit = {}) => {
    const call: Call = { url, init, body: init.body ? JSON.parse(String(init.body)) : undefined };
    calls.push(call);
    return new Response(JSON.stringify(handler(call)), { status: 200, headers: { "content-type": "application/json" } });
  }));
  return calls;
}

const jinaRerankOnly = {
  rerank: "openai:jina-reranker-v2-base-multilingual",
  openaiBaseUrl: JINA,
};

const savedEnv = { ...process.env };
beforeEach(() => {
  for (const k of ["QMD_OPENAI_BASE_URL", "QMD_OPENAI_API_KEY", "JINA_API_KEY", "JINA_API"]) delete process.env[k];
});
afterEach(() => {
  vi.unstubAllGlobals();
  process.env = { ...savedEnv };
});

describe("Jina key", () => {
  test("JINA_API_KEY wins, JINA_API is the fallback, and only a jina.ai host gets either", async () => {
    const calls = stubFetch(() => ({ results: [{ index: 0, relevance_score: 0.5 }] }));
    process.env.JINA_API = "fallback";
    await createLlm(jinaRerankOnly).rerank("q", [{ file: "f", text: "t" }]);
    expect((calls[0]!.init.headers as Record<string, string>).authorization).toBe("Bearer fallback");

    process.env.JINA_API_KEY = SECRET;
    await createLlm(jinaRerankOnly).rerank("q", [{ file: "f", text: "t" }]);
    expect((calls[1]!.init.headers as Record<string, string>).authorization).toBe(`Bearer ${SECRET}`);

    // The key must never travel to another host.
    await createLlm({ rerank: "openai:r", openaiBaseUrl: "http://127.0.0.1:4100/v1" }).rerank("q", [{ file: "f", text: "t" }]);
    expect((calls[2]!.init.headers as Record<string, string>).authorization).toBeUndefined();
  });

  test("isJinaEndpoint matches jina.ai hosts only", () => {
    expect(isJinaEndpoint("https://api.jina.ai/v1")).toBe(true);
    expect(isJinaEndpoint("https://jina.ai.evil.example/v1")).toBe(false);
    expect(isJinaEndpoint("not a url")).toBe(false);
  });
});

describe("remote rerank beside local embeddings", () => {
  test("only rerank goes remote; embed and generate stay local and nothing is loaded", async () => {
    process.env.JINA_API_KEY = SECRET;
    const calls = stubFetch(({ url }) => {
      expect(url).toBe(`${JINA}/rerank`);
      return { results: [{ index: 1, relevance_score: 0.9 }, { index: 0, relevance_score: 0.1 }] };
    });
    const llm = createLlm(jinaRerankOnly);
    expect(llm).toBeInstanceOf(CompositeLLM);
    const mixed = llm as CompositeLLM;
    expect(mixed.rerankBackend).toBeInstanceOf(OpenAiLLM);
    expect(mixed.embedBackend).toBeInstanceOf(LlamaCpp);
    expect(llm.embedModelName).not.toMatch(/^openai:/);
    expect(llm.rerankModelName).toBe("openai:jina-reranker-v2-base-multilingual");

    const out = await llm.rerank("q", [{ file: "a", text: "x" }, { file: "b", text: "y" }]);
    expect(out.results[0]).toMatchObject({ file: "b", score: 0.9 });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.body.model).toBe("jina-reranker-v2-base-multilingual");
  });

  test("a remote reranker has no native state: no load step and nothing to poison", () => {
    const llm = createLlm(jinaRerankOnly);
    expect(llm.prepareRerank).toBeUndefined();
    expect(llm.poison).toBeUndefined();
  });

  test("an unconfigured process sends nothing and stays fully local", () => {
    const calls = stubFetch(() => ({}));
    expect(createLlm({})).toBeInstanceOf(LlamaCpp);
    expect(calls).toHaveLength(0);
  });
});

describe("a hung remote rerank", () => {
  test("times out into the RRF fallback and never poisons the process", async () => {
    process.env.QMD_RERANK_TIMEOUT_MS = "50";
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {}))); // never answers
    const store = createStore(join(await mkdtemp(join(tmpdir(), "qmd-jina-hang-")), "i.sqlite"));
    try {
      const llm = createLlm(jinaRerankOnly);
      await expect(rerank("q", [{ file: "a", text: "x" }], llm.rerankModelName, store.db, undefined, llm))
        .rejects.toBeInstanceOf(RerankTimeoutError);
      // exitIfRerankHung keys off this flag: a remote hang must not kill the daemon.
      expect(isLlamaPoisoned()).toBe(false);
    } finally {
      store.close();
      delete process.env.QMD_RERANK_TIMEOUT_MS;
    }
  });
});

describe("Jina embeddings", () => {
  test("sends the retrieval task for documents and queries, text stays raw", async () => {
    process.env.JINA_API_KEY = SECRET;
    const calls = stubFetch(({ body }) => ({
      data: body.input.map((_: string, index: number) => ({ index, embedding: [0.1, 0.2] })),
    }));
    const llm = new OpenAiLLM({ baseUrl: JINA, embedModel: "openai:jina-embeddings-v3", roles: ["embed"] });
    await llm.embedBatch(["doc"]);
    await llm.embed("query", { isQuery: true });
    expect(calls.map(c => c.body.task)).toEqual(["retrieval.passage", "retrieval.query"]);
    expect(calls[0]!.body.input).toEqual(["doc"]);
    expect(calls[0]!.body.model).toBe("jina-embeddings-v3");
  });

  test("a non-Jina endpoint gets no task field", async () => {
    const calls = stubFetch(({ body }) => ({
      data: body.input.map((_: string, index: number) => ({ index, embedding: [1] })),
    }));
    await new OpenAiLLM({ baseUrl: "http://127.0.0.1:4100/v1", embedModel: "openai:e", roles: ["embed"] }).embedBatch(["a"]);
    expect(calls[0]!.body.task).toBeUndefined();
  });
});
