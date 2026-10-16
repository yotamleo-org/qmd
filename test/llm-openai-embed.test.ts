/**
 * llm-openai-embed.test.ts - Remote embed tuning on the OpenAI-compatible
 * backend: env knobs, retry on 429/5xx, and the OpenRouter provider pin.
 * Every HTTP call goes through a stubbed global fetch.
 */

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import { OpenAiLLM } from "../src/llm.js";

type Call = { url: string; init: RequestInit; body: any };

function stubFetch(handler: (call: Call) => unknown | Promise<unknown>): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit = {}) => {
    const call: Call = { url, init, body: init.body ? JSON.parse(String(init.body)) : undefined };
    calls.push(call);
    const result = await handler(call);
    if (result instanceof Response) return result;
    return new Response(JSON.stringify(result), { status: 200, headers: { "content-type": "application/json" } });
  }));
  return calls;
}

const config = {
  baseUrl: "http://127.0.0.1:4100/v1/",
  apiKey: "test-key",
  embedModel: "openai:text-embedding-v4",
  generateModel: "openai:general",
  rerankModel: "openai:auto",
};

const echo = ({ body }: Call) => ({
  data: body.input.map((text: string, index: number) => ({ index, embedding: [text.length] })),
});

describe("OpenAiLLM remote embed tuning", () => {
  const savedEnv = { ...process.env };

  beforeEach(() => {
    for (const name of ["QMD_OPENAI_BASE_URL", "QMD_OPENAI_API_KEY", "QMD_OPENAI_EMBED_BATCH_SIZE", "QMD_OPENAI_EMBED_CONCURRENCY", "QMD_OPENAI_PROVIDER"]) {
      delete process.env[name];
    }
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    process.env = { ...savedEnv };
  });

  test("defaults stay conservative: batch 10, concurrency 4", () => {
    expect(new OpenAiLLM(config).embedTuning).toEqual({ batchSize: 10, concurrency: 4 });
  });

  test("QMD_OPENAI_EMBED_BATCH_SIZE sets the request size and QMD_OPENAI_EMBED_CONCURRENCY the requests in flight", async () => {
    process.env.QMD_OPENAI_EMBED_BATCH_SIZE = "3";
    process.env.QMD_OPENAI_EMBED_CONCURRENCY = "2";
    let inFlight = 0;
    let peak = 0;
    const calls = stubFetch(async call => {
      inFlight++; peak = Math.max(peak, inFlight);
      await new Promise(resolve => setTimeout(resolve, 15));
      inFlight--;
      return echo(call);
    });
    const llm = new OpenAiLLM(config);
    expect(llm.embedTuning).toEqual({ batchSize: 3, concurrency: 2 });
    const results = await llm.embedBatch(Array.from({ length: 12 }, (_, i) => "y".repeat(i + 1)));
    expect(calls.map(c => c.body.input.length)).toEqual([3, 3, 3, 3]);
    expect(peak).toBe(2);
    results.forEach((r, i) => expect(r?.embedding[0]).toBe(i + 1));
  });

  test("a 429 with Retry-After is retried and reported; a 400 is not retried", async () => {
    let attempts = 0;
    const calls = stubFetch(call => {
      attempts++;
      if (attempts <= 2) return new Response("rate limited", { status: 429, headers: { "retry-after": "0" } });
      return echo(call);
    });
    const throttled: number[] = [];
    const llm = new OpenAiLLM(config);
    const results = await llm.embedBatch(["abc"], { onThrottle: status => { throttled.push(status); } });
    expect(results[0]?.embedding[0]).toBe(3);
    expect(calls).toHaveLength(3);
    expect(throttled).toEqual([429, 429]);

    const bad = stubFetch(() => new Response("bad input", { status: 400 }));
    await expect(llm.embedBatch(["abc"])).rejects.toThrow(/400/);
    expect(bad).toHaveLength(1);
  });

  test("QMD_OPENAI_PROVIDER pins the provider with no fallback, only on OpenRouter and only for bulk embeds", async () => {
    process.env.QMD_OPENAI_PROVIDER = "DeepInfra";
    const calls = stubFetch(echo);

    await new OpenAiLLM({ ...config, baseUrl: "https://openrouter.ai/api/v1" }).embedBatch(["a"]);
    expect(calls[0]!.body.provider).toEqual({ order: ["DeepInfra"], allow_fallbacks: false });

    await new OpenAiLLM({ ...config, baseUrl: "https://openrouter.ai/api/v1" }).embedBatch(["a"], { isQuery: true });
    expect(calls[1]!.body.provider).toBeUndefined();

    await new OpenAiLLM({ ...config, baseUrl: "https://api.deepinfra.com/v1/openai" }).embedBatch(["a"]);
    await new OpenAiLLM(config).embedBatch(["a"]);
    await new OpenAiLLM({ ...config, baseUrl: "https://api.jina.ai/v1" }).embedBatch(["a"]);
    expect(calls[2]!.body.provider).toBeUndefined();
    expect(calls[3]!.body.provider).toBeUndefined();
    expect(calls[4]!.body.provider).toBeUndefined();
  });

  test("without QMD_OPENAI_PROVIDER an OpenRouter body is unchanged", async () => {
    const calls = stubFetch(echo);
    await new OpenAiLLM({ ...config, baseUrl: "https://openrouter.ai/api/v1" }).embedBatch(["a"]);
    expect(calls[0]!.body.provider).toBeUndefined();
  });
});
