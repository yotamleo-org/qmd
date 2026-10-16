/**
 * Test preload file to ensure proper cleanup of native resources.
 *
 * Uses bun:test afterAll to dispose of llama.cpp Metal resources before
 * the process exits — necessary on darwin to avoid the upstream rsets
 * destructor assertion (ggml-org/llama.cpp#22593, fix open as #22595).
 *
 * The runner-level mitigation `GGML_METAL_NO_RESIDENCY=1` must be set
 * BEFORE bun/node starts (libggml-metal reads it via libc getenv at
 * module load). Bun does not propagate `process.env` writes to libc
 * setenv, so setting it from here would be a no-op for the native
 * binding. The env var is injected by:
 *   - bin/qmd for production CLI runs
 *   - scripts/test-all.mjs for `npm test`
 *   - package.json test:bun / test:unit scripts for direct invocation
 * See CLAUDE.md for invoking `bun test` manually on darwin.
 */
import { afterAll, vi } from "bun:test";
import { disposeDefaultLlamaCpp } from "./llm";

// bun:test's `vi` (what `import { vi } from "vitest"` resolves to under bun)
// lacks hoisted/stubGlobal/unstubAllGlobals; the mocking suites need them.
if (!("hoisted" in vi)) {
  Object.assign(vi, { hoisted: <T>(factory: () => T): T => factory() });
}

if (!("stubGlobal" in vi)) {
  const saved = new Map<PropertyKey, PropertyDescriptor | undefined>();
  Object.assign(vi, {
    stubGlobal(name: PropertyKey, value: unknown) {
      if (!saved.has(name)) saved.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
      Object.defineProperty(globalThis, name, { value, writable: true, configurable: true, enumerable: true });
      return vi;
    },
    unstubAllGlobals() {
      for (const [name, desc] of saved) {
        if (desc) Object.defineProperty(globalThis, name, desc);
        else Reflect.deleteProperty(globalThis, name);
      }
      saved.clear();
      return vi;
    },
  });
}

// Global afterAll runs after all test files complete
afterAll(async () => {
  await disposeDefaultLlamaCpp();
});
