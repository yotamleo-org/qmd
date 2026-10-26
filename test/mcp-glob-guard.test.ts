/**
 * MCP multi_get glob guard (HIMMEL-7227).
 *
 * picomatch/braces have no patched release for their ReDoS / resource
 * exhaustion advisories, so crafted patterns are refused before they reach
 * picomatch. These tests cover the pure validator; the wire path is covered in
 * mcp.test.ts.
 */

import { describe, test, expect } from "vitest";
import picomatch from "picomatch";
import { validateMcpGlobPattern, MCP_GLOB_MAX_LENGTH, MCP_GLOB_MAX_WILDCARD_RUNS } from "../src/mcp/glob-guard.js";

describe("validateMcpGlobPattern", () => {
  test.each([
    "journals/2025-05*.md",
    "docs/**/*.md",
    "meetings/*.md",
    "docs/readme?.md",
    "readme.md, api.md",
    "readme.md,api.md",
    "#abc123",
    "#abc123, #def456",
    "qmd://docs/meetings/*.md",
  ])("accepts documented shape %s", (pattern) => {
    expect(validateMcpGlobPattern(pattern)).toBeNull();
  });

  test.each([
    ["brace expansion", "docs/{readme,changelog}.md"],
    ["brace range", "docs/{1..9999}.md"],
    ["nested braces", "{a,{b,{c,d}}}"],
    ["lone open brace", "docs/{a.md"],
    ["lone close brace", "docs/a}.md"],
    ["brace in a comma list", "readme.md, {a,b}.md"],
    ["extglob @(", "docs/@(a|b).md"],
    ["extglob !(", "docs/!(a|b).md"],
    ["extglob +(", "docs/+(a|aa).md"],
    ["extglob *(", "docs/*(a|aa).md"],
    ["extglob ?(", "docs/?(a|aa).md"],
  ])("refuses %s", (_label, pattern) => {
    const error = validateMcpGlobPattern(pattern);
    expect(error).not.toBeNull();
    expect(error).toMatch(/brace|extglob/i);
  });

  test("refuses a glob longer than the cap", () => {
    expect(MCP_GLOB_MAX_LENGTH).toBe(256);
    expect(validateMcpGlobPattern("a*".repeat(MCP_GLOB_MAX_LENGTH))).toMatch(/too long/i);
    expect(validateMcpGlobPattern("*".repeat(MCP_GLOB_MAX_LENGTH + 1))).toMatch(/too long/i);
    expect(validateMcpGlobPattern("*".repeat(MCP_GLOB_MAX_LENGTH))).toBeNull();
  });

  test("does not cap a long comma-separated list of plain paths", () => {
    const list = Array.from({ length: 40 }, (_, i) => `notes/file-${i}.md`).join(", ");
    expect(list.length).toBeGreaterThan(MCP_GLOB_MAX_LENGTH);
    expect(validateMcpGlobPattern(list)).toBeNull();
  });

  test("a known ReDoS-shaped pattern is refused in well under a second", () => {
    const evil = "+(a|aa)".repeat(30) + "!";
    const start = performance.now();
    const error = validateMcpGlobPattern(evil);
    expect(performance.now() - start).toBeLessThan(100);
    expect(error).not.toBeNull();
  });

  test("refuses more than the allowed runs of * (the plain-star backtracking blow-up)", () => {
    expect(MCP_GLOB_MAX_WILDCARD_RUNS).toBe(3);
    expect(validateMcpGlobPattern("a*".repeat(MCP_GLOB_MAX_WILDCARD_RUNS + 1) + "b")).toMatch(/wildcard/i);
    expect(validateMcpGlobPattern("a*".repeat(MCP_GLOB_MAX_WILDCARD_RUNS) + "b")).toBeNull();
    // ** is one run, so the documented recursive shapes stay legal
    expect(validateMcpGlobPattern("docs/**/2025-*/*.md")).toBeNull();
  });

  test("a plain-star ReDoS pattern is refused, and the most the guard accepts matches fast", () => {
    const path = "a".repeat(300);
    expect(validateMcpGlobPattern("a*".repeat(12) + "b")).not.toBeNull();
    const worstAccepted = "a*".repeat(MCP_GLOB_MAX_WILDCARD_RUNS) + "b";
    expect(validateMcpGlobPattern(worstAccepted)).toBeNull();
    const start = performance.now();
    picomatch(worstAccepted)(path);
    expect(performance.now() - start).toBeLessThan(1000);
  });
});
