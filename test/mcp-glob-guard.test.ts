/**
 * MCP multi_get glob guard (HIMMEL-7227).
 *
 * picomatch/braces have no patched release for their ReDoS / resource
 * exhaustion advisories, so MCP patterns are allowlisted before they reach
 * picomatch. These tests cover the pure validator; the wire path is covered in
 * mcp.test.ts.
 */

import { describe, test, expect } from "vitest";
import picomatch from "picomatch";
import {
  validateMcpGlobPattern,
  MCP_GLOB_MAX_LENGTH,
  MCP_GLOB_MAX_WILDCARD_RUNS,
  MCP_GLOB_MAX_CLASSES,
} from "../src/mcp/glob-guard.js";

describe("validateMcpGlobPattern", () => {
  test.each([
    "journals/2025-05*.md",
    "docs/**/*.md",
    "meetings/*.md",
    "docs/readme?.md",
    "docs/note-[0-9].md",
    "docs/[abc]*.md",
    "notes/日本語*.md",
    "my notes/it's & mine=1%.md",
    "readme.md, api.md",
    "readme.md,api.md",
    "notes (1).md, notes (2).md",
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
    ["plain group with a quantifier", "([a-z0-9./:_-]+)+Q"],
    ["plain group with alternation", "(a|a)+Q"],
    ["alternation bar", "docs/a|b.md"],
    ["plus after a class", "docs/[a-z]+Q"],
    ["plus after a group", "docs/(ab)+Q"],
    ["plus on its own", "docs/a+b.md"],
    ["class with a plus inside", "docs/[a+]Q"],
    ["class with a group inside", "docs/[(a)]Q"],
    ["negated class", "docs/[!a]Q"],
    ["caret class", "docs/[^a]Q"],
    ["empty class", "docs/[]Q"],
    ["unterminated class", "docs/[abc"],
    ["stray close bracket", "docs/abc]"],
    ["leading negation", "!docs/*.md"],
    ["backslash escape", "docs/a\\*b"],
    ["class repeating a plus", "[a-z./]+[a-z./]+[a-z./]+[a-z./]+[a-z./]+Q"],
  ])("refuses %s", (_label, pattern) => {
    expect(validateMcpGlobPattern(pattern)).toMatch(/not supported/i);
  });

  test("the refusal names what is allowed", () => {
    expect(validateMcpGlobPattern("(a)")).toMatch(/Allowed: letters, digits/);
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

  test("refuses more than the allowed runs of * (the plain-star backtracking blow-up)", () => {
    expect(MCP_GLOB_MAX_WILDCARD_RUNS).toBe(3);
    expect(validateMcpGlobPattern("a*".repeat(MCP_GLOB_MAX_WILDCARD_RUNS + 1) + "b")).toMatch(/wildcard/i);
    expect(validateMcpGlobPattern("a*".repeat(MCP_GLOB_MAX_WILDCARD_RUNS) + "b")).toBeNull();
    // ** is one run, so the documented recursive shapes stay legal
    expect(validateMcpGlobPattern("docs/**/2025-*/*.md")).toBeNull();
  });

  test("refuses more than the allowed number of [...] classes", () => {
    expect(MCP_GLOB_MAX_CLASSES).toBe(3);
    expect(validateMcpGlobPattern("[a]".repeat(MCP_GLOB_MAX_CLASSES + 1))).toMatch(/classes/i);
    expect(validateMcpGlobPattern("[a]".repeat(MCP_GLOB_MAX_CLASSES))).toBeNull();
  });

  test("known ReDoS shapes are refused instantly, never handed to picomatch", () => {
    const evil = ["+(a|aa)".repeat(30) + "!", "([a-z0-9./:_-]|[a-z0-9./:_-])+Q", "([a-z0-9./:_-]+)+Q"];
    for (const pattern of evil) {
      const start = performance.now();
      expect(validateMcpGlobPattern(pattern)).not.toBeNull();
      expect(performance.now() - start).toBeLessThan(1000);
    }
  });

  test("the costliest accepted shapes match a long path in bounded time", () => {
    const path = "a".repeat(300);
    const worstAccepted = [
      "a*".repeat(MCP_GLOB_MAX_WILDCARD_RUNS) + "b",
      "*a*a*a" + "[a]".repeat(MCP_GLOB_MAX_CLASSES) + "b",
      "a?".repeat(100) + "b",
    ];
    for (const pattern of worstAccepted) {
      expect(validateMcpGlobPattern(pattern)).toBeNull();
      const start = performance.now();
      picomatch(pattern)(path);
      // measured at ~16 ms; the bound is generous so a loaded CI box cannot flake it,
      // and still far below the multi-second blow-up the guard exists to prevent
      expect(performance.now() - start).toBeLessThan(5000);
    }
  });
});
