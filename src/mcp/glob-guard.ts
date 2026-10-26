/**
 * glob-guard.ts — input validation for the MCP `multi_get` glob pattern.
 *
 * `multi_get` hands its `pattern` to picomatch (which expands braces via the
 * `braces` package). Both have unpatched ReDoS / resource-exhaustion advisories,
 * and the pattern comes from the MCP client, so it is checked here, before it
 * reaches picomatch. picomatch passes `(`, `|`, `+` and bracket bodies straight
 * into the regex as live operators, so the check is an allowlist rather than a
 * list of known-bad syntax. The CLI and SDK paths are not affected: their pattern
 * comes from the local operator, who keeps brace expansion.
 *
 * Pure so it can be unit-tested without a server.
 */

/** Longest glob accepted over MCP. Real patterns (`journals/2025-05*.md`) are tens of chars. */
export const MCP_GLOB_MAX_LENGTH = 256;

/**
 * Most `*` runs accepted. picomatch backtracks polynomially in the run count
 * (`a*a*a*a*a*b` against 200 chars takes ~10 s, 4 runs ~0.25 s, 3 runs ~16 ms),
 * so the length cap alone does not bound it. `**` counts as one run.
 */
export const MCP_GLOB_MAX_WILDCARD_RUNS = 3;

/** Most `[...]` classes accepted. */
export const MCP_GLOB_MAX_CLASSES = 3;

const CLASS = /\[([^\]]*)\]/g;
/** Outside a class: path and name characters, `,` and the wildcards `*` `?`. */
const DISALLOWED_OUTSIDE = /[^\p{L}\p{N}\p{M} ._\-/:#~,@&=%'*?]/u;
/** Inside a class: plain name characters, no negation, ranges of letters and digits only via `-`. */
const DISALLOWED_IN_CLASS = /[^\p{L}\p{N}\p{M} ._\-/:~]/u;

const ALLOWED_SET =
  "letters, digits, space and . _ - / : # ~ , @ & = % ' plus the wildcards * ? and [...] classes of letters, digits and . _ - / : ~";

/** The same test `findDocuments` uses to route a pattern to the comma-list path (no picomatch). */
function isPlainCommaList(pattern: string): boolean {
  return pattern.includes(",") && !/[*?{]/.test(pattern);
}

function unsupported(char: string): string {
  return `The character ${JSON.stringify(char)} is not supported in MCP glob patterns. Allowed: ${ALLOWED_SET}.`;
}

/** Returns an error message for a refused pattern, or null when it may go on to picomatch. */
export function validateMcpGlobPattern(pattern: string): string | null {
  if (isPlainCommaList(pattern)) return null;

  if (pattern.length > MCP_GLOB_MAX_LENGTH) {
    return `Glob pattern too long (${pattern.length} chars, max ${MCP_GLOB_MAX_LENGTH}).`;
  }

  const classes = [...pattern.matchAll(CLASS)];
  for (const [, body] of classes) {
    const bad = body!.length === 0 ? "[]" : body!.match(DISALLOWED_IN_CLASS)?.[0];
    if (bad) return unsupported(bad);
  }
  const outside = pattern.replace(CLASS, "");
  const bad = outside.match(DISALLOWED_OUTSIDE)?.[0];
  if (bad) return unsupported(bad);

  if (classes.length > MCP_GLOB_MAX_CLASSES) {
    return `Too many [...] classes in glob pattern (max ${MCP_GLOB_MAX_CLASSES}).`;
  }
  if ((outside.match(/\*+/g)?.length ?? 0) > MCP_GLOB_MAX_WILDCARD_RUNS) {
    return `Too many wildcards in glob pattern (max ${MCP_GLOB_MAX_WILDCARD_RUNS} runs of *).`;
  }
  return null;
}
