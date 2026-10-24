import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const workflow = (name: string) =>
  readFileSync(join(repoRoot, ".github", "workflows", name), "utf8");

const bunVersion = (yml: string) => yml.match(/^\s*bun-version:\s*(\S+)/m)?.[1];

describe("fork tag-push Publish workflow", () => {
  const publish = workflow("publish.yml");

  test("pins Bun to the version ci.yml pins, not latest", () => {
    expect(bunVersion(publish)).toBeDefined();
    expect(bunVersion(publish)).toBe(bunVersion(workflow("ci.yml")));
  });

  test("never npm-publishes the upstream package name", () => {
    expect(publish).not.toMatch(/npm publish/);
  });

  test("still creates the GitHub release", () => {
    expect(publish).toMatch(/gh release create/);
  });
});
