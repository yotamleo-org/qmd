/**
 * auth.ts — bearer-token auth for the HTTP MCP transport.
 *
 * The origin guard keeps web pages out, but any local process can still curl
 * the loopback daemon. A bearer token read from a 0600 file (generated on first
 * start) keeps other local users and sandboxed sessions out: only a process
 * that can read the operator's token file can query the index.
 */
import { randomBytes, timingSafeEqual } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

/** Default token path: $QMD_HTTP_TOKEN_FILE, else $XDG_CONFIG_HOME/qmd/http-token. */
export function defaultTokenPath(): string {
  if (process.env.QMD_HTTP_TOKEN_FILE) return process.env.QMD_HTTP_TOKEN_FILE;
  return join(process.env.XDG_CONFIG_HOME || join(homedir(), ".config"), "qmd", "http-token");
}

/** Read the token file, creating it (mode 0600, random 256-bit hex) on first use. */
export function loadOrCreateToken(path: string = defaultTokenPath()): string {
  if (!existsSync(path)) {
    mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
    // "wx": never overwrite a token another start just wrote.
    try {
      writeFileSync(path, randomBytes(32).toString("hex") + "\n", { mode: 0o600, flag: "wx" });
    } catch (e: unknown) {
      if (!(typeof e === "object" && e !== null && "code" in e && e.code === "EEXIST")) throw e;
    }
  }
  if (process.platform !== "win32" && (statSync(path).mode & 0o077) !== 0) {
    throw new Error(`qmd http token file ${path} must be mode 0600 (chmod 600 it, or delete it to regenerate)`);
  }
  const token = readFileSync(path, "utf8").trim();
  if (token.length < 32) throw new Error(`qmd http token file ${path} holds a token shorter than 32 characters`);
  return token;
}

/** Constant-time check of an `Authorization: Bearer <token>` header. */
export function checkBearer(header: string | undefined, token: string): boolean {
  if (!header || !header.startsWith("Bearer ")) return false;
  const got = Buffer.from(header.slice(7));
  const want = Buffer.from(token);
  return got.length === want.length && timingSafeEqual(got, want);
}
