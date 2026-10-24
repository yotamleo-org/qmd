import { describe, test, expect, afterEach, vi } from "vitest";
import { MAX_TASKS, TASKS_EXTENSION, TASK_TTL_MS, getTask, withTask } from "../src/mcp/tasks.ts";
import { CLIENT_CAPABILITIES_META_KEY } from "@modelcontextprotocol/server";

const ctx = { mcpReq: { envelope: { [CLIENT_CAPABILITIES_META_KEY]: { extensions: { [TASKS_EXTENSION]: {} } } } } };
const track = <T extends (...args: never[]) => unknown>(fn: T) => fn;
const ok = { content: [{ type: "text" as const, text: "done" }] };

type Created = { taskId?: string; resultType?: string; isError?: boolean; content?: { text: string }[] };
const start = (run: () => Promise<unknown>): Promise<Created> => withTask<object, typeof ctx, Created>(async () => (await run()) as Created, track)({}, ctx);

// bun's vitest shim has no vi.waitFor
async function completed(taskId: string): Promise<void> {
  for (let i = 0; i < 200 && getTask(taskId, ctx).status !== "completed"; i++) await new Promise(r => setTimeout(r, 10));
  expect(getTask(taskId, ctx).status).toBe("completed");
}

describe("task registry", () => {
  afterEach(() => vi.restoreAllMocks());

  test("a finished task is swept once its ttl has passed", async () => {
    const created = await start(async () => ok);
    await completed(created.taskId!);
    const later = Date.now() + TASK_TTL_MS + 1000;
    vi.spyOn(Date, "now").mockReturnValue(later);
    expect(() => getTask(created.taskId!, ctx)).toThrow(/Unknown task/);
  });

  test("above the cap a new task is refused with an error result", async () => {
    let free!: () => void;
    const hold = new Promise<void>(r => { free = r; });
    const held: Created[] = [];
    for (let i = 0; i < MAX_TASKS; i++) held.push(await start(() => hold));
    const refused = await start(async () => ok);
    expect(refused.isError).toBe(true);
    expect(refused.taskId).toBeUndefined();
    expect(refused.content?.[0].text).toContain("Busy");
    free();
    await completed(held[0].taskId!);
  });
});
