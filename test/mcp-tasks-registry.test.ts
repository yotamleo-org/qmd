import { describe, test, expect, afterEach, vi } from "vitest";
import { MAX_TASKS, TASKS_EXTENSION, TASK_TTL_MS, cancelTask, getTask, withTask } from "../src/mcp/tasks.ts";
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

  test("finished tasks do not count toward the cap", async () => {
    for (let i = 0; i < MAX_TASKS; i++) {
      const done = await start(async () => ok);
      await completed(done.taskId!);
    }
    const next = await start(async () => ok);
    expect(next.resultType).toBe("task");
  });

  test("at the cap of running tasks a new call runs synchronously instead of becoming a task", async () => {
    let free!: () => void;
    const hold = new Promise<void>(r => { free = r; });
    const held: Created[] = [];
    for (let i = 0; i < MAX_TASKS; i++) held.push(await start(() => hold));
    const sync = await start(async () => ok);
    expect(sync).toEqual(ok);
    free();
    await completed(held[0].taskId!);
  });

  test("a cancelled task still holds its slot until its handler settles", async () => {
    let free!: () => void;
    const hold = new Promise<void>(r => { free = r; });
    const held: Created[] = [];
    for (let i = 0; i < MAX_TASKS; i++) held.push(await start(() => hold));
    cancelTask(held[0].taskId!, ctx);
    const sync = await start(async () => ok);
    expect(sync).toEqual(ok);
    free();
    await new Promise(r => setTimeout(r, 20));
    const next = await start(async () => ok);
    expect(next.resultType).toBe("task");
  });

  test("tasks/cancel aborts the signal the running handler holds", async () => {
    let seen: AbortSignal | undefined;
    const created = await withTask<object, typeof ctx, Created>(async (_a, c) => {
      seen = (c as { mcpReq?: { signal?: AbortSignal } }).mcpReq?.signal;
      await new Promise(r => setTimeout(r, 20));
      return ok as Created;
    }, track)({}, ctx);
    await new Promise(r => setTimeout(r, 5));
    cancelTask(created.taskId!, ctx);
    expect(seen?.aborted).toBe(true);
  });
});
