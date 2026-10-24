import { describe, test, expect, afterEach, vi } from "vitest";
import { MAX_TASKS, TASKS_EXTENSION, TASK_TTL_MS, getTask, withTask } from "../src/mcp/tasks.ts";
import { CLIENT_CAPABILITIES_META_KEY } from "@modelcontextprotocol/server";

const ctx = { mcpReq: { envelope: { [CLIENT_CAPABILITIES_META_KEY]: { extensions: { [TASKS_EXTENSION]: {} } } } } };
const track = <T extends (...args: never[]) => unknown>(fn: T) => fn;
const ok = { content: [{ type: "text" as const, text: "done" }] };

type Created = { taskId?: string; resultType?: string; isError?: boolean; content?: { text: string }[] };
const start = (run: () => Promise<unknown>): Promise<Created> => withTask<object, typeof ctx, Created>(async () => (await run()) as Created, track)({}, ctx);

describe("task registry", () => {
  afterEach(() => vi.useRealTimers());

  test("a finished task is swept once its ttl has passed", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    const created = await start(async () => ok);
    await vi.waitFor(() => expect(getTask(created.taskId!, ctx).status).toBe("completed"));
    vi.setSystemTime(Date.now() + TASK_TTL_MS + 1000);
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
    await vi.waitFor(() => expect(getTask(held[0].taskId!, ctx).status).toBe("completed"));
  });
});
