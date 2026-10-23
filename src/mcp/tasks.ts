/**
 * MCP tasks (io.modelcontextprotocol/tasks, protocol revision 2026-07-28) for
 * slow tool calls. A client that declares the extension on a request gets a
 * task back at once and polls `tasks/get`; any other client gets the plain
 * synchronous result. The SDK has no tasks runtime, so the registry lives here.
 *
 * The registry is process-wide: HTTP builds one server instance per request,
 * so a task must outlive the instance that created it.
 */

import { randomUUID } from "node:crypto";
import { CLIENT_CAPABILITIES_META_KEY, ProtocolError, ProtocolErrorCode } from "@modelcontextprotocol/server";

export const TASKS_EXTENSION = "io.modelcontextprotocol/tasks";

const POLL_INTERVAL_MS = 2000;
// How long a finished task stays readable after its last update.
const TASK_TTL_MS = 60 * 60 * 1000;

type TaskStatus = "working" | "completed" | "failed";

type TaskRecord = {
  taskId: string;
  status: TaskStatus;
  createdAt: string;
  lastUpdatedAt: string;
  result?: unknown;
  error?: { code: number; message: string };
};

const tasks = new Map<string, TaskRecord>();

function sweep(now: number): void {
  for (const [id, t] of tasks) {
    if (t.status !== "working" && now - Date.parse(t.lastUpdatedAt) > TASK_TTL_MS) tasks.delete(id);
  }
}

function settle(task: TaskRecord, patch: Pick<TaskRecord, "status"> & Partial<TaskRecord>): void {
  Object.assign(task, patch, { lastUpdatedAt: new Date().toISOString() });
}

export type ToolCallContext = { mcpReq?: { envelope?: Record<string, unknown> } };

/** Whether this request's declared client capabilities include the tasks extension. */
export function clientWantsTask(ctx: ToolCallContext | undefined): boolean {
  const caps = ctx?.mcpReq?.envelope?.[CLIENT_CAPABILITIES_META_KEY] as { extensions?: Record<string, unknown> } | undefined;
  return caps?.extensions?.[TASKS_EXTENSION] !== undefined;
}

/**
 * Wrap a tool handler: with the extension the handler runs in the background
 * and the call answers `resultType: "task"`; without it the handler runs as-is.
 * `track` keeps the background run visible to the stdio EOF shutdown.
 */
export function withTask<A, C extends ToolCallContext, R>(
  handler: (args: A, ctx: C) => Promise<R>,
  track: <T extends (...args: never[]) => unknown>(fn: T) => T,
): (args: A, ctx: C) => Promise<R> {
  return async (args, ctx) => {
    if (!clientWantsTask(ctx)) return handler(args, ctx);
    const now = new Date().toISOString();
    sweep(Date.now());
    const task: TaskRecord = { taskId: randomUUID(), status: "working", createdAt: now, lastUpdatedAt: now };
    tasks.set(task.taskId, task);
    // The HTTP request that created the task ends at once, so its abort signal must not reach the background run.
    const detached = { ...ctx, mcpReq: { ...(ctx as { mcpReq?: object }).mcpReq, signal: new AbortController().signal } } as C;
    void track(async () => {
      try {
        settle(task, { status: "completed", result: await handler(args, detached) });
      } catch (err) {
        settle(task, { status: "failed", error: { code: -32603, message: err instanceof Error ? err.message : String(err) } });
      }
    })();
    // The SDK passes a non-"complete" resultType through for tools/call untouched.
    const created: Record<string, unknown> = { resultType: "task", ...view(task) };
    return created as R;
  };
}

function view(task: TaskRecord): Record<string, unknown> {
  return {
    taskId: task.taskId,
    status: task.status,
    createdAt: task.createdAt,
    lastUpdatedAt: task.lastUpdatedAt,
    ttlMs: TASK_TTL_MS,
    ...(task.status === "working" ? { pollIntervalMs: POLL_INTERVAL_MS } : {}),
  };
}

/** The `tasks/get` result for a task id; the SDK stamps `resultType: "complete"` on it. */
export function getTask(taskId: string): Record<string, unknown> {
  sweep(Date.now());
  const task = tasks.get(taskId);
  if (!task) throw new ProtocolError(ProtocolErrorCode.InvalidParams, `Unknown task: ${taskId}`);
  return {
    ...view(task),
    ...(task.status === "completed" ? { result: task.result } : {}),
    ...(task.status === "failed" ? { error: task.error } : {}),
  };
}
