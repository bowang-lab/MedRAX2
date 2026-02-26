/**
 * Tool History API Functions
 *
 * API calls for tool execution history.
 */

import { openapiClient, authHeaders } from "../openapi/client";
import type { ToolExecution } from "../types/tool";
import type { ApiToolExecutionResponse } from "../types/api";
import { toUiToolExecution } from "../openapi/transformers";

/**
 * Get tool execution history for a specific message.
 * Returns all tool executions associated with that message.
 * Backend always returns List[ToolExecutionResponse] (never null)
 */
export async function getToolExecutionsByMessage(messageId: string): Promise<ToolExecution[]> {
  const { data, error } = await openapiClient.GET("/api/messages/{message_id}/tool-history", {
    params: { path: { message_id: messageId } },
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("No data returned from server");
  return data.map((exec: ApiToolExecutionResponse) => toUiToolExecution(exec));
}

/**
 * Get all tool executions for a chat (across all messages).
 *
 * @param chatId - The chat ID to get tool history for
 * @param filterByRequest - Optional request ID filter
 * @param filterByTool - Optional tool name filter
 * @param latestOnly - Only return latest execution per tool (default: false)
 */
export async function getToolExecutionsByChat(
  chatId: string,
  filterByRequest?: string,
  filterByTool?: string,
  latestOnly?: boolean
): Promise<ToolExecution[]> {
  const { data, error } = await openapiClient.GET("/api/chats/{chat_id}/tool-history", {
    params: {
      path: { chat_id: chatId },
      query: {
        filter_by_request: filterByRequest ?? null,
        filter_by_tool: filterByTool ?? null,
        latest_only: latestOnly ?? false,
      },
    },
    headers: authHeaders(),
  });
  if (error) throw error;
  // Backend always returns List[ToolExecutionResponse] (never null)
  if (!data) throw new Error("No data returned from server");
  return data.map((exec: ApiToolExecutionResponse) => toUiToolExecution(exec));
}
