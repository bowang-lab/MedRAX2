/**
 * Tool Execution API Functions
 *
 * API calls for tool execution data.
 */

import { openapiClient, authHeaders } from "../openapi/client";
import type { ToolExecution, ToolExecutionLog, ToolExecutionResult } from "../types/tool";
import type { ApiToolExecutionDetailResponse } from "../types/api";
import { toUiToolExecution, toUiToolExecutionLog, toUiToolExecutionResult } from "../openapi/transformers";

export interface ToolExecutionDetail {
  execution: ToolExecution;
  logs: ToolExecutionLog[];
  result: ToolExecutionResult | null;
}

/**
 * Get detailed tool execution data (logs + result)
 */
export async function getToolExecutionDetail(executionId: string): Promise<ToolExecutionDetail> {
  const { data, error } = await openapiClient.GET("/api/tools/executions/{execution_id}", {
    params: { path: { execution_id: executionId } },
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("Tool execution not found");

  const response = data as ApiToolExecutionDetailResponse;

  return {
    execution: toUiToolExecution(response.execution),
    logs: response.logs.map(toUiToolExecutionLog),
    result: response.result ? toUiToolExecutionResult(response.result) : null,
  };
}
