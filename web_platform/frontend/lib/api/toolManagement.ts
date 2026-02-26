/**
 * Tool Management API Functions
 *
 * API calls for managing tool loading/unloading.
 */

import { openapiClient, authHeaders } from "../openapi/client";
import { toUiTool } from "../openapi/transformers";
import type { ApiToolBulkLoadRequest, ApiToolBulkLoadResponse, ApiToolInfo } from "../types/api";

export interface Tool {
  id: string;
  name: string;
  description: string;
  status: "available" | "unavailable" | "loaded" | "unloaded" | "error" | "loading";
  category: string;
  loaded_at?: string;
  // Additional fields returned by backend
  dependencies?: string[];
  requires_gpu?: boolean;
  error_message?: string;
}

export interface ToolBulkLoadResult {
  id: string;
  success: boolean;
  status: string;
  message?: string;
}

/**
 * Get all available tools
 */
export async function getTools(): Promise<Tool[]> {
  const { data, error } = await openapiClient.GET("/api/tools", {
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("No data returned from server");

  return (data as ApiToolInfo[]).map(toUiTool);
}

/**
 * Load a tool
 */
export async function loadTool(toolId: string): Promise<void> {
  const { error } = await openapiClient.POST("/api/tools/{tool_id}/load", {
    params: { path: { tool_id: toolId } },
    headers: authHeaders(),
  });
  if (error) throw error;
}

/**
 * Unload a tool
 */
export async function unloadTool(toolId: string): Promise<void> {
  const { error } = await openapiClient.POST("/api/tools/{tool_id}/unload", {
    params: { path: { tool_id: toolId } },
    headers: authHeaders(),
  });
  if (error) throw error;
}

/**
 * Bulk load tools
 */
export async function bulkLoadTools(params: { toolIds?: string[]; loadAll?: boolean }): Promise<{
  results: ToolBulkLoadResult[];
}> {
  const requestBody: ApiToolBulkLoadRequest = {
    tool_ids: params.toolIds ?? null,
    load_all: params.loadAll ?? false,
  };

  const { data, error } = await openapiClient.POST("/api/tools/bulk-load", {
    body: requestBody,
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("Failed to bulk load tools");

  const response = data as ApiToolBulkLoadResponse;
  return {
    results: response.results.map((result: { id: string; success: boolean; status: string; message?: string | null }) => ({
      id: result.id,
      success: result.success,
      status: result.status,
      message: result.message ?? undefined,
    })),
  };
}
