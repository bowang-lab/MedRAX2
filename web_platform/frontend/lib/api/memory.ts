/**
 * Memory Management API Functions
 *
 * API calls for managing chat memory and conversation context.
 */

import { openapiClient, authHeaders } from "../openapi/client";
import {
  toUiMemoryStats,
  toUiClearMemoryResponse,
  toUiSystemCleanupStats,
} from "../openapi/transformers";
import type {
  ApiMemoryStatsResponse,
  ApiClearMemoryResponse,
  ApiSystemCleanupStatsResponse,
} from "../types/api";

export interface MemoryStats {
  chatId: string;
  messageCount: number;
  scanCount: number;
  toolExecutionCount: number;
  hasContext: boolean;
}

export interface ClearMemoryResponse {
  success: boolean;
  message: string;
  chatId: string;
}

export interface SystemCleanupStats {
  success: boolean;
  message: string;
  stats: {
    checkpointsCleared: number;
    memoryFreedMb: number;
  };
}

/**
 * Clear conversation memory for a chat.
 * Resets the LangGraph checkpointer state, effectively starting a new conversation context.
 */
export async function clearChatMemory(
  chatId: string
): Promise<ClearMemoryResponse> {
  const { data, error } = await openapiClient.POST("/api/chats/{chat_id}/memory/clear", {
    params: { path: { chat_id: chatId } },
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("No data returned from server");

  return toUiClearMemoryResponse(data as ApiClearMemoryResponse);
}

/**
 * Get memory statistics for a chat.
 * Shows how much context/memory is being used.
 */
export async function getChatMemoryStats(chatId: string): Promise<MemoryStats> {
  const { data, error } = await openapiClient.GET("/api/chats/{chat_id}/memory/stats", {
    params: { path: { chat_id: chatId } },
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("No data returned from server");

  return toUiMemoryStats(data as ApiMemoryStatsResponse);
}

/**
 * Trigger system-wide memory cleanup (admin operation).
 * Clears old checkpointer states and performs garbage collection.
 */
export async function cleanupSystemMemory(): Promise<SystemCleanupStats> {
  const { data, error } = await openapiClient.POST("/api/system/memory/cleanup", {
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("No data returned from server");

  return toUiSystemCleanupStats(data as ApiSystemCleanupStatsResponse);
}
