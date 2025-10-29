/**
 * Memory Management API Functions
 * 
 * API calls for managing chat memory and conversation context.
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';

export interface MemoryStats {
    chatId: string;
    messageCount: number;
    totalCharacters: number;
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
export async function clearChatMemory(chatId: string): Promise<ClearMemoryResponse> {
    const response = await apiClient.post<{
        success: boolean;
        message: string;
        chat_id: string;
    }>(API_ENDPOINTS.CHAT_MEMORY_CLEAR(chatId));

    return {
        success: response.data.success,
        message: response.data.message,
        chatId: response.data.chat_id,
    };
}

/**
 * Get memory statistics for a chat.
 * Shows how much context/memory is being used.
 */
export async function getChatMemoryStats(chatId: string): Promise<MemoryStats> {
    const response = await apiClient.get<{
        chat_id: string;
        message_count: number;
        total_characters: number;
        has_context: boolean;
    }>(API_ENDPOINTS.CHAT_MEMORY_STATS(chatId));

    return {
        chatId: response.data.chat_id,
        messageCount: response.data.message_count,
        totalCharacters: response.data.total_characters,
        hasContext: response.data.has_context,
    };
}

/**
 * Trigger system-wide memory cleanup (admin operation).
 * Clears old checkpointer states and performs garbage collection.
 */
export async function cleanupSystemMemory(): Promise<SystemCleanupStats> {
    const response = await apiClient.post<{
        success: boolean;
        message: string;
        stats: {
            checkpoints_cleared: number;
            memory_freed_mb: number;
        };
    }>(API_ENDPOINTS.SYSTEM_MEMORY_CLEANUP);

    return {
        success: response.data.success,
        message: response.data.message,
        stats: {
            checkpointsCleared: response.data.stats.checkpoints_cleared,
            memoryFreedMb: response.data.stats.memory_freed_mb,
        },
    };
}

