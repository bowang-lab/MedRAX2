/**
 * Tool History API Client
 * 
 * API functions for fetching tool execution history.
 */

import apiClient from './client';
import type { ToolExecution } from '../types/tool';

// Backend response types (snake_case from API)
interface BackendToolExecution {
    id: string;
    message_id: string;
    request_id: string | null;
    tool_name: string;
    tool_display_name: string;
    status: string;
    started_at: string;
    completed_at: string | null;
    execution_time_ms: number | null;
    image_paths: string[] | null;
}

/**
 * Transform backend tool execution to frontend format
 * (snake_case to camelCase)
 */
function transformToolExecution(backendExecution: BackendToolExecution): ToolExecution {
    return {
        id: backendExecution.id,
        messageId: backendExecution.message_id,
        requestId: backendExecution.request_id,
        toolName: backendExecution.tool_name,
        toolDisplayName: backendExecution.tool_display_name,
        status: backendExecution.status,
        startedAt: backendExecution.started_at,
        completedAt: backendExecution.completed_at,
        executionTimeMs: backendExecution.execution_time_ms,
        imagePaths: backendExecution.image_paths,
    };
}

/**
 * Get tool execution history for a chat.
 */
export async function getChatToolHistory(
    chatId: string,
    filters?: {
        filterByRequest?: string;
        filterByTool?: string;
        latestOnly?: boolean;
    }
): Promise<ToolExecution[]> {
    const params = new URLSearchParams();
    if (filters?.filterByRequest) params.append('filter_by_request', filters.filterByRequest);
    if (filters?.filterByTool) params.append('filter_by_tool', filters.filterByTool);
    if (filters?.latestOnly) params.append('latest_only', 'true');

    const url = `/chats/${chatId}/tool-history${params.toString() ? `?${params.toString()}` : ''}`;
    const response = await apiClient.get<BackendToolExecution[]>(url);

    // Transform snake_case from backend to camelCase for frontend
    return response.data.map(transformToolExecution);
}

/**
 * Get tool execution history for a specific message.
 * 
 * This is the key feature for "show me tool history for this message".
 */
export async function getMessageToolHistory(messageId: string): Promise<ToolExecution[]> {
    const response = await apiClient.get<BackendToolExecution[]>(`/messages/${messageId}/tool-history`);

    // Transform snake_case from backend to camelCase for frontend
    return response.data.map(transformToolExecution);
}

/**
 * Get detailed information about a specific tool execution.
 */
export async function getToolExecution(executionId: string): Promise<ToolExecution> {
    const response = await apiClient.get<BackendToolExecution>(`/tool-executions/${executionId}`);

    // Transform snake_case from backend to camelCase for frontend
    return transformToolExecution(response.data);
}

