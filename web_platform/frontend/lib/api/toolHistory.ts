/**
 * Tool History API Client
 * 
 * API functions for fetching tool execution history.
 */

import apiClient from './client';
import type { ToolExecution } from '../types/tool';

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
    const response = await apiClient.get<ToolExecution[]>(url);
    return response.data;
}

/**
 * Get tool execution history for a specific message.
 * 
 * This is the key feature for "show me tool history for this message".
 */
export async function getMessageToolHistory(messageId: string): Promise<ToolExecution[]> {
    const response = await apiClient.get<ToolExecution[]>(`/messages/${messageId}/tool-history`);
    return response.data;
}

/**
 * Get detailed information about a specific tool execution.
 */
export async function getToolExecution(executionId: string): Promise<ToolExecution> {
    const response = await apiClient.get<ToolExecution>(`/tool-executions/${executionId}`);
    return response.data;
}

