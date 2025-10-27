/**
 * Tool Management API Functions
 * 
 * API calls for managing tool loading/unloading.
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';

export interface Tool {
    id: string;
    name: string;
    description: string;
    status: 'available' | 'unavailable' | 'loaded' | 'unloaded' | 'error' | 'loading';
    category: string;
    loaded_at?: string;
    // Additional fields returned by backend
    dependencies?: string[];
    requires_gpu?: boolean;
    error_message?: string;
}

/**
 * Get all available tools
 */
export async function getTools(): Promise<Tool[]> {
    const response = await apiClient.get<Tool[]>(
        API_ENDPOINTS.TOOLS
    );
    return response.data;
}

/**
 * Load a tool
 */
export async function loadTool(toolId: string): Promise<void> {
    await apiClient.post(API_ENDPOINTS.TOOL_LOAD(toolId));
}

/**
 * Unload a tool
 */
export async function unloadTool(toolId: string): Promise<void> {
    await apiClient.post(API_ENDPOINTS.TOOL_UNLOAD(toolId));
}

/**
 * Bulk load tools
 */
export async function bulkLoadTools(params: { toolIds?: string[]; loadAll?: boolean }): Promise<{
    results: { id: string; success: boolean; status: string; message?: string }[];
}> {
    const response = await apiClient.post(
        API_ENDPOINTS.TOOLS_BULK_LOAD,
        {
            tool_ids: params.toolIds,
            load_all: params.loadAll ?? false,
        }
    );
    return response.data;
}

