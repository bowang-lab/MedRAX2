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
    status: 'available' | 'unavailable' | 'loaded' | 'unloaded' | 'error';
    category: string;
    loaded_at?: string;
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

