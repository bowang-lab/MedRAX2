/**
 * Tool Execution API Functions
 * 
 * API calls for tool execution data.
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';
import type { ToolExecution, ToolExecutionLog, ToolExecutionResult } from '../types/tool';

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

interface BackendToolExecutionLog {
    id: string;
    execution_id: string;
    log_level: string;
    message: string;
    timestamp: string;
}

interface BackendToolExecutionResult {
    id: string;
    execution_id: string;
    result_data: Record<string, unknown>;
    result_metadata: Record<string, unknown> | null;
    created_at: string;
}

interface BackendToolExecutionDetail {
    execution: BackendToolExecution;
    logs: BackendToolExecutionLog[];
    result: BackendToolExecutionResult | null;
}

/**
 * Get tool executions for a message
 */
export async function getToolExecutions(messageId: string): Promise<ToolExecution[]> {
    const response = await apiClient.get<{ executions: ToolExecution[] }>(
        API_ENDPOINTS.MESSAGE_EXECUTIONS(messageId)
    );
    return response.data.executions;
}

/**
 * Get detailed tool execution data (logs + result)
 */
export async function getToolExecutionDetail(executionId: string): Promise<{
    execution: ToolExecution;
    logs: ToolExecutionLog[];
    result: ToolExecutionResult | null;
}> {
    const response = await apiClient.get<BackendToolExecutionDetail>(
        API_ENDPOINTS.EXECUTION_DETAIL(executionId)
    );

    // Transform snake_case from backend to camelCase for frontend
    const data = response.data;

    return {
        execution: {
            id: data.execution.id,
            messageId: data.execution.message_id,
            requestId: data.execution.request_id,
            toolName: data.execution.tool_name,
            toolDisplayName: data.execution.tool_display_name,
            status: data.execution.status,
            startedAt: data.execution.started_at,
            completedAt: data.execution.completed_at,
            executionTimeMs: data.execution.execution_time_ms,
            imagePaths: data.execution.image_paths,
        },
        logs: data.logs.map((log) => ({
            id: log.id,
            executionId: log.execution_id,
            logLevel: log.log_level,
            message: log.message,
            timestamp: log.timestamp,
        })),
        result: data.result ? {
            id: data.result.id,
            executionId: data.result.execution_id,
            resultData: data.result.result_data,
            resultMetadata: data.result.result_metadata,
            createdAt: data.result.created_at,
        } : null,
    };
}

