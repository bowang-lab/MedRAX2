/**
 * Tool Execution API Functions
 * 
 * API calls for tool execution data.
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';
import type { ToolExecution, ToolExecutionLog, ToolExecutionResult } from '../types/tool';

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
    const response = await apiClient.get<{
        execution: ToolExecution;
        logs: ToolExecutionLog[];
        result: ToolExecutionResult | null;
    }>(API_ENDPOINTS.EXECUTION_DETAIL(executionId));

    return response.data;
}

