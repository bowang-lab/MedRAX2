/**
 * Message API Functions
 * 
 * API calls for message management and streaming.
 */

import apiClient from './client';
import { API_ENDPOINTS, API_SECRET_CONFIG } from '../config/api';
import type { MessageWithDetails, SSEEvent } from '../types/message';
import type { Scan } from '../types/scan';
import type { ToolExecution } from '../types/tool';

/**
 * Backend message response interface (snake_case)
 */
interface MessageAPIResponse {
    id: string;
    chat_id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    created_at: string;
    attached_scans: Array<{
        id: string;
        chat_id: string;
        file_path: string;
        display_path: string;
        file_type: string;
        file_size: number;
        uploaded_at: string;
    }>;
    tool_executions: Array<{
        id: string;
        message_id: string;
        request_id: string;
        tool_name: string;
        tool_display_name: string;
        status: string;
        started_at: string;
        completed_at: string | null;
        execution_time_ms: number | null;
        image_paths: string[] | null;
    }>;
}

/**
 * Map backend message response to frontend MessageWithDetails type
 */
function mapMessageResponse(data: MessageAPIResponse): MessageWithDetails {
    return {
        id: data.id,
        chatId: data.chat_id,
        role: data.role,
        content: data.content,
        createdAt: data.created_at,
        attachedScans: data.attached_scans.map(scan => ({
            id: scan.id,
            chatId: scan.chat_id,
            filePath: scan.file_path,
            displayPath: scan.display_path,
            fileType: scan.file_type as any,
            fileSize: scan.file_size,
            uploadedAt: scan.uploaded_at,
        })),
        toolExecutions: data.tool_executions.map(exec => ({
            id: exec.id,
            messageId: exec.message_id,
            requestId: exec.request_id,
            toolName: exec.tool_name,
            toolDisplayName: exec.tool_display_name,
            status: exec.status as any, // Backend string -> frontend ToolStatus enum
            startedAt: exec.started_at,
            completedAt: exec.completed_at,
            executionTimeMs: exec.execution_time_ms,
            imagePaths: exec.image_paths,
        })),
    };
}

/**
 * Get all messages for a chat (with attached scans and tool executions)
 */
export async function getMessages(chatId: string): Promise<MessageWithDetails[]> {
    const response = await apiClient.get<MessageAPIResponse[]>(
        API_ENDPOINTS.CHAT_MESSAGES(chatId)
    );
    return response.data.map(mapMessageResponse);
}

/**
 * Stream chat response using Server-Sent Events (SSE)
 * 
 * This is the primary way to send messages and receive AI responses.
 * Uses streaming to provide real-time updates on tool execution and content generation.
 * 
 * Event types:
 * - message_start: New message created
 * - content_chunk: Streaming content chunk  
 * - tool_start: Tool execution started
 * - tool_done: Tool execution completed
 * - message_done: Message completed
 * - error: Error occurred
 * 
 * @returns Cancellation function to abort the stream
 */
export function streamChatResponse(
    chatId: string,
    content: string,
    onEvent: (event: SSEEvent) => void,
    onComplete: () => void,
    onError: (error: Error) => void,
    scanIds?: string[]
): () => void {
    const controller = new AbortController();
    const { signal } = controller;

    const url = `${apiClient.defaults.baseURL}${API_ENDPOINTS.CHAT_STREAM(chatId)}`;
    const token = localStorage.getItem('medrax_auth_token');
    const apiSecret = API_SECRET_CONFIG.getSecret();

    const requestBody = { content, scan_ids: scanIds };
    console.log(`🌐 Streaming chat request:`, { url, scanIds });

    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            Authorization: token ? `Bearer ${token}` : '',
            'X-API-Secret': apiSecret || '', // Include API secret for middleware
        },
        body: JSON.stringify(requestBody),
        signal,
    })
        .then(async (response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body?.getReader();
            const decoder = new TextDecoder();

            if (!reader) {
                throw new Error('Response body is not readable');
            }

            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const events = buffer.split('\n\n');

                // Keep the last incomplete event in buffer
                buffer = events.pop() || '';

                for (const eventText of events) {
                    if (!eventText.trim()) continue;

                    // Parse SSE event (format: "event: type\ndata: {...}")
                    const lines = eventText.split('\n');
                    let eventType = 'message'; // Default SSE event type
                    let eventData = '';

                    for (const line of lines) {
                        if (line.startsWith('event: ')) {
                            eventType = line.slice(7).trim();
                        } else if (line.startsWith('data: ')) {
                            eventData = line.slice(6);
                        }
                    }

                    if (eventData === '[DONE]') {
                        onComplete();
                        return;
                    }

                    if (eventData) {
                        try {
                            const data = JSON.parse(eventData);
                            // Create SSEEvent with type and data
                            const sseEvent: SSEEvent = {
                                type: eventType as any,
                                data: data
                            };
                            onEvent(sseEvent);
                        } catch (error) {
                            console.error('Failed to parse SSE event:', error, 'Raw:', eventData);
                        }
                    }
                }
            }

            onComplete();
        })
        .catch((error) => {
            if (error.name === 'AbortError') {
                // Request was cancelled
                return;
            }
            onError(error);
        });

    // Return cancellation function
    return () => controller.abort();
}

