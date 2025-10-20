/**
 * Message API Functions
 * 
 * API calls for message management and streaming.
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';
import type { MessageWithDetails, SSEEvent } from '../types/message';

/**
 * Get all messages for a chat (with attached scans and tool executions)
 */
export async function getMessages(chatId: string): Promise<MessageWithDetails[]> {
    const response = await apiClient.get<MessageWithDetails[]>(
        API_ENDPOINTS.CHAT_MESSAGES(chatId)
    );
    return response.data;
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

    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            Authorization: token ? `Bearer ${token}` : '',
        },
        body: JSON.stringify({ content, scanIds }),
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

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        if (dataStr === '[DONE]') {
                            onComplete();
                            return;
                        }

                        try {
                            const event = JSON.parse(dataStr);
                            onEvent(event);
                        } catch (error) {
                            console.error('Failed to parse SSE event:', error);
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

