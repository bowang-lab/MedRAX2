/**
 * Message API Functions
 * 
 * API calls for message management and streaming.
 */

import { openapiClient, authHeaders } from '../openapi/client';
import { API_ENDPOINTS, API_SECRET_CONFIG, API_CONFIG } from '../config/api';
import type { MessageWithDetails, SSEEvent } from '../types/message';
import type { ApiMessageWithDetails } from '../types/api';
import { toUiMessage } from '../openapi/transformers';

/**
 * Get all messages for a chat (with attached scans and tool executions)
 * Backend always returns List[MessageWithDetails] (never null)
 */
export async function getMessages(chatId: string): Promise<MessageWithDetails[]> {
    const { data, error } = await openapiClient.GET('/api/chats/{chat_id}/messages', {
        params: { path: { chat_id: chatId } },
        headers: authHeaders(),
    });
    if (error) throw error;
    if (!data) throw new Error('No data returned from server');
    
    return data.map((msg: ApiMessageWithDetails) => toUiMessage(msg));
}

/**
 * Stream chat response using Server-Sent Events (SSE)
 * 
 * This is the primary way to send messages and receive AI responses.
 * Uses EventSource for real-time streaming of:
 * - message_start: New message created
 * - content_chunk: Incremental content from AI
 * - tool_start/tool_done: Tool execution events
 * - message_done: Message complete
 */
export function streamChatResponse(
    chatId: string,
    content: string,
    scanIds: string[],
    onEvent: (event: SSEEvent) => void,
    onComplete: () => void,
    onError: (error: Error) => void
): () => void {
    const url = new URL(`${API_CONFIG.baseURL}${API_ENDPOINTS.CHAT_STREAM(chatId)}`);
    
    // Add auth headers as query params for SSE (EventSource doesn't support custom headers)
    const token = localStorage.getItem('medrax_auth_token');
    const apiSecret = API_SECRET_CONFIG.getSecret();
    
    if (token) {
        url.searchParams.append('token', token);
    }
    if (apiSecret) {
        url.searchParams.append('api_secret', apiSecret);
    }

    const eventSource = new EventSource(url.toString());
    let hasReceivedData = false;

    eventSource.addEventListener('message_start', (e) => {
        hasReceivedData = true;
        try {
            const data = JSON.parse(e.data);
            onEvent({ type: 'message_start', data });
        } catch (err) {
            console.error('Failed to parse message_start:', err);
        }
    });

    eventSource.addEventListener('content_chunk', (e) => {
        hasReceivedData = true;
        try {
            const data = JSON.parse(e.data);
            onEvent({ type: 'content_chunk', data });
        } catch (err) {
            console.error('Failed to parse content_chunk:', err);
        }
    });

    eventSource.addEventListener('tool_start', (e) => {
        hasReceivedData = true;
        try {
            const data = JSON.parse(e.data);
            onEvent({ type: 'tool_start', data });
        } catch (err) {
            console.error('Failed to parse tool_start:', err);
        }
    });

    eventSource.addEventListener('tool_done', (e) => {
        hasReceivedData = true;
        try {
            const data = JSON.parse(e.data);
            onEvent({ type: 'tool_done', data });
        } catch (err) {
            console.error('Failed to parse tool_done:', err);
        }
    });

    eventSource.addEventListener('message_done', (e) => {
        hasReceivedData = true;
        try {
            const data = JSON.parse(e.data);
            onEvent({ type: 'message_done', data });
        } catch (err) {
            console.error('Failed to parse message_done:', err);
        }
        eventSource.close();
        onComplete();
    });

    eventSource.addEventListener('error', (e) => {
        console.error('SSE Error:', e);
        
        // Only call onError if we haven't received any data yet
        // (after data is received, errors are usually just connection cleanup)
        if (!hasReceivedData) {
            const errorEvent = e as ErrorEvent;
            const errorMessage = errorEvent.message || 'Stream connection failed';
            onError(new Error(errorMessage));
        }
        
        eventSource.close();
    });

    // Send the message to start the stream
    fetch(`${API_CONFIG.baseURL}${API_ENDPOINTS.CHAT_STREAM(chatId)}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...authHeaders(),
        },
        body: JSON.stringify({
            content,
            scan_ids: scanIds,
        }),
    }).catch((err) => {
        console.error('Failed to start stream:', err);
        onError(err instanceof Error ? err : new Error('Failed to start stream'));
        eventSource.close();
    });

    // Return cleanup function
    return () => {
        eventSource.close();
    };
}
