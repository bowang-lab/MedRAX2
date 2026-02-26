/**
 * Message API Functions
 *
 * API calls for message management and streaming.
 */

import { openapiClient, authHeaders } from "../openapi/client";
import { API_ENDPOINTS, API_CONFIG } from "../config/api";
import type { MessageWithDetails, SSEEvent } from "../types/message";
import type { ApiMessageWithDetails } from "../types/api";
import { toUiMessage } from "../openapi/transformers";

/**
 * Get all messages for a chat (with attached scans and tool executions)
 * Backend always returns List[MessageWithDetails] (never null)
 */
export async function getMessages(chatId: string): Promise<MessageWithDetails[]> {
  const { data, error } = await openapiClient.GET("/api/chats/{chat_id}/messages", {
    params: { path: { chat_id: chatId } },
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("No data returned from server");

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
  const abortController = new AbortController();
  let hasReceivedData = false;

  // Use fetch with streaming to handle POST SSE properly
  const startStream = async () => {
    try {
      const response = await fetch(`${API_CONFIG.baseURL}${API_ENDPOINTS.CHAT_STREAM(chatId)}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders(),
        },
        body: JSON.stringify({
          content,
          scan_ids: scanIds,
        }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
        throw new Error("Response body is null");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        // Decode the chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages in the buffer
        // SSE format: "event: eventType\ndata: jsonData\n\n"
        const messages = buffer.split("\n\n");
        buffer = messages.pop() || ""; // Keep incomplete message in buffer

        for (const message of messages) {
          if (!message.trim()) continue;

          const lines = message.split("\n");
          let eventType = "";
          let eventData: Record<string, unknown> = {};

          for (const line of lines) {
            if (line.startsWith("event:")) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              const jsonData = line.slice(5).trim();
              try {
                eventData = JSON.parse(jsonData);
              } catch (err) {
                console.error("Failed to parse SSE data:", jsonData, err);
                continue;
              }
            }
          }

          if (eventType) {
            hasReceivedData = true;
            onEvent({ type: eventType as SSEEvent["type"], data: eventData });

            // Check if this is the completion event
            if (eventType === "message_done") {
              onComplete();
              return;
            }
          }
        }
      }

      // Stream ended without message_done
      if (hasReceivedData) {
        onComplete();
      } else {
        onError(new Error("Stream ended without receiving data"));
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        // Stream was aborted by user - this is normal
        return;
      }

      console.error("Stream error:", err);
      if (!hasReceivedData) {
        onError(err instanceof Error ? err : new Error("Stream connection failed"));
      }
    }
  };

  // Start the stream
  startStream();

  // Return cleanup function
  return () => {
    abortController.abort();
  };
}
