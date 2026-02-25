/**
 * Message Types
 *
 * Messages are the core communication in chats.
 * Each message can have attached scans and trigger tool executions.
 *
 * The frontend primarily uses MessageWithDetails which includes
 * all related data (scans, tool executions). The base Message type
 * is kept for API responses that might not include full details.
 */

import { Scan } from "./scan";
import { ToolExecution } from "./tool";

/**
 * Base message structure
 * Used when creating messages or in API responses without full details
 */
export interface Message {
  id: string;
  chatId: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: string;
}

/**
 * Message with full details (attachedScans and toolExecutions)
 * This is the primary type used throughout the frontend
 */
export interface MessageWithDetails extends Message {
  attachedScans: Scan[];
  toolExecutions: ToolExecution[];
}

/**
 * Server-Sent Event types for streaming chat responses
 */
export type SSEEventType =
  | "message_start"
  | "content_chunk"
  | "tool_start"
  | "tool_output"
  | "tool_done"
  | "tool_error"
  | "message_done"
  | "error";

export interface SSEEvent {
  type: SSEEventType;
  data: {
    messageId?: string;
    content?: string;
    toolName?: string;
    toolId?: string;
    error?: string;
    [key: string]: unknown; // Allow additional properties
  };
}
