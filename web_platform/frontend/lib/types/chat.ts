/**
 * Chat Types
 *
 * Each patient can have multiple independent chats.
 * First chat is auto-created as "Initial Consultation".
 * Subsequent chats are auto-named by date/time but can be renamed.
 */

export interface Chat {
  id: string;
  patientId: string;
  name: string; // Auto: "2025-01-15 10:30 AM" or custom
  createdAt: string;
  updatedAt: string;
  lastMessageAt: string | null;
  messageCount: number;
  scanCount: number;
}

export interface ChatCreate {
  patientId: string;
  name?: string; // Optional, will auto-generate if not provided
}

export interface ChatUpdate {
  name: string; // Rename chat
}
