/**
 * Chat API Functions
 *
 * API calls for chat management.
 */

import { openapiClient, authHeaders } from "../openapi/client";
import type { Chat } from "../types/chat";
import type { ApiChatResponse, ApiChatCreate, ApiChatUpdate } from "../types/api";
import { toUiChat } from "../openapi/transformers";

/**
 * Get all chats for a patient
 * Backend always returns List[ChatResponse] (never null)
 */
export async function getChats(patientId: string): Promise<Chat[]> {
  const { data, error } = await openapiClient.GET("/api/patients/{patient_id}/chats", {
    params: { path: { patient_id: patientId } },
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("No data returned from server");
  return data.map((chat: ApiChatResponse) => toUiChat(chat));
}

/**
 * Get single chat by ID
 */
export async function getChat(chatId: string): Promise<Chat> {
  const { data, error } = await openapiClient.GET("/api/chats/{chat_id}", {
    params: { path: { chat_id: chatId } },
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("Chat not found");
  return toUiChat(data);
}

/**
 * Create new chat for patient
 */
export async function createChat(
  patientId: string,
  payload: { name?: string }
): Promise<Chat> {
  const requestBody: ApiChatCreate = {
    name: payload.name ?? null,
  };

  const { data, error } = await openapiClient.POST("/api/patients/{patient_id}/chats", {
    params: { path: { patient_id: patientId } },
    body: requestBody,
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("Failed to create chat");
  return toUiChat(data);
}

/**
 * Update chat name
 */
export async function updateChat(
  chatId: string,
  payload: { name: string }
): Promise<Chat> {
  const requestBody: ApiChatUpdate = {
    name: payload.name,
  };

  const { data, error } = await openapiClient.PATCH("/api/chats/{chat_id}", {
    params: { path: { chat_id: chatId } },
    body: requestBody,
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("Failed to update chat");
  return toUiChat(data);
}

/**
 * Delete chat
 */
export async function deleteChat(chatId: string): Promise<void> {
  const { error } = await openapiClient.DELETE("/api/chats/{chat_id}", {
    params: { path: { chat_id: chatId } },
    headers: authHeaders(),
  });
  if (error) throw error;
}
