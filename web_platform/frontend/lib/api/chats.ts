/**
 * Chat API Functions
 * 
 * API calls for chat management.
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';
import type { Chat } from '../types/chat';

/**
 * Backend chat response interface (snake_case)
 */
interface ChatAPIResponse {
    id: string;
    patient_id: string;
    name: string;
    created_at: string;
    updated_at: string;
    last_message_at: string | null;
    message_count: number;
    scan_count: number;
}

/**
 * Map backend chat response to frontend Chat type
 */
function mapChatResponse(data: ChatAPIResponse): Chat {
    return {
        id: data.id,
        patientId: data.patient_id,
        name: data.name,
        createdAt: data.created_at,
        updatedAt: data.updated_at,
        lastMessageAt: data.last_message_at,
        messageCount: data.message_count,
        scanCount: data.scan_count,
    };
}

/**
 * Get all chats for a patient
 */
export async function getChats(patientId: string): Promise<Chat[]> {
    const response = await apiClient.get<ChatAPIResponse[]>(
        API_ENDPOINTS.PATIENT_CHATS(patientId)
    );
    return response.data.map(mapChatResponse);
}

/**
 * Get single chat by ID
 */
export async function getChat(chatId: string): Promise<Chat> {
    const response = await apiClient.get<ChatAPIResponse>(
        API_ENDPOINTS.CHAT_DETAIL(chatId)
    );
    return mapChatResponse(response.data);
}

/**
 * Create new chat for patient
 */
export async function createChat(
    patientId: string,
    data: { name?: string }
): Promise<Chat> {
    const response = await apiClient.post<ChatAPIResponse>(
        API_ENDPOINTS.PATIENT_CHATS(patientId),
        data
    );
    return mapChatResponse(response.data);
}

/**
 * Update chat name
 */
export async function updateChat(
    chatId: string,
    data: { name: string }
): Promise<Chat> {
    const response = await apiClient.patch<ChatAPIResponse>(
        API_ENDPOINTS.CHAT_DETAIL(chatId),
        data
    );
    return mapChatResponse(response.data);
}

/**
 * Delete chat
 */
export async function deleteChat(chatId: string): Promise<void> {
    await apiClient.delete(API_ENDPOINTS.CHAT_DETAIL(chatId));
}

