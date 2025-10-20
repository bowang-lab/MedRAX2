/**
 * Chat API Functions
 * 
 * API calls for chat management.
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';
import type { Chat } from '../types/chat';

/**
 * Get all chats for a patient
 */
export async function getChats(patientId: string): Promise<Chat[]> {
    const response = await apiClient.get<Chat[]>(
        API_ENDPOINTS.PATIENT_CHATS(patientId)
    );
    return response.data;
}

/**
 * Get single chat by ID
 */
export async function getChat(chatId: string): Promise<Chat> {
    const response = await apiClient.get<Chat>(
        API_ENDPOINTS.CHAT_DETAIL(chatId)
    );
    return response.data;
}

/**
 * Create new chat for patient
 */
export async function createChat(
    patientId: string,
    data: { name?: string }
): Promise<Chat> {
    const response = await apiClient.post<Chat>(
        API_ENDPOINTS.PATIENT_CHATS(patientId),
        data
    );
    return response.data;
}

/**
 * Update chat name
 */
export async function updateChat(
    chatId: string,
    data: { name: string }
): Promise<Chat> {
    const response = await apiClient.patch<Chat>(
        API_ENDPOINTS.CHAT_DETAIL(chatId),
        data
    );
    return response.data;
}

/**
 * Delete chat
 */
export async function deleteChat(chatId: string): Promise<void> {
    await apiClient.delete(API_ENDPOINTS.CHAT_DETAIL(chatId));
}

