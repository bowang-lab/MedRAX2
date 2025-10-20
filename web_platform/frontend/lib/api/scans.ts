/**
 * Scan API Functions
 * 
 * API calls for scan/image management.
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';
import type { Scan } from '../types/scan';

/**
 * Get all scans for a chat
 */
export async function getScans(chatId: string): Promise<Scan[]> {
    const response = await apiClient.get<Scan[]>(
        API_ENDPOINTS.CHAT_SCANS(chatId)
    );
    return response.data;
}

/**
 * Get all scans for a patient (across all chats)
 */
export async function getPatientScans(patientId: string): Promise<Scan[]> {
    const response = await apiClient.get<Scan[]>(
        API_ENDPOINTS.PATIENT_SCANS(patientId)
    );
    return response.data;
}

/**
 * Upload scan(s) to a chat
 */
export async function uploadScans(
    chatId: string,
    files: File[]
): Promise<Scan[]> {
    const formData = new FormData();
    files.forEach((file) => {
        formData.append('files', file);
    });

    const response = await apiClient.post<Scan[]>(
        API_ENDPOINTS.CHAT_SCANS(chatId),
        formData,
        {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        }
    );

    return response.data;
}

/**
 * Delete a scan
 */
export async function deleteScan(scanId: string): Promise<void> {
    await apiClient.delete(API_ENDPOINTS.SCAN_DETAIL(scanId));
}

