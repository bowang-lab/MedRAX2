/**
 * Scan API Functions
 * 
 * API calls for scan/image management.
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';
import type { Scan } from '../types/scan';

/**
 * Backend scan response interface (snake_case)
 */
interface ScanAPIResponse {
    id: string;
    chat_id: string;
    file_path: string;
    display_path: string;
    file_type: string;
    file_size: number;
    uploaded_at: string;
}

/**
 * Map backend scan response to frontend Scan type
 */
function mapScanResponse(data: ScanAPIResponse): Scan {
    return {
        id: data.id,
        chatId: data.chat_id,
        filePath: data.file_path,
        displayPath: data.display_path,
        fileType: data.file_type as any,
        fileSize: data.file_size,
        uploadedAt: data.uploaded_at,
    };
}

/**
 * Get all scans for a chat
 */
export async function getScans(chatId: string): Promise<Scan[]> {
    const response = await apiClient.get<ScanAPIResponse[]>(
        API_ENDPOINTS.CHAT_SCANS(chatId)
    );
    return response.data.map(mapScanResponse);
}

/**
 * Get all scans for a patient (across all chats)
 */
export async function getPatientScans(patientId: string): Promise<Scan[]> {
    const response = await apiClient.get<ScanAPIResponse[]>(
        API_ENDPOINTS.PATIENT_SCANS(patientId)
    );
    return response.data.map(mapScanResponse);
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

    const response = await apiClient.post<ScanAPIResponse[]>(
        API_ENDPOINTS.CHAT_SCANS(chatId),
        formData,
        {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        }
    );

    return response.data.map(mapScanResponse);
}

/**
 * Delete a scan
 */
export async function deleteScan(scanId: string): Promise<void> {
    await apiClient.delete(API_ENDPOINTS.SCAN_DETAIL(scanId));
}

