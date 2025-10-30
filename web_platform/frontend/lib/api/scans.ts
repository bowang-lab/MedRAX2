/**
 * Scan API Functions
 * 
 * API calls for scan/image management.
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';
import type { Scan } from '../types/scan';

/**
 * Backend scan response interface (camelCase - backend uses serialization_alias)
 */
interface ScanAPIResponse {
    id: string;
    chatId: string;
    filePath: string;
    displayPath: string;
    fileType: string;
    fileSize: number;
    uploadedAt: string;
}

/**
 * Map backend scan response to frontend Scan type
 * Backend now sends camelCase directly, so this is a simple passthrough
 */
function mapScanResponse(data: ScanAPIResponse): Scan {
    // Debug: Log the raw response to help track down any issues
    if (!data.displayPath) {
        console.error('⚠️ Scan response missing displayPath:', data);
    }
    
    return {
        id: data.id,
        chatId: data.chatId,
        filePath: data.filePath,
        displayPath: data.displayPath,
        fileType: data.fileType as any,
        fileSize: data.fileSize,
        uploadedAt: data.uploadedAt,
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
    console.log(`📤 Uploading ${files.length} file(s) to chat ${chatId}:`, files.map(f => ({ name: f.name, size: f.size })));
    
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

    console.log(`📥 Received upload response:`, response.data);
    
    const scans = response.data.map(mapScanResponse);
    console.log(`✅ Mapped scans:`, scans.map(s => ({ id: s.id, displayPath: s.displayPath })));
    
    return scans;
}

/**
 * Delete a scan
 */
export async function deleteScan(scanId: string): Promise<void> {
    await apiClient.delete(API_ENDPOINTS.SCAN_DETAIL(scanId));
}

