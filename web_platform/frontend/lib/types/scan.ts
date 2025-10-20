/**
 * Scan (Medical Image) Types
 * 
 * Scans can be attached to specific messages or belong to the chat generally.
 * Supports DICOM, JPG, and PNG formats.
 */

export type ScanFileType = 'dicom' | 'jpg' | 'png';

export interface Scan {
    id: string;
    chatId: string;
    messageId: string | null;  // null = general chat scan, not message-specific
    filePath: string;
    displayPath: string;  // Path for display (DICOM converted to image)
    fileType: ScanFileType;
    uploadedAt: string;
}

export interface ScanUploadRequest {
    chatId: string;
    messageId?: string;  // Optional: attach to specific message
    file: File;
}

export interface ScanUploadResponse {
    scan: Scan;
    message: string;
}

