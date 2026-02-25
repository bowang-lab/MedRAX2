/**
 * Scan (Medical Image) Types
 *
 * Scans can be attached to specific messages or belong to the chat generally.
 * Supported formats match backend ALLOWED_EXTENSIONS in config.py:
 * jpg, jpeg, png, gif, dcm, dicom
 */

export type ScanFileType = "jpg" | "jpeg" | "png" | "gif" | "dcm" | "dicom";

export interface Scan {
  id: string;
  chatId: string;
  filePath: string;
  displayPath: string; // Path for display (DICOM converted to image)
  fileType: ScanFileType;
  fileSize: number; // Size in bytes
  uploadedAt: string;
}

export interface ScanUploadRequest {
  chatId: string;
  messageId?: string; // Optional: attach to specific message
  file: File;
}

export interface ScanUploadResponse {
  scan: Scan;
  message: string;
}
