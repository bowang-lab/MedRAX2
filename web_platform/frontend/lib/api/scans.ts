/**
 * Scan API Functions
 *
 * API calls for scan/image management.
 */

import { openapiClient, authHeaders } from "../openapi/client";
import { API_CONFIG } from "../config/api";
import type { Scan } from "../types/scan";
import type { ApiScanResponse } from "../types/api";
import { toUiScan } from "../openapi/transformers";

/**
 * Get all scans for a chat
 * Backend always returns List[ScanResponse] (never null)
 */
export async function getScans(chatId: string): Promise<Scan[]> {
  const { data, error } = await openapiClient.GET("/api/chats/{chat_id}/scans", {
    params: { path: { chat_id: chatId } },
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("No data returned from server");
  return data.map((scan: ApiScanResponse) => toUiScan(scan));
}

/**
 * Get all scans for a patient (across all chats)
 * Backend always returns List[ScanResponse] (never null)
 */
export async function getPatientScans(patientId: string): Promise<Scan[]> {
  const { data, error } = await openapiClient.GET("/api/patients/{patient_id}/scans", {
    params: { path: { patient_id: patientId } },
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("No data returned from server");
  return data.map((scan: ApiScanResponse) => toUiScan(scan));
}

/**
 * Upload scan(s) to a chat
 *
 * Note: Uses native fetch() instead of openapi-fetch because the OpenAPI schema
 * incorrectly types the multipart/form-data body as { files: string[] } instead
 * of properly supporting File uploads. This is a known limitation of OpenAPI 3.0
 * schema generation for file uploads in FastAPI.
 *
 * TODO: When backend is updated to OpenAPI 3.1 with proper file upload types,
 * migrate this to use openapiClient.POST()
 */
export async function uploadScans(chatId: string, files: File[]): Promise<Scan[]> {
  console.log(
    `📤 Uploading ${files.length} file(s) to chat ${chatId}:`,
    files.map((f) => ({ name: f.name, size: f.size }))
  );

  const formData = new FormData();
  files.forEach((file) => {
    formData.append("files", file);
  });

  // Use native fetch for multipart/form-data uploads
  const response = await fetch(`${API_CONFIG.baseURL}/api/chats/${chatId}/scans`, {
    method: "POST",
    headers: authHeaders(),
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(errorData.detail || `Upload failed with status ${response.status}`);
  }

  const data: ApiScanResponse[] = await response.json();
  console.log(`📥 Received upload response:`, data);

  // Backend always returns List[ScanResponse] (never null)
  if (!data) throw new Error("No data returned from upload");
  const scans = data.map((scan: ApiScanResponse) => toUiScan(scan));
  console.log(`✅ Mapped scans:`, scans.map((s) => ({ id: s.id, displayPath: s.displayPath })));

  return scans;
}

/**
 * Delete a scan
 */
export async function deleteScan(scanId: string): Promise<void> {
  const { error } = await openapiClient.DELETE("/api/{scan_id}", {
    params: { path: { scan_id: scanId } },
    headers: authHeaders(),
  });
  if (error) throw error;
}
