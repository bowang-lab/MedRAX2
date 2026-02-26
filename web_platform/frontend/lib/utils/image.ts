/**
 * Image Utilities
 *
 * Helpers for handling image URLs and paths.
 */

import { API_CONFIG } from "../config/api";

/**
 * Get full image URL from backend display path
 * Backend returns paths like "/uploads/chats/xyz/file.jpg" for scans
 * But tool_executions.image_paths contains "uploads/chats/xyz/file.jpg" without leading /
 * Frontend needs full URL like "http://localhost:8000/uploads/chats/xyz/file.jpg"
 */
export function getImageUrl(displayPath: string | null | undefined): string | null {
  if (!displayPath || displayPath.trim() === "") return null;

  // If already a full URL, return as-is
  if (displayPath.startsWith("http://") || displayPath.startsWith("https://")) {
    return displayPath;
  }

  // If it's a data URL (base64), return as-is
  if (displayPath.startsWith("data:")) {
    return displayPath;
  }

  // Ensure the path starts with / for proper URL construction
  const normalizedPath = displayPath.startsWith("/") ? displayPath : `/${displayPath}`;

  // Prepend backend base URL
  return `${API_CONFIG.baseURL}${normalizedPath}`;
}
