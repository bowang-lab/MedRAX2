/**
 * Image Utilities
 *
 * Helpers for handling image URLs and paths.
 */

import { API_CONFIG } from "../config/api";

/**
 * Get full image URL from backend display path.
 * Canonical paths: /medrax/uploads/..., /medrax/generated/...
 * Legacy /uploads/ and /temp/ are rewritten for older stored rows.
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

  let normalizedPath = displayPath.startsWith("/") ? displayPath : `/${displayPath}`;
  if (normalizedPath.startsWith("/uploads/")) {
    normalizedPath = `/medrax/uploads/${normalizedPath.slice("/uploads/".length)}`;
  } else if (normalizedPath.startsWith("/temp/")) {
    normalizedPath = `/medrax/generated/${normalizedPath.slice("/temp/".length)}`;
  }

  return `${API_CONFIG.baseURL}${normalizedPath}`;
}
