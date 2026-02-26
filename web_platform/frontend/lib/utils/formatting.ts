/**
 * Formatting Utilities
 *
 * Functions for formatting dates, times, numbers, etc.
 */

/**
 * Format ISO date string to readable format
 */
export function formatDate(isoString: string | undefined | null): string {
  if (!isoString) {
    return "Today";
  }

  const date = new Date(isoString);

  // Check if date is valid
  if (isNaN(date.getTime())) {
    return "Today";
  }

  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  }).format(date);
}

/**
 * Format ISO date string to time only
 */
export function formatTime(isoString: string | undefined | null): string {
  if (!isoString) {
    return "Now";
  }

  const date = new Date(isoString);

  // Check if date is valid
  if (isNaN(date.getTime())) {
    return "Now";
  }

  return new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }).format(date);
}

/**
 * Format ISO date string to date and time
 */
export function formatDateTime(isoString: string | undefined | null): string {
  if (!isoString) {
    return "Just now";
  }

  const date = new Date(isoString);

  // Check if date is valid
  if (isNaN(date.getTime())) {
    return "Just now";
  }

  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }).format(date);
}

/**
 * Generate a default chat name based on current date/time
 * Format: "01/15/2025, 2:30 PM"
 * @param date - Optional date to use (defaults to current date/time)
 * @returns Formatted date string suitable for chat name
 */
export function generateChatName(date: Date = new Date()): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }).format(date);
}

/** @deprecated Use generateChatName instead */
export const formatChatName = generateChatName;

/**
 * Format relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffSec < 60) {
    return "Just now";
  } else if (diffMin < 60) {
    return `${diffMin} minute${diffMin !== 1 ? "s" : ""} ago`;
  } else if (diffHour < 24) {
    return `${diffHour} hour${diffHour !== 1 ? "s" : ""} ago`;
  } else if (diffDay < 7) {
    return `${diffDay} day${diffDay !== 1 ? "s" : ""} ago`;
  } else {
    return formatDate(isoString);
  }
}

/**
 * Format file size in bytes to human-readable format
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 Bytes";

  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
}

/**
 * Format execution time in milliseconds to readable format
 */
export function formatExecutionTime(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  } else if (ms < 60000) {
    return `${(ms / 1000).toFixed(1)}s`;
  } else {
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  }
}

/**
 * Truncate text with ellipsis
 */
export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + "...";
}
