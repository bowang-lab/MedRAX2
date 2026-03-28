/**
 * Central Type Exports
 *
 * Re-export all types for easy importing throughout the app.
 */

export * from "./doctor";
export * from "./patient";
export * from "./chat";
export * from "./message";
export * from "./scan";
export * from "./tool";
export * from "./question";

/**
 * API Error Type
 * Used for standardized error handling across the app
 */
export interface ApiError {
  message: string;
  code?: string;
  // Backend can return various error detail formats
  details?: unknown;
}
