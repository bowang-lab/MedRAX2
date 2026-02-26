/**
 * Application Constants
 *
 * Centralized constants for routes and defaults.
 * Configuration values are in lib/config/app.ts
 */

/**
 * Application routes
 */
export const ROUTES = {
  HOME: "/",
  LOGIN: "/login",
  REGISTER: "/register",
  APP: "/app",
  SETTINGS: "/app/settings",
  PROFILE: "/app/profile",
} as const;

/**
 * Chat defaults
 */
export const CHAT_DEFAULTS = {
  /** Default name for the first chat when a patient is created */
  INITIAL_CHAT_NAME: "Initial Consultation",
  SYSTEM_WELCOME: "Welcome! Upload medical images to begin analysis.",
} as const;

/**
 * Default suggested questions shown to doctors
 */
export const DEFAULT_SUGGESTED_QUESTIONS = [
  "Is there pneumonia?",
  "Measure heart size",
  "What abnormalities do you see?",
  "Generate a report",
] as const;
