/**
 * API Configuration
 *
 * Central configuration for API base URL and settings.
 */

export const API_CONFIG = {
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  timeout: parseInt(process.env.NEXT_PUBLIC_API_TIMEOUT || "30000"),
  headers: {
    "Content-Type": "application/json",
  },
};

// API Secret Management
// Secret is NOT stored in env vars (to prevent browser exposure)
// Instead, users enter it once and it's stored in localStorage
export const API_SECRET_CONFIG = {
  storageKey: "medrax_api_secret",

  getSecret(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem(this.storageKey);
  },

  setSecret(secret: string): void {
    if (typeof window === "undefined") return;
    localStorage.setItem(this.storageKey, secret);
  },

  clearSecret(): void {
    if (typeof window === "undefined") return;
    localStorage.removeItem(this.storageKey);
  },

  hasSecret(): boolean {
    return this.getSecret() !== null;
  },
};

export const API_ENDPOINTS = {
  // Auth
  AUTH_REGISTER: "/api/auth/register",
  AUTH_LOGIN: "/api/auth/login",
  AUTH_LOGOUT: "/api/auth/logout",
  AUTH_ME: "/api/auth/me",

  // Patients
  PATIENTS: "/api/patients",
  PATIENT_DETAIL: (id: string) => `/api/patients/${id}`,
  PATIENT_CHATS: (id: string) => `/api/patients/${id}/chats`,
  PATIENT_SCANS: (id: string) => `/api/patients/${id}/scans`,

  // Chats
  CHAT_DETAIL: (id: string) => `/api/chats/${id}`,
  CHAT_MESSAGES: (id: string) => `/api/chats/${id}/messages`,
  CHAT_SCANS: (id: string) => `/api/chats/${id}/scans`,
  CHAT_STREAM: (id: string) => `/api/chats/${id}/stream`,

  // Messages
  MESSAGE_DETAIL: (id: string) => `/api/messages/${id}`,
  MESSAGE_EXECUTIONS: (id: string) => `/api/messages/${id}/executions`,

  // Tool Executions
  EXECUTION_DETAIL: (id: string) => `/api/tools/executions/${id}`,

  // Questions
  QUESTIONS: "/api/questions",
  QUESTION_DETAIL: (id: string) => `/api/questions/${id}`,

  // Memory Management
  CHAT_MEMORY_CLEAR: (chatId: string) => `/api/chats/${chatId}/memory/clear`,
  CHAT_MEMORY_STATS: (chatId: string) => `/api/chats/${chatId}/memory/stats`,
  SYSTEM_MEMORY_CLEANUP: "/api/system/memory/cleanup",

  // Tools
  TOOLS: "/api/tools",
  TOOL_LOAD: (id: string) => `/api/tools/${id}/load`,
  TOOL_UNLOAD: (id: string) => `/api/tools/${id}/unload`,
  TOOLS_BULK_LOAD: "/api/tools/bulk-load",
};
