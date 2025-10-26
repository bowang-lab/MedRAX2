/**
 * API Configuration
 * 
 * Central configuration for API base URL and settings.
 */

export const API_CONFIG = {
    baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    timeout: 30000,  // 30 seconds
    headers: {
        'Content-Type': 'application/json',
    },
};

export const API_ENDPOINTS = {
    // Auth
    AUTH_REGISTER: '/api/auth/register',
    AUTH_LOGIN: '/api/auth/login',
    AUTH_LOGOUT: '/api/auth/logout',
    AUTH_ME: '/api/auth/me',

    // Patients
    PATIENTS: '/api/patients',
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

    // Scans
    SCAN_DETAIL: (id: string) => `/api/scans/${id}`,

    // Tool Executions
    EXECUTION_DETAIL: (id: string) => `/api/tools/executions/${id}`,
    EXECUTION_LOGS: (id: string) => `/api/tools/executions/${id}/logs`,
    EXECUTION_RESULT: (id: string) => `/api/tools/executions/${id}/result`,

    // Questions
    QUESTIONS: '/api/questions',
    QUESTION_DETAIL: (id: string) => `/api/questions/${id}`,

    // Tools
    TOOLS: '/api/tools',
    TOOL_LOAD: (id: string) => `/api/tools/${id}/load`,
    TOOL_UNLOAD: (id: string) => `/api/tools/${id}/unload`,
    TOOLS_BULK_LOAD: '/api/tools/bulk-load',
};

