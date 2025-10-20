/**
 * API Client
 * 
 * Axios client with authentication interceptors and error handling.
 */

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import { API_CONFIG } from '../config/api';
import { AUTH_CONFIG } from '../config/app';
import { ApiError } from '../types';

/**
 * Backend error response structure
 * FastAPI typically returns errors in this format
 */
interface BackendErrorResponse {
    detail?: string | { msg: string; type: string }[];
    message?: string;
    [key: string]: unknown;
}

// Create axios instance
export const apiClient: AxiosInstance = axios.create({
    baseURL: API_CONFIG.baseURL,
    timeout: API_CONFIG.timeout,
    headers: API_CONFIG.headers,
});

// Request interceptor - Add auth token
apiClient.interceptors.request.use(
    (config: InternalAxiosRequestConfig) => {
        const token = localStorage.getItem(AUTH_CONFIG.tokenKey);

        if (token && config.headers) {
            config.headers.Authorization = `Bearer ${token}`;
        }

        return config;
    },
    (error: AxiosError) => {
        return Promise.reject(error);
    }
);

// Response interceptor - Handle errors
apiClient.interceptors.response.use(
    (response) => {
        return response;
    },
    (error: AxiosError<BackendErrorResponse>) => {
        if (error.response) {
            // Server responded with error status
            const backendError = error.response.data;
            let errorMessage = error.message;

            // Extract error message from various backend formats
            if (typeof backendError?.detail === 'string') {
                errorMessage = backendError.detail;
            } else if (Array.isArray(backendError?.detail)) {
                errorMessage = backendError.detail.map(e => e.msg).join(', ');
            } else if (backendError?.message) {
                errorMessage = backendError.message;
            }

            const apiError: ApiError = {
                message: errorMessage,
                code: error.response.status.toString(),
                details: error.response.data,
            };

            // Handle 401 - Unauthorized (token expired or invalid)
            if (error.response.status === 401) {
                // Clear auth data
                localStorage.removeItem(AUTH_CONFIG.tokenKey);
                localStorage.removeItem(AUTH_CONFIG.doctorKey);

                // Redirect to login if not already there
                if (!window.location.pathname.includes('/login')) {
                    window.location.href = '/login';
                }
            }

            return Promise.reject(apiError);
        } else if (error.request) {
            // Request made but no response
            const apiError: ApiError = {
                message: 'No response from server. Please check your connection.',
                code: 'NETWORK_ERROR',
            };
            return Promise.reject(apiError);
        } else {
            // Something else happened
            const apiError: ApiError = {
                message: error.message || 'An unexpected error occurred',
                code: 'UNKNOWN_ERROR',
            };
            return Promise.reject(apiError);
        }
    }
);

export default apiClient;

