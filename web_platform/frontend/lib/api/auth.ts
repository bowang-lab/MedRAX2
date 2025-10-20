/**
 * Auth API Functions
 * 
 * API calls for authentication (login, register, logout).
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';
import type {
    Doctor,
    DoctorRegistration,
    DoctorLogin,
    AuthSession,
} from '../types/doctor';

/**
 * Register a new doctor
 */
export async function registerDoctor(
    data: DoctorRegistration
): Promise<AuthSession> {
    const response = await apiClient.post<{
        doctor: Doctor;
        access_token: string;
        token_type: string;
    }>(API_ENDPOINTS.AUTH_REGISTER, data);

    return {
        doctor: response.data.doctor,
        token: response.data.access_token, // Fixed: use access_token from backend
        expiresAt: '', // Backend doesn't provide this, can be calculated if needed
    };
}

/**
 * Login doctor
 */
export async function loginDoctor(data: DoctorLogin): Promise<AuthSession> {
    const response = await apiClient.post<{
        doctor: Doctor;
        access_token: string;
        token_type: string;
    }>(API_ENDPOINTS.AUTH_LOGIN, data);

    return {
        doctor: response.data.doctor,
        token: response.data.access_token, // Fixed: use access_token from backend
        expiresAt: '', // Backend doesn't provide this, can be calculated if needed
    };
}

/**
 * Logout doctor
 */
export async function logoutDoctor(): Promise<void> {
    await apiClient.post(API_ENDPOINTS.AUTH_LOGOUT);
}

/**
 * Get current doctor info
 */
export async function getCurrentDoctor(): Promise<Doctor> {
    const response = await apiClient.get<Doctor>(
        API_ENDPOINTS.AUTH_ME
    );
    return response.data; // Backend returns doctor directly, not wrapped
}

