/**
 * Doctor Profile API Functions
 * 
 * API calls for updating doctor profile.
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';
import type { Doctor } from '../types/doctor';

/**
 * Update doctor profile (name)
 */
export async function updateDoctor(
    id: string,
    data: { name: string }
): Promise<Doctor> {
    const response = await apiClient.patch<{ doctor: Doctor }>(
        API_ENDPOINTS.AUTH_ME, // Assuming profile update uses the same endpoint
        data
    );
    return response.data.doctor;
}

/**
 * Update password
 */
export async function updatePassword(
    id: string,
    data: { currentPassword: string; newPassword: string }
): Promise<void> {
    await apiClient.post(`${API_ENDPOINTS.AUTH_ME}/password`, data);
}

