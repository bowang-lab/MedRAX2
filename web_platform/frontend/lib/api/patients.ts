/**
 * Patient API Functions
 * 
 * API calls for patient management.
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';
import type { PatientWithStats } from '../types/patient';

/**
 * Get all patients for current doctor
 */
export async function getPatients(): Promise<PatientWithStats[]> {
    const response = await apiClient.get<PatientWithStats[]>(
        API_ENDPOINTS.PATIENTS
    );
    return response.data;
}

/**
 * Create new patient
 */
export async function createPatient(data: {
    name?: string | null;
}): Promise<PatientWithStats> {
    const response = await apiClient.post<PatientWithStats>(
        API_ENDPOINTS.PATIENTS,
        data
    );
    return response.data;
}

/**
 * Update patient
 */
export async function updatePatient(
    id: string,
    data: { name?: string | null }
): Promise<PatientWithStats> {
    const response = await apiClient.patch<PatientWithStats>(
        API_ENDPOINTS.PATIENT_DETAIL(id),
        data
    );
    return response.data;
}

/**
 * Delete patient
 */
export async function deletePatient(id: string): Promise<void> {
    await apiClient.delete(API_ENDPOINTS.PATIENT_DETAIL(id));
}

