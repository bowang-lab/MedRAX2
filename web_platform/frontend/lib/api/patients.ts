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
interface PatientAPIResponse {
    id: string;
    name: string | null;
    doctor_id: string;
    created_at: string;
    last_activity_at: string | null;
    total_chats: number;
    total_scans: number;
}

export async function getPatients(): Promise<PatientWithStats[]> {
    const response = await apiClient.get<PatientAPIResponse[]>(
        API_ENDPOINTS.PATIENTS
    );
    // Map backend fields to frontend fields
    return response.data.map(patient => ({
        id: patient.id,
        name: patient.name,
        doctorId: patient.doctor_id,
        createdAt: patient.created_at,
        lastActivityAt: patient.last_activity_at,
        chatCount: patient.total_chats || 0,
        scanCount: patient.total_scans || 0,
    }));
}

/**
 * Create new patient
 */
export async function createPatient(data: {
    name?: string | null;
}): Promise<PatientWithStats> {
    const response = await apiClient.post<PatientAPIResponse>(
        API_ENDPOINTS.PATIENTS,
        data
    );
    const patient = response.data;
    return {
        id: patient.id,
        name: patient.name,
        doctorId: patient.doctor_id,
        createdAt: patient.created_at,
        lastActivityAt: patient.last_activity_at,
        chatCount: patient.total_chats || 0,
        scanCount: patient.total_scans || 0,
    };
}

/**
 * Update patient
 */
export async function updatePatient(
    id: string,
    data: { name?: string | null }
): Promise<PatientWithStats> {
    const response = await apiClient.patch<PatientAPIResponse>(
        API_ENDPOINTS.PATIENT_DETAIL(id),
        data
    );
    const patient = response.data;
    return {
        id: patient.id,
        name: patient.name,
        doctorId: patient.doctor_id,
        createdAt: patient.created_at,
        lastActivityAt: patient.last_activity_at,
        chatCount: patient.total_chats || 0,
        scanCount: patient.total_scans || 0,
    };
}

/**
 * Delete patient
 */
export async function deletePatient(id: string): Promise<void> {
    await apiClient.delete(API_ENDPOINTS.PATIENT_DETAIL(id));
}

