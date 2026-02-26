/**
 * Patient API Functions
 *
 * API calls for patient management.
 */

import { openapiClient, authHeaders } from "../openapi/client";
import type { PatientWithStats } from "../types/patient";
import type { ApiPatientWithStats, ApiPatientCreate, ApiPatientUpdate } from "../types/api";
import { toUiPatientWithStats } from "../openapi/transformers";

/**
 * Get all patients for current doctor
 * Backend always returns List[PatientWithStats] (never null)
 */
export async function getPatients(): Promise<PatientWithStats[]> {
  const { data, error } = await openapiClient.GET("/api/patients", {
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("No data returned from server");
  return data.map((patient: ApiPatientWithStats) => toUiPatientWithStats(patient));
}

/**
 * Create new patient
 */
export async function createPatient(
  data: { name?: string | null }
): Promise<PatientWithStats> {
  const requestBody: ApiPatientCreate = {
    name: data.name ?? null,
  };

  const { data: response, error } = await openapiClient.POST("/api/patients", {
    body: requestBody,
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!response) throw new Error("Failed to create patient");
  return toUiPatientWithStats(response);
}

/**
 * Update patient
 */
export async function updatePatient(
  id: string,
  data: { name?: string | null }
): Promise<PatientWithStats> {
  const requestBody: ApiPatientUpdate = {
    name: data.name ?? null,
  };

  const { data: response, error } = await openapiClient.PATCH(
    "/api/patients/{patient_id}",
    {
      params: { path: { patient_id: id } },
      body: requestBody,
      headers: authHeaders(),
    }
  );
  if (error) throw error;
  if (!response) throw new Error("Failed to update patient");
  return toUiPatientWithStats(response);
}

/**
 * Delete patient
 */
export async function deletePatient(id: string): Promise<void> {
  const { error } = await openapiClient.DELETE("/api/patients/{patient_id}", {
    params: { path: { patient_id: id } },
    headers: authHeaders(),
  });
  if (error) throw error;
}
