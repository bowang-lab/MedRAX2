/**
 * Doctor Profile API Functions
 *
 * API calls for updating doctor profile.
 */

import { openapiClient, authHeaders } from "../openapi/client";
import type { Doctor } from "../types/doctor";
import type { ApiDoctorResponse } from "../types/api";
import { toUiDoctor } from "../openapi/transformers";

/**
 * Update doctor profile (name)
 *
 * Note: Backend endpoint uses PATCH /api/auth/me which returns DoctorResponse directly
 * (verified in backend/app/api/auth.py line 90: response_model=DoctorResponse)
 */
export async function updateDoctor(data: { name: string }): Promise<Doctor> {
  const { data: response, error } = await openapiClient.PATCH("/api/auth/me", {
    body: { name: data.name },
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!response) throw new Error("Failed to update doctor");

  return toUiDoctor(response as ApiDoctorResponse);
}

/**
 * Update password
 *
 * TODO: Backend does not currently have a password update endpoint.
 * Need to add PATCH /api/auth/me/password endpoint to backend API.
 */
export async function updatePassword(data: {
  currentPassword: string;
  newPassword: string;
}): Promise<void> {
  // TODO: Backend does not currently have a password update endpoint
  // When implemented, use: await apiClient.PATCH('/api/auth/me/password', { body: { ... } })
  console.warn("Password update attempted but not yet implemented:", { hasData: !!data });
  throw new Error("Password update not yet implemented in backend API");
}
