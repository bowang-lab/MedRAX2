/**
 * Auth API Functions
 *
 * API calls for authentication (login, register, logout).
 */

import { openapiClient, authHeaders } from "../openapi/client";
import type { Doctor, DoctorRegistration, DoctorLogin, AuthSession } from "../types/doctor";
import type { ApiTokenResponse, ApiDoctorResponse } from "../types/api";
import { toAuthSession, toUiDoctor } from "../openapi/transformers";

/**
 * Register a new doctor
 */
export async function registerDoctor(
  data: DoctorRegistration
): Promise<AuthSession> {
  const { data: response, error } = await openapiClient.POST("/api/auth/register", {
    body: { name: data.name, password: data.password },
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!response) throw new Error("Failed to register");
  return toAuthSession(response as ApiTokenResponse);
}

/**
 * Login doctor
 */
export async function loginDoctor(data: DoctorLogin): Promise<AuthSession> {
  const { data: response, error } = await openapiClient.POST("/api/auth/login", {
    body: { name: data.name, password: data.password },
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!response) throw new Error("Failed to login");
  return toAuthSession(response as ApiTokenResponse);
}

/**
 * Logout doctor
 */
export async function logoutDoctor(): Promise<void> {
  const { error } = await openapiClient.POST("/api/auth/logout", {});
  if (error) throw error;
}

/**
 * Get current doctor info
 */
export async function getCurrentDoctor(): Promise<Doctor> {
  const { data, error } = await openapiClient.GET("/api/auth/me", {
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("Failed to get doctor info");
  return toUiDoctor(data as ApiDoctorResponse);
}
