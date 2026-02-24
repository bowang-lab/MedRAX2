import createClient from "openapi-fetch";
import type { paths } from "../types/openapi";
import { API_CONFIG, API_SECRET_CONFIG } from "../config/api";

function getAuthHeaders(): HeadersInit {
  if (typeof window === "undefined") return {};
  const token = localStorage.getItem("medrax_auth_token");
  const apiSecret = API_SECRET_CONFIG.getSecret();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (apiSecret) headers["X-API-Secret"] = apiSecret;
  return headers;
}

export const openapiClient = createClient<paths>({
  baseUrl: API_CONFIG.baseURL,
});

export function authHeaders(): HeadersInit {
  return getAuthHeaders();
}
