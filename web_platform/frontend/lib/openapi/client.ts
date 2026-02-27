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
  fetch: async (input, init) => {
    const headers = new Headers(init?.headers || {});
    const auth = getAuthHeaders();

    for (const [key, value] of Object.entries(auth)) {
      if (!headers.has(key)) {
        headers.set(key, value);
      }
    }

    return fetch(input, {
      ...init,
      headers,
    });
  },
});

export function authHeaders(): HeadersInit {
  return getAuthHeaders();
}
