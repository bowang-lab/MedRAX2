import createClient, { type Middleware } from "openapi-fetch";
import type { paths } from "../types/openapi";
import { API_CONFIG, API_SECRET_CONFIG } from "../config/api";

function getAuthHeaders(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const token = localStorage.getItem("medrax_auth_token");
  const apiSecret = API_SECRET_CONFIG.getSecret();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (apiSecret) headers["X-API-Secret"] = apiSecret;
  return headers;
}

const authMiddleware: Middleware = {
  onRequest: ({ request }) => {
    const auth = getAuthHeaders();
    for (const [key, value] of Object.entries(auth)) {
      if (!request.headers.has(key)) {
        request.headers.set(key, value);
      }
    }
    return request;
  },
};

export const openapiClient = createClient<paths>({
  baseUrl: API_CONFIG.baseURL,
});

openapiClient.use(authMiddleware);

export function authHeaders(): HeadersInit {
  return getAuthHeaders();
}
