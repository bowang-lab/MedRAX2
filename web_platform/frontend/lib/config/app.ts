/**
 * Application Configuration
 *
 * General app settings and constants.
 */

export const APP_CONFIG = {
  name: "MedRAX Platform",
  description: "Medical Reasoning Agent for Chest X-ray Analysis",
  version: "2.0.0",
};

export const AUTH_CONFIG = {
  tokenKey: "medrax_auth_token",
  doctorKey: "medrax_doctor",
  tokenExpiryDays: 30,
};

export const UI_CONFIG = {
  sidebarMinWidth: 250,
  sidebarMaxWidth: 400,
  sidebarDefaultWidth: 320,

  maxFileSize: 100 * 1024 * 1024, // 100MB (matches backend MAX_UPLOAD_SIZE)
  allowedFileTypes: ["image/jpeg", "image/png", "image/gif", "application/dicom"],
  allowedFileExtensions: [".jpg", ".jpeg", ".png", ".gif", ".dcm", ".dicom"],
};

export const PAGINATION_CONFIG = {
  defaultPageSize: 20,
  maxPageSize: 100,
};
