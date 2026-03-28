/**
 * API Response Types
 *
 * These types match the OpenAPI schema exactly (using snake_case as the backend returns).
 * They are imported from the generated openapi.d.ts file.
 *
 * DO NOT define custom interfaces here - use the generated types from openapi.d.ts
 */

import type { components } from "./openapi";

// ============================================================================
// Re-export OpenAPI schema types for use in API modules
// ============================================================================

/** Message with attached scans and tool executions */
export type ApiMessageWithDetails = components["schemas"]["MessageWithDetails"];

/** Chat response from API */
export type ApiChatResponse = components["schemas"]["ChatResponse"];

/** Patient with statistics */
export type ApiPatientWithStats = components["schemas"]["PatientWithStats"];

/** Scan response (NOTE: This uses camelCase due to backend serialization_alias) */
export type ApiScanResponse = components["schemas"]["ScanResponse"];

/** Doctor profile response */
export type ApiDoctorResponse = components["schemas"]["DoctorResponse"];

/** Suggested question response */
export type ApiQuestionResponse = components["schemas"]["QuestionResponse"];

/** Tool execution response */
export type ApiToolExecutionResponse = components["schemas"]["ToolExecutionResponse"];

/** Tool execution log */
export type ApiToolExecutionLogResponse =
  components["schemas"]["ToolExecutionLogResponse"];

/** Tool execution result */
export type ApiToolExecutionResultResponse =
  components["schemas"]["ToolExecutionResultResponse"];

/** Tool execution detail (execution + logs + result) */
export type ApiToolExecutionDetailResponse =
  components["schemas"]["ToolExecutionDetailResponse"];

/** Authentication token response */
export type ApiTokenResponse = components["schemas"]["TokenResponse"];

/** Question create request */
export type ApiQuestionCreate = components["schemas"]["QuestionCreate"];

/** Patient create request */
export type ApiPatientCreate = components["schemas"]["PatientCreate"];

/** Patient update request */
export type ApiPatientUpdate = components["schemas"]["PatientUpdate"];

/** Chat create request */
export type ApiChatCreate = components["schemas"]["ChatCreate"];

/** Chat update request */
export type ApiChatUpdate = components["schemas"]["ChatUpdate"];

/** Message create request */
export type ApiMessageCreate = components["schemas"]["MessageCreate"];

/** Tool bulk load request */
export type ApiToolBulkLoadRequest =
  components["schemas"]["ToolBulkLoadRequest"];

/** Tool bulk load response */
export type ApiToolBulkLoadResponse =
  components["schemas"]["ToolBulkLoadResponse"];

/** Memory statistics response */
export type ApiMemoryStatsResponse = components["schemas"]["MemoryStatsResponse"];

/** Clear memory response */
export type ApiClearMemoryResponse =
  components["schemas"]["ClearMemoryResponse"];

/** System cleanup statistics response */
export type ApiSystemCleanupStatsResponse =
  components["schemas"]["SystemCleanupStatsResponse"];

/** Tool information response */
export type ApiToolInfo = components["schemas"]["ToolInfo"];
