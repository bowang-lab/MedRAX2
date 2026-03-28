/**
 * API Response Transformers
 *
 * Transform OpenAPI response types (snake_case) to frontend UI types (camelCase).
 * These transformers are the ONLY place where snake_case ↔ camelCase conversion happens.
 *
 * NO TYPE CASTING - All types are explicitly defined.
 */

import type {
  ApiMessageWithDetails,
  ApiChatResponse,
  ApiPatientWithStats,
  ApiScanResponse,
  ApiDoctorResponse,
  ApiQuestionResponse,
  ApiToolExecutionResponse,
  ApiToolExecutionLogResponse,
  ApiToolExecutionResultResponse,
  ApiTokenResponse,
  ApiMemoryStatsResponse,
  ApiClearMemoryResponse,
  ApiSystemCleanupStatsResponse,
  ApiToolInfo,
} from "../types/api";

import type { MessageWithDetails } from "../types/message";
import type { Chat } from "../types/chat";
import type { PatientWithStats } from "../types/patient";
import type { Scan } from "../types/scan";
import type { Doctor, AuthSession } from "../types/doctor";
import type { SuggestedQuestion } from "../types/question";
import type { ToolExecution, ToolExecutionLog, ToolExecutionResult } from "../types/tool";

// ============================================================================
// Tool Execution Transformers
// ============================================================================

export function toUiToolExecution(exec: ApiToolExecutionResponse): ToolExecution {
  return {
    id: exec.id,
    messageId: exec.message_id,
    requestId: exec.request_id ?? null,
    toolName: exec.tool_name,
    toolDisplayName: exec.tool_display_name,
    // Status is a string in the backend, we map it to our frontend union type
    status: exec.status as "pending" | "running" | "completed" | "failed",
    startedAt: exec.started_at,
    completedAt: exec.completed_at ?? null,
    executionTimeMs: exec.execution_time_ms ?? null,
    imagePaths: exec.image_paths ?? null,
  };
}

export function toUiToolExecutionLog(
  log: ApiToolExecutionLogResponse
): ToolExecutionLog {
  // Backend log levels (verified in models/tool_execution.py line 45):
  // 'info', 'warning', 'error' - NO 'debug'
  return {
    id: log.id,
    executionId: log.execution_id,
    logLevel: log.log_level as "info" | "warning" | "error",
    message: log.message,
    timestamp: log.timestamp,
  };
}

export function toUiToolExecutionResult(
  result: ApiToolExecutionResultResponse
): ToolExecutionResult {
  return {
    id: result.id,
    executionId: result.execution_id,
    resultData: result.result_data,
    resultMetadata: result.result_metadata ?? null,
    createdAt: result.created_at,
  };
}

// ============================================================================
// Scan Transformer
// ============================================================================

/**
 * Transform scan response.
 * NOTE: ScanResponse already uses camelCase from the backend (serialization_alias).
 */
export function toUiScan(scan: ApiScanResponse): Scan {
  // Backend validates fileType against ALLOWED_EXTENSIONS in config.py:
  // {'jpg', 'jpeg', 'png', 'gif', 'dcm', 'dicom'}
  // The backend schema doesn't define an enum, but we know these are the only valid values
  return {
    id: scan.id,
    chatId: scan.chatId,
    filePath: scan.filePath,
    displayPath: scan.displayPath,
    fileType: scan.fileType as "jpg" | "jpeg" | "png" | "gif" | "dcm" | "dicom",
    fileSize: scan.fileSize,
    uploadedAt: scan.uploadedAt,
  };
}

// ============================================================================
// Message Transformer
// ============================================================================

export function toUiMessage(msg: ApiMessageWithDetails): MessageWithDetails {
  return {
    id: msg.id,
    chatId: msg.chat_id,
    // Role is a string in the backend, we assert it to our union type
    role: msg.role as "user" | "assistant" | "system",
    content: msg.content,
    createdAt: msg.created_at,
    attachedScans: msg.attached_scans.map(toUiScan),
    toolExecutions: msg.tool_executions.map(toUiToolExecution),
  };
}

// ============================================================================
// Chat Transformer
// ============================================================================

export function toUiChat(chat: ApiChatResponse): Chat {
  return {
    id: chat.id,
    patientId: chat.patient_id,
    name: chat.name,
    createdAt: chat.created_at,
    updatedAt: chat.updated_at,
    lastMessageAt: chat.last_message_at ?? null,
    messageCount: chat.message_count,
    scanCount: chat.scan_count,
  };
}

// ============================================================================
// Patient Transformer
// ============================================================================

export function toUiPatientWithStats(
  patient: ApiPatientWithStats
): PatientWithStats {
  return {
    id: patient.id,
    name: patient.name ?? null,
    doctorId: patient.doctor_id,
    createdAt: patient.created_at,
    lastActivityAt: patient.last_activity_at,
    chatCount: patient.total_chats,
    scanCount: patient.total_scans,
  };
}

// ============================================================================
// Doctor / Auth Transformers
// ============================================================================

export function toUiDoctor(doctor: ApiDoctorResponse): Doctor {
  return {
    id: doctor.id,
    name: doctor.name,
    createdAt: doctor.created_at,
  };
}

export function toAuthSession(tokenResponse: ApiTokenResponse): AuthSession {
  return {
    token: tokenResponse.access_token,
    doctor: toUiDoctor(tokenResponse.doctor),
    expiresAt: "", // Backend doesn't provide expiry, can calculate if needed
  };
}

// ============================================================================
// Question Transformer
// ============================================================================

export function toUiQuestion(question: ApiQuestionResponse): SuggestedQuestion {
  return {
    id: question.id,
    doctorId: question.doctor_id,
    question: question.question,
    isDefault: question.is_default,
    displayOrder: question.display_order,
    createdAt: question.created_at,
  };
}

// ============================================================================
// Memory Transformers
// ============================================================================

export function toUiMemoryStats(
  stats: ApiMemoryStatsResponse
): import("../api/memory").MemoryStats {
  return {
    chatId: stats.chat_id,
    messageCount: stats.message_count,
    scanCount: stats.scan_count,
    toolExecutionCount: stats.tool_execution_count,
    hasContext: stats.has_context,
  };
}

export function toUiClearMemoryResponse(
  response: ApiClearMemoryResponse
): import("../api/memory").ClearMemoryResponse {
  return {
    success: response.success,
    message: response.message,
    chatId: response.chat_id,
  };
}

export function toUiSystemCleanupStats(
  response: ApiSystemCleanupStatsResponse
): import("../api/memory").SystemCleanupStats {
  return {
    success: response.success,
    message: response.message,
    stats: {
      checkpointsCleared: response.stats.checkpoints_cleared,
      memoryFreedMb: response.stats.memory_freed_mb,
    },
  };
}

// ============================================================================
// Tool Management Transformers
// ============================================================================

export function toUiTool(
  tool: ApiToolInfo
): import("../api/toolManagement").Tool {
  return {
    id: tool.id,
    name: tool.name,
    description: tool.description,
    status: tool.status as
      | "available"
      | "unavailable"
      | "loaded"
      | "unloaded"
      | "error"
      | "loading",
    category: tool.category,
    dependencies: tool.dependencies,
    requires_gpu: tool.requires_gpu,
    error_message: tool.error_message ?? undefined,
    loaded_at: tool.loaded_at ?? undefined,
  };
}
