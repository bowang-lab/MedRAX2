/**
 * Tool Execution Types
 *
 * Tools are AI models that process medical images.
 * Each message can trigger multiple tool executions.
 * Full execution history is stored with logs and results.
 */

export type ToolStatus = "pending" | "running" | "completed" | "failed";
export type ToolLogLevel = "info" | "warning" | "error";

export interface ToolExecution {
  id: string;
  messageId: string;
  requestId: string | null;
  toolName: string;
  toolDisplayName: string;
  status: ToolStatus;
  startedAt: string;
  completedAt: string | null;
  executionTimeMs: number | null;
  imagePaths: string[] | null;
}

export interface ToolExecutionWithDetails extends ToolExecution {
  logs: ToolExecutionLog[];
  result: ToolExecutionResult | null;
}

export interface ToolExecutionLog {
  id: string; // Fixed: Backend returns UUID string, not number
  executionId: string;
  logLevel: ToolLogLevel;
  message: string;
  timestamp: string; // Fixed: Backend uses 'timestamp', not 'createdAt'
}

export interface ToolExecutionResult {
  id: string; // Fixed: Backend returns UUID string, not number
  executionId: string;
  // Tool results have dynamic structures that vary by tool type (classification, segmentation, VQA, etc.)
  // Each tool returns different JSON structures (Record<string, unknown> is more type-safe than 'any')
  resultData: Record<string, unknown>;
  // Optional metadata can contain various tool-specific information
  resultMetadata: Record<string, unknown> | null; // Fixed: Backend uses 'result_metadata' which becomes 'resultMetadata'
  createdAt: string; // Added: Backend returns this field
}

// Tool Management
export interface Tool {
  id: string;
  name: string;
  displayName: string;
  description: string;
  category: string;
  status: "available" | "loaded" | "unavailable" | "error";
  requiresModel: boolean;
  modelSizeGb: number;
  macCompatible: boolean;
  isLoaded: boolean;
  isCached: boolean;
  errorMessage: string | null;
}

export interface ToolLoadRequest {
  toolId: string;
}

export interface ToolLoadResponse {
  success: boolean;
  toolId: string;
  message: string;
  toolInfo: Tool;
}
