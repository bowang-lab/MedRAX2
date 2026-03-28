/**
 * MessageActivity Component
 *
 * Shows tool executions inline with the message:
 * - Compact list of tools used
 * - Click to see detailed tool history
 */

"use client";

import { Wrench, CheckCircle, XCircle, Loader2 } from "lucide-react";
import type { ToolExecution } from "../../lib/types/tool";
import { Badge } from "../ui/Badge";

/**
 * MessageActivity Component Props
 * @property executions - Array of tool executions for this message (required)
 * @property onShowDetails - Optional callback when user wants to see all tool execution details
 */
interface MessageActivityProps {
  /** Array of tool executions for this message */
  executions: ToolExecution[];
  /** Optional callback when user wants to see all tool execution details */
  onShowDetails?: () => void;
}

export function MessageActivity({ executions, onShowDetails }: MessageActivityProps) {
  if (!executions || executions.length === 0) return null;

  // Format duration helper: prefer ms for sub-second, otherwise seconds with 2 decimals
  const formatDuration = (durationMs: number): string => {
    if (durationMs < 1000) {
      return `${Math.max(0, Math.round(durationMs))}ms`;
    }
    return `${(durationMs / 1000).toFixed(2)}s`;
  };

  // Compute duration with fallback: use provided executionTimeMs, else derive from timestamps
  const getDurationMs = (e: ToolExecution): number | null => {
    if (e.executionTimeMs != null) return e.executionTimeMs;
    if (e.startedAt && e.completedAt) {
      const start = new Date(e.startedAt).getTime();
      const end = new Date(e.completedAt).getTime();
      if (!Number.isNaN(start) && !Number.isNaN(end) && end >= start) {
        return end - start;
      }
    }
    return null;
  };

  // Calculate summary stats
  const completed = executions.filter((e) => e.status === "completed").length;
  const failed = executions.filter((e) => e.status === "failed").length;
  const running = executions.filter((e) => e.status === "running").length;

  return (
    <div className="mt-2 p-3 bg-zinc-900/50 rounded-lg border border-zinc-800">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <Wrench className="h-4 w-4 text-zinc-400" />
          <span className="text-xs font-medium text-zinc-400">Tool Activity</span>
        </div>
        <div className="flex items-center space-x-2 text-xs">
          {completed > 0 && <span className="text-emerald-400">✓ {completed}</span>}
          {failed > 0 && <span className="text-red-400">✗ {failed}</span>}
          {running > 0 && <span className="text-blue-400">⟳ {running}</span>}
        </div>
      </div>

      <div className="space-y-2">
        {executions.map((execution, index) => {
          // Determine icon based on status
          const getStatusIcon = () => {
            switch (execution.status) {
              case "completed":
                return <CheckCircle className="h-4 w-4 text-emerald-400 flex-shrink-0" />;
              case "failed":
                return <XCircle className="h-4 w-4 text-red-400 flex-shrink-0" />;
              case "running":
                return <Loader2 className="h-4 w-4 text-blue-400 flex-shrink-0 animate-spin" />;
              case "pending":
                return <Loader2 className="h-4 w-4 text-zinc-500 flex-shrink-0" />;
              default:
                return <Wrench className="h-4 w-4 text-zinc-500 flex-shrink-0" />;
            }
          };

          // Determine badge variant
          const getBadgeVariant = () => {
            switch (execution.status) {
              case "completed":
                return "success" as const;
              case "failed":
                return "error" as const;
              case "running":
                return "info" as const;
              default:
                return "default" as const;
            }
          };

          return (
            <div key={execution.id} className="w-full flex items-center justify-between p-2 rounded">
              <div className="flex items-center space-x-2 flex-1 min-w-0">
                {/* Execution order number */}
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-zinc-700 text-zinc-300 text-[10px] font-semibold flex items-center justify-center">
                  {index + 1}
                </span>

                {getStatusIcon()}

                <span className="text-sm text-zinc-300 truncate">
                  {execution.toolDisplayName || execution.toolName || "Tool"}
                </span>
              </div>

              <div className="flex items-center space-x-2">
                {(() => {
                  const durationMs = getDurationMs(execution);
                  if (durationMs != null) {
                    return <span className="text-xs text-zinc-500">{formatDuration(durationMs)}</span>;
                  }
                  return null;
                })()}
                <Badge variant={getBadgeVariant()}>{execution.status}</Badge>
              </div>
            </div>
          );
        })}
      </div>

      {onShowDetails && (
        <button
          onClick={onShowDetails}
          className="mt-2 text-xs text-blue-400 hover:text-blue-300 transition-colors"
        >
          📊 View detailed tool outputs ({executions.length} tool{executions.length !== 1 ? "s" : ""})
        </button>
      )}
    </div>
  );
}
