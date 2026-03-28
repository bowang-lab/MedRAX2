/**
 * ToolExecutionTimeline Component
 *
 * Timeline view of tool execution logs showing:
 * - Log level (info, warning, error)
 * - Timestamp
 * - Message
 */

"use client";

import { Info, AlertTriangle, XCircle } from "lucide-react";
import type { ToolExecutionLog } from "../../lib/types/tool";
import { formatDateTime, classNames } from "../../lib/utils";

/**
 * ToolExecutionTimeline Component Props
 * @property logs - Array of execution logs to display in timeline (required)
 */
interface ToolExecutionTimelineProps {
  /** Array of execution logs to display in timeline */
  logs: ToolExecutionLog[];
}

export function ToolExecutionTimeline({ logs }: ToolExecutionTimelineProps) {
  if (!logs || logs.length === 0) return null;

  const getLogIcon = (level: string) => {
    switch (level) {
      case "error":
        return <XCircle className="h-4 w-4 text-red-400" />;
      case "warning":
        return <AlertTriangle className="h-4 w-4 text-yellow-400" />;
      case "info":
      default:
        return <Info className="h-4 w-4 text-blue-400" />;
    }
  };

  const getLogColor = (level: string) => {
    switch (level) {
      case "error":
        return "text-red-400";
      case "warning":
        return "text-yellow-400";
      case "info":
      default:
        return "text-zinc-300";
    }
  };

  return (
    <div className="p-4 bg-zinc-800 rounded-lg">
      <h4 className="text-sm font-semibold text-white mb-3">Execution Logs</h4>

      <div className="space-y-3">
        {(logs || []).map((log) => (
          <div key={log.id} className="flex items-start space-x-3">
            <div className="flex-shrink-0 mt-0.5">{getLogIcon(log.logLevel)}</div>
            <div className="flex-1 min-w-0">
              <div className="flex items-baseline justify-between mb-1">
                <span className={classNames("text-xs font-medium uppercase", getLogColor(log.logLevel))}>
                  {log.logLevel}
                </span>
                <span className="text-xs text-zinc-500">{formatDateTime(log.timestamp)}</span>
              </div>
              <p className="text-sm text-zinc-300 whitespace-pre-wrap">{log.message}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
