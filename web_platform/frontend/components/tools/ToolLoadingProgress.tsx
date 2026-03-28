/**
 * Tool Loading Progress Component
 *
 * Shows real-time progress during tool loading via SSE.
 */

import React from "react";

export interface ToolLoadingProgressProps {
  progress: number; // 0-100
  message: string;
  isError?: boolean;
}

export function ToolLoadingProgress({ progress, message, isError = false }: ToolLoadingProgressProps) {
  return (
    <div className="space-y-2">
      {/* Progress bar */}
      <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
        <div
          className={`h-full transition-all duration-300 ${isError ? "bg-red-500" : "bg-blue-500"}`}
          style={{ width: `${Math.min(Math.max(progress, 0), 100)}%` }}
        />
      </div>

      {/* Message and percentage */}
      <div className="flex items-center justify-between text-sm">
        <span className={isError ? "text-red-600" : "text-gray-600"}>{message}</span>
        <span className={`font-medium ${isError ? "text-red-600" : "text-blue-600"}`}>
          {Math.round(progress)}%
        </span>
      </div>
    </div>
  );
}
