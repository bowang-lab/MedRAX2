/**
 * Tool Outputs Sidebar Component
 *
 * Right sidebar showing detailed tool execution information for a message.
 * Displays all tools used, their inputs, outputs, logs, and results.
 */

"use client";

import { useState, useEffect, useRef } from "react";
import { X, ChevronDown, ChevronRight, Clock, Image as ImageIcon, FileText } from "lucide-react";
import { getToolExecutionsByMessage } from "@/lib/api/toolHistory";
import { getToolExecutionDetail } from "@/lib/api/tools";
import { Spinner } from "../ui/Spinner";
import { Badge } from "../ui/Badge";
import type { ToolExecution, ToolExecutionLog, ToolExecutionResult } from "@/lib/types/tool";
import { getImageUrl } from "@/lib/utils/image";
import { ToolResultCard } from "./ToolResultCard";

interface ToolOutputsSidebarProps {
  messageId: string | null;
  isOpen: boolean;
  onClose: () => void;
}

interface ToolExecutionDetail {
  execution: ToolExecution;
  logs: ToolExecutionLog[];
  result: ToolExecutionResult | null;
}

interface ExecutionDetailState {
  detail: ToolExecutionDetail | null;
  loading: boolean;
  error: string | null;
}

export function ToolOutputsSidebar({ messageId, isOpen, onClose }: ToolOutputsSidebarProps) {
  const [executions, setExecutions] = useState<ToolExecution[]>([]);
  const [expandedExecutions, setExpandedExecutions] = useState<Set<string>>(new Set());
  const [executionDetails, setExecutionDetails] = useState<Map<string, ExecutionDetailState>>(new Map());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isMountedRef = useRef(true);

  // Track mounted state for cleanup
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // Handle Escape key to close sidebar and prevent body scroll
  useEffect(() => {
    if (!isOpen) return;

    // Prevent body scroll when sidebar is open
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    document.addEventListener("keydown", handleEscape);

    return () => {
      document.body.style.overflow = originalOverflow;
      document.removeEventListener("keydown", handleEscape);
    };
  }, [isOpen, onClose]);

  // Load tool executions when sidebar opens
  useEffect(() => {
    let cancelled = false;

    const loadData = async () => {
      if (!messageId) return;

      setLoading(true);
      setError(null);
      try {
        const history = await getToolExecutionsByMessage(messageId);
        if (cancelled) return; // Don't update state if component unmounted

        setExecutions(history);
        // Auto-expand all executions and load their details
        const allIds = new Set(history.map((ex) => ex.id));
        setExpandedExecutions(allIds);

        // Initialize loading state for all executions
        const initialStates = new Map<string, ExecutionDetailState>();
        history.forEach((ex) => {
          initialStates.set(ex.id, { detail: null, loading: true, error: null });
        });
        setExecutionDetails(initialStates);

        // Load details for all executions
        const detailPromises = history.map(async (ex) => {
          try {
            const detail = await getToolExecutionDetail(ex.id);
            if (!cancelled) {
              setExecutionDetails(
                (prev) =>
                  new Map(prev).set(ex.id, {
                    detail,
                    loading: false,
                    error: null,
                  })
              );
            }
          } catch (err) {
            console.error(`Failed to load details for execution ${ex.id}:`, err);
            if (!cancelled) {
              setExecutionDetails(
                (prev) =>
                  new Map(prev).set(ex.id, {
                    detail: null,
                    loading: false,
                    error: err instanceof Error ? err.message : "Failed to load details",
                  })
              );
            }
          }
        });

        await Promise.all(detailPromises);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load tool history");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    if (isOpen && messageId) {
      loadData();
    } else {
      // Reset state when closing
      setExecutions([]);
      setExpandedExecutions(new Set());
      setExecutionDetails(new Map());
    }

    return () => {
      cancelled = true;
    };
  }, [isOpen, messageId]);

  const toggleExecution = (executionId: string) => {
    setExpandedExecutions((prev) => {
      const next = new Set(prev);
      if (next.has(executionId)) {
        next.delete(executionId);
      } else {
        next.add(executionId);
        // Load details if not already loaded or if loading failed
        const currentState = executionDetails.get(executionId);
        if (!currentState || (!currentState.detail && !currentState.loading && currentState.error)) {
          loadExecutionDetail(executionId);
        }
      }
      return next;
    });
  };

  const loadExecutionDetail = (executionId: string) => {
    // Don't load if component is unmounted or sidebar is closed
    if (!isMountedRef.current || !isOpen) return;

    // Set loading state
    setExecutionDetails(
      (prev) =>
        new Map(prev).set(executionId, {
          detail: null,
          loading: true,
          error: null,
        })
    );

    // Load details asynchronously
    getToolExecutionDetail(executionId)
      .then((detail) => {
        // Only update state if still mounted and sidebar is still open
        if (isMountedRef.current && isOpen) {
          setExecutionDetails(
            (prev) =>
              new Map(prev).set(executionId, {
                detail,
                loading: false,
                error: null,
              })
          );
        }
      })
      .catch((err) => {
        console.error(`Failed to load details for execution ${executionId}:`, err);
        // Only update state if still mounted and sidebar is still open
        if (isMountedRef.current && isOpen) {
          setExecutionDetails(
            (prev) =>
              new Map(prev).set(executionId, {
                detail: null,
                loading: false,
                error: err instanceof Error ? err.message : "Failed to load details",
              })
          );
        }
      });
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "completed":
        return "success";
      case "running":
        return "info";
      case "failed":
        return "error";
      default:
        return "default";
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  // Note: renderJsonValue function removed as we now use ToolResultCard for all formatted outputs

  if (!isOpen) return null;

  return (
    <>
      {/* Overlay */}
      <div className="fixed inset-0 bg-black/50 z-40" onClick={onClose} />

      {/* Sidebar */}
      <aside className="fixed right-0 top-0 bottom-0 w-[500px] bg-zinc-900 border-l border-zinc-800 flex flex-col z-50 shadow-2xl">
        {/* Header */}
        <div className="h-16 border-b border-zinc-800 flex items-center justify-between px-4 flex-shrink-0">
          <div>
            <h2 className="text-lg font-semibold text-white">Tool Outputs</h2>
            {executions.length > 0 && (
              <p className="text-xs text-zinc-500">
                {executions.length} tool{executions.length !== 1 ? "s" : ""} used
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-2 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-md transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Spinner size="md" />
            </div>
          ) : error ? (
            <div className="text-red-400 text-sm text-center py-12">{error}</div>
          ) : executions.length === 0 ? (
            <div className="text-zinc-500 text-sm text-center py-12">
              No tools were used for this message.
            </div>
          ) : (
            executions.map((execution) => {
              const isExpanded = expandedExecutions.has(execution.id);
              const detailState = executionDetails.get(execution.id);
              const detail = detailState?.detail;

              const displayName = execution.toolDisplayName || execution.toolName || "Tool";

              return (
                <div key={execution.id} className="bg-zinc-800 rounded-lg border border-zinc-700">
                  {/* Tool Header */}
                  <button
                    onClick={() => toggleExecution(execution.id)}
                    className="w-full p-3 flex items-center justify-between hover:bg-zinc-700/50 transition-colors rounded-lg"
                  >
                    <div className="flex items-center space-x-3">
                      {isExpanded ? (
                        <ChevronDown className="h-4 w-4 text-zinc-400" />
                      ) : (
                        <ChevronRight className="h-4 w-4 text-zinc-400" />
                      )}
                      <div className="text-left">
                        <p className="text-sm font-medium text-white">{displayName}</p>
                        <div className="flex items-center space-x-2 mt-1">
                          <Badge variant={getStatusColor(execution.status)} size="sm">
                            {execution.status}
                          </Badge>
                          {execution.executionTimeMs != null && (
                            <span className="text-xs text-zinc-400 flex items-center">
                              <Clock className="h-3 w-3 mr-1" />
                              {(execution.executionTimeMs / 1000).toFixed(2)}s
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </button>

                  {/* Expanded Content */}
                  {isExpanded && (
                    <div className="border-t border-zinc-700 p-4 space-y-4">
                      {/* Input Images (filter only original uploads, exclude generated temp/visualizations) */}
                      {execution.imagePaths &&
                        execution.imagePaths.filter(
                          (p) => typeof p === "string" && (p.startsWith("uploads/") || p.includes("/uploads/"))
                        ).length > 0 && (
                          <div>
                            <div className="flex items-center space-x-2 mb-2">
                              <ImageIcon className="h-4 w-4 text-zinc-400" />
                              <h4 className="text-xs font-medium text-zinc-400 uppercase">Inputs</h4>
                            </div>
                            <div className="space-y-2">
                              {execution.imagePaths
                                .filter(
                                  (path) =>
                                    typeof path === "string" &&
                                    (path.startsWith("uploads/") || path.includes("/uploads/"))
                                )
                                .map((path, idx) => {
                                  const imageUrl = getImageUrl(path);
                                  return (
                                    <div key={idx} className="text-xs">
                                      <p className="text-zinc-500 mb-1">Image {idx + 1}:</p>
                                      {!imageUrl ? (
                                        <div className="bg-red-900/20 border border-red-800 rounded p-2 text-red-400 text-xs">
                                          ⚠️ Invalid image path
                                        </div>
                                      ) : (
                                        /* eslint-disable-next-line @next/next/no-img-element */
                                        <img
                                          src={imageUrl}
                                          alt={`Input ${idx + 1}`}
                                          className="w-full h-auto rounded border border-zinc-700"
                                          onError={(e) => {
                                            const img = e.currentTarget;
                                            img.style.display = "none";
                                            const errorDiv = document.createElement("div");
                                            errorDiv.className =
                                              "bg-red-900/20 border border-red-800 rounded p-2 text-red-400 text-xs";
                                            errorDiv.innerHTML = `
                                                                                    <div>⚠️ Failed to load image</div>
                                                                                    <div class="text-[10px] mt-1">
                                                                                        <div>Path: ${path}</div>
                                                                                        <div>URL: ${imageUrl}</div>
                                                                                    </div>
                                                                                `;
                                            img.parentElement?.insertBefore(errorDiv, img);
                                          }}
                                        />
                                      )}
                                      <p className="text-zinc-600 font-mono text-[10px] mt-1 truncate">
                                        {path}
                                      </p>
                                    </div>
                                  );
                                })}
                            </div>
                          </div>
                        )}

                      {/* Tool Result - Beautiful Formatted Output */}
                      {detail?.result && (
                        <div>
                          <h4 className="text-xs font-medium text-zinc-400 uppercase mb-3">Output</h4>
                          <ToolResultCard toolName={execution.toolName} result={detail.result} />
                        </div>
                      )}

                      {/* Execution Logs */}
                      {detail?.logs && detail.logs.length > 0 && (
                        <div>
                          <div className="flex items-center space-x-2 mb-2">
                            <FileText className="h-4 w-4 text-zinc-400" />
                            <h4 className="text-xs font-medium text-zinc-400 uppercase">Logs</h4>
                          </div>
                          <div className="space-y-1 bg-zinc-900 rounded p-3">
                            {detail.logs.map((log) => (
                              <div key={log.id} className="text-xs font-mono">
                                <span
                                  className={
                                    log.logLevel === "error"
                                      ? "text-red-400"
                                      : log.logLevel === "warning"
                                        ? "text-yellow-400"
                                        : "text-zinc-400"
                                  }
                                >
                                  [{log.logLevel.toUpperCase()}]
                                </span>{" "}
                                <span className="text-zinc-500">{formatDate(log.timestamp)}</span>{" "}
                                <span className="text-white">{log.message}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Show loading/error state */}
                      {!detail && detailState?.loading && (
                        <div className="flex items-center justify-center py-4">
                          <Spinner size="sm" />
                          <span className="ml-2 text-xs text-zinc-500">Loading details...</span>
                        </div>
                      )}

                      {!detail && detailState?.error && (
                        <div className="bg-red-900/20 border border-red-800 rounded p-3 text-sm">
                          <p className="text-red-400 mb-2">Failed to load tool details</p>
                          <p className="text-red-300 text-xs">{detailState.error}</p>
                          <button
                            onClick={() => loadExecutionDetail(execution.id)}
                            className="mt-2 text-xs text-red-400 hover:text-red-300 underline"
                          >
                            Retry
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </aside>
    </>
  );
}
