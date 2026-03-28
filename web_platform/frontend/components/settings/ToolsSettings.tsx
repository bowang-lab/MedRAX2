/**
 * ToolsSettings Component
 *
 * Manage medical imaging tools:
 * - View available tools grouped by category
 * - Load/unload tools dynamically with real-time SSE progress
 * - View tool status, dependencies, and info
 * - Bulk load tools with SSE for each tool
 */

"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Wrench, Loader2, Info, Download, Check, X, AlertCircle } from "lucide-react";
import { getTools, loadTool, unloadTool, bulkLoadTools, Tool } from "../../lib/api/toolManagement";
import { ToolLoadingProgress } from "../tools/ToolLoadingProgress";
import { Card } from "../ui/Card";
import { Button } from "../ui/Button";
import { Spinner } from "../ui/Spinner";
import { Badge } from "../ui/Badge";
import { API_CONFIG } from "../../lib/config/api";
import { AUTH_CONFIG } from "../../lib/config/app";

interface ToolsByCategory {
  [category: string]: Tool[];
}

interface ToolLoadingState {
  progress: number;
  message: string;
}

const CATEGORY_DISPLAY_NAMES: { [key: string]: string } = {
  classification: "Classification",
  vqa: "Visual Question Answering",
  segmentation: "Segmentation",
  generation: "Generation",
  grounding: "Grounding",
  processing: "Image Processing",
  retrieval: "Retrieval & Search",
  execution: "Code Execution",
};

const CATEGORY_DESCRIPTIONS: { [key: string]: string } = {
  classification: "Identify pathologies and conditions in medical images",
  vqa: "Answer questions about medical images using AI",
  segmentation: "Segment and identify regions in medical images",
  generation: "Generate reports and synthetic medical images",
  grounding: "Locate specific findings in medical images",
  processing: "Process and convert medical image formats",
  retrieval: "Search medical knowledge and web resources",
  execution: "Execute Python code for custom analysis",
};

export function ToolsSettings() {
  const [tools, setTools] = useState<Tool[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set());
  const [selectedTool, setSelectedTool] = useState<Tool | null>(null);
  const [selectedToolIds, setSelectedToolIds] = useState<Set<string>>(new Set());

  // Track loading tools with SSE connections
  const [loadingTools, setLoadingTools] = useState<Map<string, ToolLoadingState>>(new Map());
  const sseConnectionsRef = useRef<Map<string, EventSource>>(new Map());

  const loadTools = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const fetchedTools = await getTools();
      setTools(fetchedTools);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load tools");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTools();
    // Expand all categories by default
    setExpandedCategories(new Set(Object.keys(CATEGORY_DISPLAY_NAMES)));

    // Capture the current ref value for cleanup
    const sseConnections = sseConnectionsRef.current;

    // Cleanup SSE connections on unmount
    return () => {
      sseConnections.forEach((connection) => connection.close());
      sseConnections.clear();
    };
  }, [loadTools]);

  // Create SSE connection for a specific tool
  const createSSEConnection = useCallback((toolId: string) => {
    // Don't create duplicate connections
    if (sseConnectionsRef.current.has(toolId)) {
      return;
    }

    const token = localStorage.getItem(AUTH_CONFIG.tokenKey);
    if (!token) {
      console.error("No auth token available");
      return;
    }

    const url = `${API_CONFIG.baseURL}/api/tools/${toolId}/load-stream?token=${encodeURIComponent(token)}`;
    const eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.status === "loading") {
          setLoadingTools((prev) =>
            new Map(prev).set(toolId, {
              progress: data.progress,
              message: data.message,
            })
          );
        } else if (data.status === "loaded") {
          // Tool loaded successfully
          console.log(`✅ Tool ${toolId} loaded successfully`);
          setLoadingTools((prev) => {
            const next = new Map(prev);
            next.delete(toolId);
            return next;
          });
          eventSource.close();
          sseConnectionsRef.current.delete(toolId);
          loadTools(); // Refresh to get final state
        } else if (data.status === "error") {
          console.error(`❌ Tool ${toolId} loading error:`, data.message);
          setError(`${toolId}: ${data.message}`);
          setLoadingTools((prev) => {
            const next = new Map(prev);
            next.delete(toolId);
            return next;
          });
          eventSource.close();
          sseConnectionsRef.current.delete(toolId);
          loadTools(); // Refresh to get error state
        }
      } catch (err) {
        console.error("Failed to parse SSE message:", err);
      }
    };

    eventSource.onerror = (err) => {
      console.error(`SSE connection error for ${toolId}:`, err);
      setLoadingTools((prev) => {
        const next = new Map(prev);
        next.delete(toolId);
        return next;
      });
      eventSource.close();
      sseConnectionsRef.current.delete(toolId);
      loadTools();
    };

    sseConnectionsRef.current.set(toolId, eventSource);
  }, [loadTools]);

  const handleLoadTool = async (toolId: string) => {
    try {
      setError(null);

      // Update UI immediately
      setTools((prev) => prev.map((t) => (t.id === toolId ? { ...t, status: "loading" as const } : t)));

      // Start SSE connection BEFORE calling load
      setLoadingTools((prev) =>
        new Map(prev).set(toolId, {
          progress: 0,
          message: "Initiating load...",
        })
      );

      // Initiate the actual load request (backend starts background loading)
      await loadTool(toolId);

      // Create SSE connection to track progress
      createSSEConnection(toolId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load tool");
      setLoadingTools((prev) => {
        const next = new Map(prev);
        next.delete(toolId);
        return next;
      });
      loadTools();
    }
  };

  const handleUnloadTool = async (toolId: string) => {
    // Update UI immediately
    setTools((prev) => prev.map((t) => (t.id === toolId ? { ...t, status: "loading" as const } : t)));

    try {
      await unloadTool(toolId);
      await loadTools(); // Refresh all tools
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to unload tool");
      await loadTools();
    }
  };

  const handleBulkLoadAll = async () => {
    try {
      setError(null);

      // Call bulk load endpoint
      const result = await bulkLoadTools({ loadAll: true });

      // For each tool that started loading, create an SSE connection
      const loadingToolIds = result.results
        .filter((r) => r.success && r.status === "loading")
        .map((r) => r.id);

      console.log(`📡 Starting SSE for ${loadingToolIds.length} tools:`, loadingToolIds);

      // Update UI to show loading state
      setTools((prev) =>
        prev.map((t) => (loadingToolIds.includes(t.id) ? { ...t, status: "loading" as const } : t))
      );

      // Create SSE connection for each loading tool
      loadingToolIds.forEach((toolId) => {
        setLoadingTools((prev) =>
          new Map(prev).set(toolId, {
            progress: 0,
            message: "Starting...",
          })
        );
        createSSEConnection(toolId);
      });

      // Immediate refresh to get accurate states
      await loadTools();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to bulk load tools");
    }
  };

  const handleBulkLoadSelected = async (ids: string[]) => {
    try {
      setError(null);

      // Call bulk load endpoint
      const result = await bulkLoadTools({ toolIds: ids });

      // For each tool that started loading, create an SSE connection
      const loadingToolIds = result.results
        .filter((r) => r.success && r.status === "loading")
        .map((r) => r.id);

      console.log(`📡 Starting SSE for ${loadingToolIds.length} tools:`, loadingToolIds);

      // Update UI to show loading state
      setTools((prev) =>
        prev.map((t) => (loadingToolIds.includes(t.id) ? { ...t, status: "loading" as const } : t))
      );

      // Create SSE connection for each loading tool
      loadingToolIds.forEach((toolId) => {
        setLoadingTools((prev) =>
          new Map(prev).set(toolId, {
            progress: 0,
            message: "Starting...",
          })
        );
        createSSEConnection(toolId);
      });

      // Immediate refresh to get accurate states
      await loadTools();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to bulk load tools");
    }
  };

  const groupedTools: ToolsByCategory = tools.reduce((acc, tool) => {
    const category = tool.category || "other";
    if (!acc[category]) acc[category] = [];
    acc[category].push(tool);
    return acc;
  }, {} as ToolsByCategory);

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  const toggleSelectTool = (toolId: string) => {
    setSelectedToolIds((prev) => {
      const next = new Set(prev);
      if (next.has(toolId)) {
        next.delete(toolId);
      } else {
        next.add(toolId);
      }
      return next;
    });
  };

  const handleSelectAll = () => {
    const availableToolIds = tools
      .filter((t) => t.status !== "loaded" && t.status !== "loading" && t.status !== "unavailable")
      .map((t) => t.id);
    setSelectedToolIds(new Set(availableToolIds));
  };

  const handleClearSelection = () => {
    setSelectedToolIds(new Set());
  };

  const hasActiveSSE = loadingTools.size > 0;

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case "loaded":
        return "success";
      case "loading":
        return "info";
      case "error":
        return "error";
      case "unavailable":
        return "warning";
      default:
        return "default";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "loaded":
        return <Check className="h-3 w-3" />;
      case "loading":
        return <Loader2 className="h-3 w-3 animate-spin" />;
      case "error":
        return <AlertCircle className="h-3 w-3" />;
      case "unavailable":
        return <X className="h-3 w-3" />;
      default:
        return <Download className="h-3 w-3" />;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">Tools Management</h2>
        <p className="text-sm text-zinc-400">
          Load and manage the AI tools available for analysis
        </p>
      </div>

      {/* Controls */}
      <Card className="p-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button variant="primary" size="sm" onClick={handleBulkLoadAll} disabled={hasActiveSSE}>
            Load All
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => handleBulkLoadSelected(Array.from(selectedToolIds))}
            disabled={selectedToolIds.size === 0 || hasActiveSSE}
          >
            Load Selected ({selectedToolIds.size})
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={handleSelectAll}>
            Select All
          </Button>
          <Button variant="ghost" size="sm" onClick={handleClearSelection}>
            Clear
          </Button>
        </div>
      </Card>

      {error && (
        <div className="p-3 bg-red-900/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Categories */}
      {Object.keys(groupedTools).length === 0 ? (
        <div className="text-center py-12 text-zinc-500">No tools available</div>
      ) : (
        <div className="space-y-4">
          {Object.entries(groupedTools).map(([category, categoryTools]) => (
            <Card key={category} className="p-0 overflow-hidden">
              <button
                onClick={() => toggleCategory(category)}
                className="w-full flex items-center justify-between p-4 bg-zinc-900/50 hover:bg-zinc-900 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Wrench className="h-5 w-5 text-zinc-400" />
                  <div className="text-left">
                    <h3 className="text-sm font-semibold text-white">
                      {CATEGORY_DISPLAY_NAMES[category] || category}
                    </h3>
                    <p className="text-xs text-zinc-500">
                      {CATEGORY_DESCRIPTIONS[category] || "Tools in this category"}
                    </p>
                  </div>
                </div>
                <span className="text-zinc-400 text-xs">{categoryTools.length} tools</span>
              </button>

              {expandedCategories.has(category) && (
                <div className="p-4 space-y-4">
                  {categoryTools.map((tool) => {
                    const toolLoadingState = loadingTools.get(tool.id);
                    const hasSSE = toolLoadingState !== undefined;

                    return (
                      <div
                        key={tool.id}
                        className="p-4 rounded-lg bg-zinc-900/50 border border-zinc-800 hover:border-zinc-700 transition-colors"
                      >
                        <div className="flex items-start justify-between gap-4">
                          {/* Tool Info */}
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                              <input
                                type="checkbox"
                                className="h-4 w-4 accent-blue-500"
                                checked={selectedToolIds.has(tool.id)}
                                onChange={() => toggleSelectTool(tool.id)}
                                disabled={
                                  tool.status === "loading" ||
                                  tool.status === "loaded" ||
                                  tool.status === "unavailable"
                                }
                                aria-label={`Select ${tool.name}`}
                              />
                              <h4 className="font-semibold text-white">{tool.name}</h4>
                              <Badge variant={getStatusBadgeVariant(tool.status)} size="sm">
                                <span className="flex items-center gap-1">
                                  {getStatusIcon(tool.status)}
                                  {tool.status}
                                </span>
                              </Badge>
                              {tool.requires_gpu && (
                                <Badge variant="default" size="sm">
                                  GPU Required
                                </Badge>
                              )}
                              {hasSSE && <Badge variant="info" size="sm">📡 SSE</Badge>}
                            </div>

                            <p className="text-sm text-zinc-400 mb-2">{tool.description}</p>

                            {/* Real-time SSE progress bar */}
                            {hasSSE && toolLoadingState && (
                              <div className="mb-3">
                                <ToolLoadingProgress
                                  progress={toolLoadingState.progress}
                                  message={toolLoadingState.message}
                                />
                                <p className="text-xs text-zinc-500 mt-1">
                                  ⓘ Real-time progress via SSE (estimated completion)
                                </p>
                              </div>
                            )}

                            {tool.loaded_at && (
                              <p className="text-xs text-zinc-500">
                                Loaded at: {new Date(tool.loaded_at).toLocaleString()}
                              </p>
                            )}

                            {tool.error_message && <p className="text-xs text-amber-400 mt-1">{tool.error_message}</p>}

                            {tool.dependencies && tool.dependencies.length > 0 && (
                              <div className="mt-2">
                                <button
                                  onClick={() => setSelectedTool(selectedTool?.id === tool.id ? null : tool)}
                                  className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
                                >
                                  <Info className="h-3 w-3" />
                                  Show dependencies ({tool.dependencies.length})
                                </button>
                                {selectedTool?.id === tool.id && (
                                  <div className="mt-2 p-2 bg-black/30 rounded text-xs text-zinc-400 font-mono">
                                    {tool.dependencies.join(", ")}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>

                          {/* Action Buttons */}
                          <div className="flex items-center gap-2">
                            {tool.status === "loading" ? (
                              <Button variant="secondary" size="sm" disabled>
                                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                                Loading...
                              </Button>
                            ) : tool.status === "loaded" ? (
                              <Button variant="secondary" size="sm" onClick={() => handleUnloadTool(tool.id)}>
                                Unload
                              </Button>
                            ) : tool.status === "unavailable" ? (
                              <Button variant="secondary" size="sm" disabled title="Install dependencies first">
                                Unavailable
                              </Button>
                            ) : (
                              <Button
                                variant="primary"
                                size="sm"
                                onClick={() => handleLoadTool(tool.id)}
                                disabled={hasActiveSSE}
                                title={hasActiveSSE ? "Another tool is loading" : undefined}
                              >
                                Load
                              </Button>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
