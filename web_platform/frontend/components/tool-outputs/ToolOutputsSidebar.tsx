/**
 * Tool Outputs Sidebar Component
 * 
 * Right sidebar showing detailed tool execution information for a message.
 * Displays all tools used, their inputs, outputs, logs, and results.
 */

'use client';

import { useState, useEffect, useRef } from 'react';
import { X, ChevronDown, ChevronRight, Clock, Image as ImageIcon, Code, FileText } from 'lucide-react';
import { getMessageToolHistory } from '@/lib/api/toolHistory';
import { getToolExecutionDetail } from '@/lib/api/tools';
import { Spinner } from '../ui/Spinner';
import { Badge } from '../ui/Badge';
import type { ToolExecution, ToolExecutionLog, ToolExecutionResult } from '@/lib/types/tool';
import { getImageUrl } from '@/lib/utils/image';

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
        if (isOpen) {
            // Prevent body scroll when sidebar is open
            const originalOverflow = document.body.style.overflow;
            document.body.style.overflow = 'hidden';

            const handleEscape = (e: KeyboardEvent) => {
                if (e.key === 'Escape') {
                    onClose();
                }
            };

            document.addEventListener('keydown', handleEscape);

            return () => {
                document.body.style.overflow = originalOverflow;
                document.removeEventListener('keydown', handleEscape);
            };
        }
    }, [isOpen, onClose]);

    // Load tool executions when sidebar opens
    useEffect(() => {
        let cancelled = false;

        const loadData = async () => {
            if (!messageId) return;

            setLoading(true);
            setError(null);
            try {
                const history = await getMessageToolHistory(messageId);
                if (cancelled) return; // Don't update state if component unmounted

                setExecutions(history);
                // Auto-expand all executions and load their details
                const allIds = new Set(history.map(ex => ex.id));
                setExpandedExecutions(allIds);

                // Initialize loading state for all executions
                const initialStates = new Map<string, ExecutionDetailState>();
                history.forEach(ex => {
                    initialStates.set(ex.id, { detail: null, loading: true, error: null });
                });
                setExecutionDetails(initialStates);

                // Load details for all executions
                const detailPromises = history.map(async (ex) => {
                    try {
                        const detail = await getToolExecutionDetail(ex.id);
                        if (!cancelled) {
                            setExecutionDetails(prev => new Map(prev).set(ex.id, {
                                detail,
                                loading: false,
                                error: null
                            }));
                        }
                    } catch (err) {
                        console.error(`Failed to load details for execution ${ex.id}:`, err);
                        if (!cancelled) {
                            setExecutionDetails(prev => new Map(prev).set(ex.id, {
                                detail: null,
                                loading: false,
                                error: err instanceof Error ? err.message : 'Failed to load details'
                            }));
                        }
                    }
                });

                await Promise.all(detailPromises);
            } catch (err) {
                if (!cancelled) {
                    setError(err instanceof Error ? err.message : 'Failed to load tool history');
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
        setExpandedExecutions(prev => {
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
        setExecutionDetails(prev => new Map(prev).set(executionId, {
            detail: null,
            loading: true,
            error: null
        }));

        // Load details asynchronously
        getToolExecutionDetail(executionId)
            .then(detail => {
                // Only update state if still mounted and sidebar is still open
                if (isMountedRef.current && isOpen) {
                    setExecutionDetails(prev => new Map(prev).set(executionId, {
                        detail,
                        loading: false,
                        error: null
                    }));
                }
            })
            .catch(err => {
                console.error(`Failed to load details for execution ${executionId}:`, err);
                // Only update state if still mounted and sidebar is still open
                if (isMountedRef.current && isOpen) {
                    setExecutionDetails(prev => new Map(prev).set(executionId, {
                        detail: null,
                        loading: false,
                        error: err instanceof Error ? err.message : 'Failed to load details'
                    }));
                }
            });
    };

    const getStatusColor = (status: string) => {
        switch (status.toLowerCase()) {
            case 'completed':
                return 'success';
            case 'running':
                return 'info';
            case 'failed':
                return 'error';
            default:
                return 'default';
        }
    };

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    };

    const renderJsonValue = (value: unknown, key?: string, depth: number = 0): React.ReactNode => {
        // Prevent infinite recursion on circular references or too deep nesting
        if (depth > 10) {
            return <span className="text-orange-400">[Too deeply nested]</span>;
        }
        if (value === null || value === undefined) {
            return <span className="text-zinc-600">null</span>;
        }

        if (typeof value === 'boolean') {
            return <span className="text-blue-400">{value.toString()}</span>;
        }

        if (typeof value === 'number') {
            // Handle special number values
            if (Number.isNaN(value)) {
                return <span className="text-orange-400">NaN</span>;
            }
            if (value === Infinity) {
                return <span className="text-orange-400">Infinity</span>;
            }
            if (value === -Infinity) {
                return <span className="text-orange-400">-Infinity</span>;
            }
            // Format probabilities nicely
            if (key && (key.toLowerCase().includes('prob') || key.toLowerCase().includes('score'))) {
                return <span className="text-green-400">{(value * 100).toFixed(2)}%</span>;
            }
            return <span className="text-green-400">{value}</span>;
        }

        if (typeof value === 'string') {
            // Handle empty strings
            if (value === '') {
                return <span className="text-zinc-600">"" (empty)</span>;
            }
            // Check if it's an image path
            if (value.includes('uploads/') || value.endsWith('.jpg') || value.endsWith('.png')) {
                return (
                    <div className="mt-2">
                        <img
                            src={getImageUrl(value)}
                            alt="Result"
                            className="max-w-full h-auto rounded border border-zinc-700"
                            onError={(e) => {
                                // Hide broken image and show path instead
                                e.currentTarget.style.display = 'none';
                                const parent = e.currentTarget.parentElement;
                                if (parent) {
                                    parent.innerHTML = `<span class="text-red-400 text-xs">⚠️ Image not found: ${value}</span>`;
                                }
                            }}
                        />
                    </div>
                );
            }
            return <span className="text-yellow-400">"{value}"</span>;
        }

        if (Array.isArray(value)) {
            if (value.length === 0) {
                return <span className="text-zinc-600">[]</span>;
            }
            // Limit array rendering to prevent performance issues
            const maxItems = 100;
            const displayItems = value.slice(0, maxItems);
            const hasMore = value.length > maxItems;

            return (
                <div className="ml-4 mt-1 space-y-1">
                    {displayItems.map((item, idx) => (
                        <div key={idx} className="text-sm">
                            <span className="text-zinc-600">[{idx}]:</span> {renderJsonValue(item, undefined, depth + 1)}
                        </div>
                    ))}
                    {hasMore && (
                        <div className="text-xs text-orange-400 italic">
                            ... and {value.length - maxItems} more items
                        </div>
                    )}
                </div>
            );
        }

        // typeof null === 'object' in JavaScript, so we need explicit null check
        if (typeof value === 'object' && value !== null) {
            const entries = Object.entries(value);
            if (entries.length === 0) {
                return <span className="text-zinc-600">{'{}'}</span>;
            }
            // Limit object properties to prevent performance issues
            const maxProps = 50;
            const displayEntries = entries.slice(0, maxProps);
            const hasMore = entries.length > maxProps;

            return (
                <div className="ml-4 mt-1 space-y-1">
                    {displayEntries.map(([k, v]) => (
                        <div key={k} className="text-sm">
                            <span className="text-cyan-400">{k}:</span> {renderJsonValue(v, k, depth + 1)}
                        </div>
                    ))}
                    {hasMore && (
                        <div className="text-xs text-orange-400 italic">
                            ... and {entries.length - maxProps} more properties
                        </div>
                    )}
                </div>
            );
        }

        // Handle unexpected types (function, symbol, bigint)
        if (typeof value === 'function') {
            return <span className="text-purple-400">[Function]</span>;
        }
        if (typeof value === 'symbol') {
            return <span className="text-purple-400">[Symbol]</span>;
        }
        if (typeof value === 'bigint') {
            return <span className="text-green-400">{value.toString()}n</span>;
        }

        // Fallback for any other type
        return <span className="text-zinc-400">{String(value)}</span>;
    };

    if (!isOpen) return null;

    return (
        <>
            {/* Overlay */}
            <div
                className="fixed inset-0 bg-black/50 z-40"
                onClick={onClose}
            />

            {/* Sidebar */}
            <aside className="fixed right-0 top-0 bottom-0 w-[500px] bg-zinc-900 border-l border-zinc-800 flex flex-col z-50 shadow-2xl">
                {/* Header */}
                <div className="h-16 border-b border-zinc-800 flex items-center justify-between px-4 flex-shrink-0">
                    <div>
                        <h2 className="text-lg font-semibold text-white">Tool Outputs</h2>
                        {executions.length > 0 && (
                            <p className="text-xs text-zinc-500">{executions.length} tool{executions.length !== 1 ? 's' : ''} used</p>
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
                        executions.map((execution, index) => {
                            const isExpanded = expandedExecutions.has(execution.id);
                            const detailState = executionDetails.get(execution.id);
                            const detail = detailState?.detail;

                            return (
                                <div
                                    key={execution.id}
                                    className="bg-zinc-800 rounded-lg border border-zinc-700 overflow-hidden"
                                >
                                    {/* Tool Header - Always Visible */}
                                    <button
                                        onClick={() => toggleExecution(execution.id)}
                                        className="w-full p-4 flex items-start justify-between hover:bg-zinc-750 transition-colors"
                                    >
                                        <div className="flex-1 text-left">
                                            <div className="flex items-center space-x-2 mb-2">
                                                <span className="text-zinc-400 font-mono text-xs">#{index + 1}</span>
                                                <h3 className="text-sm font-semibold text-white">
                                                    {execution.toolDisplayName}
                                                </h3>
                                                <Badge variant={getStatusColor(execution.status)}>
                                                    {execution.status}
                                                </Badge>
                                            </div>
                                            <div className="flex items-center space-x-4 text-xs text-zinc-500">
                                                <div className="flex items-center space-x-1">
                                                    <Clock className="h-3 w-3" />
                                                    <span>{formatDate(execution.startedAt)}</span>
                                                </div>
                                                {execution.executionTimeMs && (
                                                    <span>{(execution.executionTimeMs / 1000).toFixed(2)}s</span>
                                                )}
                                            </div>
                                        </div>
                                        <div className="text-zinc-400">
                                            {isExpanded ? (
                                                <ChevronDown className="h-5 w-5" />
                                            ) : (
                                                <ChevronRight className="h-5 w-5" />
                                            )}
                                        </div>
                                    </button>

                                    {/* Expanded Details */}
                                    {isExpanded && (
                                        <div className="border-t border-zinc-700 p-4 space-y-4">
                                            {/* Input Images */}
                                            {execution.imagePaths && execution.imagePaths.length > 0 && (
                                                <div>
                                                    <div className="flex items-center space-x-2 mb-2">
                                                        <ImageIcon className="h-4 w-4 text-zinc-400" />
                                                        <h4 className="text-xs font-medium text-zinc-400 uppercase">Inputs</h4>
                                                    </div>
                                                    <div className="space-y-2">
                                                        {execution.imagePaths.map((path, idx) => (
                                                            <div key={idx} className="text-xs">
                                                                <p className="text-zinc-500 mb-1">Image {idx + 1}:</p>
                                                                <img
                                                                    src={getImageUrl(path)}
                                                                    alt={`Input ${idx + 1}`}
                                                                    className="w-full h-auto rounded border border-zinc-700"
                                                                    onError={(e) => {
                                                                        e.currentTarget.style.display = 'none';
                                                                        const container = e.currentTarget.parentElement;
                                                                        if (container) {
                                                                            const errorMsg = document.createElement('div');
                                                                            errorMsg.className = 'bg-red-900/20 border border-red-800 rounded p-2 text-red-400 text-xs';
                                                                            errorMsg.textContent = '⚠️ Failed to load image';
                                                                            container.insertBefore(errorMsg, e.currentTarget);
                                                                        }
                                                                    }}
                                                                />
                                                                <p className="text-zinc-600 font-mono text-[10px] mt-1 truncate">
                                                                    {path}
                                                                </p>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Tool Result */}
                                            {detail?.result && (
                                                <div>
                                                    <div className="flex items-center space-x-2 mb-2">
                                                        <Code className="h-4 w-4 text-zinc-400" />
                                                        <h4 className="text-xs font-medium text-zinc-400 uppercase">Output</h4>
                                                    </div>
                                                    <div className="bg-zinc-900 rounded p-3 text-xs font-mono">
                                                        {renderJsonValue(detail.result.resultData)}
                                                    </div>
                                                    {detail.result.resultMetadata && Object.keys(detail.result.resultMetadata).length > 0 && (
                                                        <div className="mt-2">
                                                            <h5 className="text-xs text-zinc-500 mb-1">Metadata:</h5>
                                                            <div className="bg-zinc-900 rounded p-3 text-xs font-mono">
                                                                {renderJsonValue(detail.result.resultMetadata)}
                                                            </div>
                                                        </div>
                                                    )}
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
                                                                <span className={
                                                                    log.logLevel === 'error' ? 'text-red-400' :
                                                                        log.logLevel === 'warning' ? 'text-yellow-400' :
                                                                            'text-zinc-400'
                                                                }>
                                                                    [{log.logLevel.toUpperCase()}]
                                                                </span>
                                                                {' '}
                                                                <span className="text-zinc-500">
                                                                    {formatDate(log.timestamp)}
                                                                </span>
                                                                {' '}
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

