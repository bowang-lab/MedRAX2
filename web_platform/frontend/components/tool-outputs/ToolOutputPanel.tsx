/**
 * ToolOutputPanel Component
 * 
 * Right panel showing detailed tool execution information.
 * Opens when user clicks "Show detailed tool history" in a message.
 */

'use client';

import { useState, useEffect, useCallback } from 'react';
import { X } from 'lucide-react';
import { useAppStore } from '../../lib/store/appStore';
import { getToolExecutionDetail } from '../../lib/api/tools';
import { Spinner } from '../ui/Spinner';
import { ToolExecutionTimeline } from './ToolExecutionTimeline';
import { ToolResultCard } from './ToolResultCard';
import type { ToolExecution, ToolExecutionLog, ToolExecutionResult } from '../../lib/types/tool';

export function ToolOutputPanel() {
    const {
        isToolPanelOpen,
        selectedToolExecutionId,
        closeToolPanel,
    } = useAppStore();

    const [execution, setExecution] = useState<ToolExecution | null>(null);
    const [logs, setLogs] = useState<ToolExecutionLog[]>([]);
    const [result, setResult] = useState<ToolExecutionResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const loadExecutionDetail = useCallback(async () => {
        if (!selectedToolExecutionId) return;

        setIsLoading(true);
        setError(null);
        try {
            const data = await getToolExecutionDetail(selectedToolExecutionId);
            setExecution(data.execution);
            setLogs(data.logs);
            setResult(data.result);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load tool execution details');
        } finally {
            setIsLoading(false);
        }
    }, [selectedToolExecutionId]);

    useEffect(() => {
        if (isToolPanelOpen && selectedToolExecutionId) {
            loadExecutionDetail();
        }
    }, [isToolPanelOpen, selectedToolExecutionId, loadExecutionDetail]);

    if (!isToolPanelOpen) return null;

    return (
        <aside className="w-96 bg-zinc-900 border-l border-zinc-800 flex flex-col">
            {/* Header */}
            <div className="h-16 border-b border-zinc-800 flex items-center justify-between px-4 flex-shrink-0">
                <h2 className="text-lg font-semibold text-white">Tool Details</h2>
                <button
                    onClick={closeToolPanel}
                    className="p-2 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-md transition-colors"
                >
                    <X className="h-5 w-5" />
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {isLoading ? (
                    <div className="flex items-center justify-center py-12">
                        <Spinner size="md" />
                    </div>
                ) : error ? (
                    <div className="text-red-400 text-sm text-center py-12">{error}</div>
                ) : execution ? (
                    <>
                        {/* Tool Info */}
                        <div className="p-4 bg-zinc-800 rounded-lg">
                            <h3 className="text-sm font-semibold text-white mb-2">
                                {execution.toolDisplayName}
                            </h3>
                            <div className="text-xs text-zinc-400 space-y-1">
                                <p>Status: <span className="text-zinc-300">{execution.status}</span></p>
                                <p>Started: <span className="text-zinc-300">{new Date(execution.startedAt).toLocaleString()}</span></p>
                                {execution.completedAt && (
                                    <p>Completed: <span className="text-zinc-300">{new Date(execution.completedAt).toLocaleString()}</span></p>
                                )}
                                {execution.executionTimeMs && (
                                    <p>Duration: <span className="text-zinc-300">{(execution.executionTimeMs / 1000).toFixed(2)}s</span></p>
                                )}
                            </div>
                        </div>

                        {/* Tool Result */}
                        {result && (
                            <ToolResultCard
                                toolName={execution.toolName}
                                result={result}
                            />
                        )}

                        {/* Execution Logs */}
                        {logs.length > 0 && (
                            <ToolExecutionTimeline logs={logs} />
                        )}
                    </>
                ) : (
                    <div className="text-zinc-500 text-sm text-center py-12">
                        No execution selected
                    </div>
                )}
            </div>
        </aside>
    );
}
