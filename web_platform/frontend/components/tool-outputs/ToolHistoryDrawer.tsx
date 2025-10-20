'use client';

/**
 * Tool History Drawer Component
 * 
 * Displays tool execution history for a message in a drawer/modal.
 * Shows timeline of tool executions with details.
 */

import { useEffect, useState, useCallback } from 'react';
import { Drawer } from '../ui/Drawer';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';
import { Spinner } from '../ui/Spinner';
import { getMessageToolHistory, getToolExecution } from '@/lib/api/toolHistory';
import type { ToolExecution } from '@/lib/types/tool';

export interface ToolHistoryDrawerProps {
    /** Message ID to show tool history for */
    messageId: string;
    /** Whether the drawer is open */
    isOpen: boolean;
    /** Callback when drawer is closed */
    onClose: () => void;
}

/**
 * Drawer component to display tool execution history for a message.
 */
export function ToolHistoryDrawer({ messageId, isOpen, onClose }: ToolHistoryDrawerProps) {
    const [executions, setExecutions] = useState<ToolExecution[]>([]);
    const [selectedExecution, setSelectedExecution] = useState<ToolExecution | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const loadToolHistory = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const history = await getMessageToolHistory(messageId);
            setExecutions(history);
        } catch (err) {
            const errorMessage = (err as { message?: string })?.message || 'Failed to load tool history';
            setError(errorMessage);
            console.error('Failed to load tool history:', errorMessage);
        } finally {
            setLoading(false);
        }
    }, [messageId]);

    useEffect(() => {
        if (isOpen && messageId) {
            loadToolHistory();
        }
    }, [isOpen, messageId, loadToolHistory]);

    const handleExecutionClick = async (executionId: string) => {
        try {
            const details = await getToolExecution(executionId);
            setSelectedExecution(details);
        } catch (err) {
            console.error('Failed to load execution details:', err);
        }
    };

    const getStatusColor = (status: string) => {
        switch (status.toLowerCase()) {
            case 'completed':
            case 'success':
                return 'success';
            case 'running':
            case 'pending':
                return 'info';
            case 'failed':
            case 'error':
                return 'error';
            default:
                return 'default';
        }
    };

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    };

    return (
        <Drawer isOpen={isOpen} onClose={onClose} title="Tool Execution History" size="lg">
            <div className="flex flex-col h-full">
                {loading && (
                    <div className="flex items-center justify-center py-12">
                        <Spinner size="lg" />
                    </div>
                )}

                {error && (
                    <div className="bg-red-900/20 border border-red-800 rounded-lg p-4 text-red-400">
                        {error}
                    </div>
                )}

                {!loading && !error && (!executions || executions.length === 0) && (
                    <div className="text-center py-12 text-zinc-500">
                        No tool executions found for this message.
                    </div>
                )}

                {!loading && !error && executions && executions.length > 0 && (
                    <div className="flex-1 overflow-y-auto">
                        {!selectedExecution ? (
                            // Timeline view
                            <div className="space-y-4">
                                <div className="text-sm text-zinc-400 mb-4">
                                    {executions.length} tool execution{executions.length !== 1 ? 's' : ''}
                                </div>

                                {executions.map((execution, index) => (
                                    <div
                                        key={execution.id}
                                        className="relative pl-8 pb-6 cursor-pointer hover:bg-zinc-800/50 -mx-4 px-4 py-2 rounded-lg transition-colors"
                                        onClick={() => handleExecutionClick(execution.id)}
                                    >
                                        {/* Timeline line */}
                                        {index < executions.length - 1 && (
                                            <div className="absolute left-4 top-8 bottom-0 w-0.5 bg-zinc-700" />
                                        )}

                                        {/* Timeline dot */}
                                        <div className="absolute left-2.5 top-2 w-3 h-3 rounded-full bg-emerald-500 ring-4 ring-zinc-900" />

                                        {/* Content */}
                                        <div className="space-y-2">
                                            <div className="flex items-center justify-between">
                                                <h4 className="font-medium text-white">
                                                    {execution.toolName}
                                                </h4>
                                                <Badge variant={getStatusColor(execution.status)}>
                                                    {execution.status}
                                                </Badge>
                                            </div>

                                            <div className="text-sm text-zinc-400">
                                                Started: {formatDate(execution.startedAt)}
                                                {execution.completedAt && (
                                                    <span className="ml-3">
                                                        Completed: {formatDate(execution.completedAt)}
                                                    </span>
                                                )}
                                            </div>

                                            {execution.imagePaths && execution.imagePaths.length > 0 && (
                                                <div className="text-xs text-zinc-500">
                                                    📸 {execution.imagePaths.length} image{execution.imagePaths.length !== 1 ? 's' : ''} used
                                                </div>
                                            )}

                                            {execution.requestId && (
                                                <div className="text-xs text-zinc-600">
                                                    Request: {execution.requestId.substring(0, 8)}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            // Detailed view
                            <div>
                                <Button
                                    variant="secondary"
                                    size="sm"
                                    onClick={() => setSelectedExecution(null)}
                                    className="mb-4"
                                >
                                    ← Back to Timeline
                                </Button>

                                <div className="space-y-4">
                                    <div>
                                        <h3 className="text-lg font-semibold text-white mb-2">
                                            {selectedExecution.toolName}
                                        </h3>
                                        <Badge variant={getStatusColor(selectedExecution.status)}>
                                            {selectedExecution.status}
                                        </Badge>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4 text-sm">
                                        <div>
                                            <div className="text-zinc-400 mb-1">Started</div>
                                            <div className="text-white">
                                                {new Date(selectedExecution.startedAt).toLocaleString()}
                                            </div>
                                        </div>

                                        {selectedExecution.completedAt && (
                                            <div>
                                                <div className="text-zinc-400 mb-1">Completed</div>
                                                <div className="text-white">
                                                    {new Date(selectedExecution.completedAt).toLocaleString()}
                                                </div>
                                            </div>
                                        )}

                                        {selectedExecution.requestId && (
                                            <div className="col-span-2">
                                                <div className="text-zinc-400 mb-1">Request ID</div>
                                                <div className="text-white font-mono text-xs">
                                                    {selectedExecution.requestId}
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    {selectedExecution.imagePaths && selectedExecution.imagePaths.length > 0 && (
                                        <div>
                                            <div className="text-zinc-400 text-sm mb-2">Images Used</div>
                                            <div className="space-y-1">
                                                {selectedExecution.imagePaths.map((path, idx) => (
                                                    <div
                                                        key={idx}
                                                        className="text-xs text-zinc-500 font-mono bg-zinc-800 px-2 py-1 rounded"
                                                    >
                                                        {path}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </Drawer>
    );
}

