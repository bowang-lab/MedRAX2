/**
 * MessageActivity Component
 * 
 * Shows tool executions inline with the message:
 * - Compact list of tools used
 * - Click to see detailed tool history
 */

'use client';

import { Wrench, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import type { ToolExecution } from '../../lib/types/tool';
import { Badge } from '../ui/Badge';

/**
 * MessageActivity Component Props
 * @property executions - Array of tool executions for this message (required)
 * @property onShowDetails - Optional callback when user wants to see execution details
 */
interface MessageActivityProps {
    /** Array of tool executions for this message */
    executions: ToolExecution[];
    /** Optional callback when user wants to see execution details */
    onShowDetails?: (executionId: string) => void;
}

export function MessageActivity({ executions, onShowDetails }: MessageActivityProps) {
    if (!executions || executions.length === 0) return null;

    return (
        <div className="mt-2 p-3 bg-zinc-900/50 rounded-lg border border-zinc-800">
            <div className="flex items-center space-x-2 mb-2">
                <Wrench className="h-4 w-4 text-zinc-400" />
                <span className="text-xs font-medium text-zinc-400">Tool Activity</span>
            </div>

            <div className="space-y-2">
                {(executions || []).map((execution) => (
                    <button
                        key={execution.id}
                        onClick={() => onShowDetails && onShowDetails(execution.id)}
                        className="w-full flex items-center justify-between p-2 rounded hover:bg-zinc-800 transition-colors text-left"
                    >
                        <div className="flex items-center space-x-2 flex-1 min-w-0">
                            {execution.status === 'completed' && (
                                <CheckCircle className="h-4 w-4 text-emerald-400 flex-shrink-0" />
                            )}
                            {execution.status === 'failed' && (
                                <XCircle className="h-4 w-4 text-red-400 flex-shrink-0" />
                            )}
                            {execution.status === 'running' && (
                                <Loader2 className="h-4 w-4 text-blue-400 flex-shrink-0 animate-spin" />
                            )}
                            {execution.status === 'pending' && (
                                <Loader2 className="h-4 w-4 text-zinc-500 flex-shrink-0" />
                            )}

                            <span className="text-sm text-zinc-300 truncate">
                                {execution.toolDisplayName}
                            </span>
                        </div>

                        <div className="flex items-center space-x-2">
                            {execution.executionTimeMs && (
                                <span className="text-xs text-zinc-500">
                                    {(execution.executionTimeMs / 1000).toFixed(1)}s
                                </span>
                            )}
                            <Badge
                                variant={
                                    execution.status === 'completed'
                                        ? 'success'
                                        : execution.status === 'failed'
                                            ? 'error'
                                            : execution.status === 'running'
                                                ? 'info'
                                                : 'default'
                                }
                            >
                                {execution.status}
                            </Badge>
                        </div>
                    </button>
                ))}
            </div>

            {onShowDetails && (
                <button
                    onClick={() => executions && executions.length > 0 && onShowDetails(executions[0].id)}
                    className="mt-2 text-xs text-blue-400 hover:text-blue-300 transition-colors"
                >
                    Show detailed tool history
                </button>
            )}
        </div>
    );
}

