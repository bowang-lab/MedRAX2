/**
 * Message Component
 * 
 * Individual message bubble with:
 * - Sender role (user/assistant)
 * - Content
 * - Activity section (tool executions inline)
 * - Attached scans
 */

'use client';

import { User, Bot } from 'lucide-react';
import type { MessageWithDetails } from '../../lib/types/message';
import { MessageActivity } from './MessageActivity';
import { classNames, formatDateTime } from '../../lib/utils';

/**
 * Message Component Props
 * @property message - The complete message data including attached scans and tool executions
 * @property onShowToolDetails - Optional callback triggered when user clicks to view detailed tool execution info
 */
interface MessageProps {
    /** The message to display with all its details (scans, tool executions) */
    message: MessageWithDetails;
    /** Optional callback when user clicks to view tool execution details */
    onShowToolDetails?: (executionId: string) => void;
}

export function Message({ message, onShowToolDetails }: MessageProps) {
    const isUser = message.role === 'user';
    const isAssistant = message.role === 'assistant';

    return (
        <div
            className={classNames(
                'flex',
                isUser ? 'justify-end' : 'justify-start'
            )}
        >
            <div className={classNames('flex max-w-3xl', isUser ? 'flex-row-reverse' : 'flex-row')}>
                {/* Avatar */}
                <div
                    className={classNames(
                        'flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center',
                        isUser ? 'bg-blue-600 ml-3' : 'bg-zinc-700 mr-3'
                    )}
                >
                    {isUser ? (
                        <User className="h-5 w-5 text-white" />
                    ) : (
                        <Bot className="h-5 w-5 text-white" />
                    )}
                </div>

                {/* Content */}
                <div className="flex-1">
                    <div
                        className={classNames(
                            'rounded-lg px-4 py-3',
                            isUser
                                ? 'bg-blue-600 text-white'
                                : 'bg-zinc-800 text-zinc-100'
                        )}
                    >
                        <p className="text-sm whitespace-pre-wrap">{message.content}</p>

                        {/* Attached Scans */}
                        {message.attachedScans && message.attachedScans.length > 0 && (
                            <div className="mt-3 flex flex-wrap gap-2">
                                {message.attachedScans.map((scan) => (
                                    // eslint-disable-next-line @next/next/no-img-element -- Dynamic medical images from backend
                                    <img
                                        key={scan.id}
                                        src={scan.displayPath}
                                        alt="Scan"
                                        className="h-24 w-24 object-cover rounded border border-zinc-700"
                                    />
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Tool Activity (for assistant messages) */}
                    {isAssistant && message.toolExecutions && message.toolExecutions.length > 0 && (
                        <MessageActivity
                            executions={message.toolExecutions}
                            onShowDetails={onShowToolDetails}
                        />
                    )}

                    {/* Timestamp */}
                    <div
                        className={classNames(
                            'text-xs text-zinc-500 mt-1',
                            isUser ? 'text-right' : 'text-left'
                        )}
                    >
                        {formatDateTime(message.createdAt)}
                    </div>
                </div>
            </div>
        </div>
    );
}

