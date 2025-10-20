/**
 * ChatInput Component
 * 
 * Input area with:
 * - Textarea for message
 * - Upload button
 * - Send button
 */

'use client';

import { useState, useRef, KeyboardEvent } from 'react';
import { Send, Paperclip, Loader2 } from 'lucide-react';
import { Button } from '../ui/Button';
import { classNames } from '../../lib/utils';

/**
 * ChatInput Component Props
 * @property onSend - Callback when user sends a message (required, async)
 * @property disabled - Whether input is disabled (default: false)
 * @property placeholder - Custom placeholder text
 */
interface ChatInputProps {
    /** Callback when user sends a message (async) */
    onSend: (content: string, scanIds?: string[]) => Promise<void>;
    /** Whether input is disabled */
    disabled?: boolean;
    /** Custom placeholder text */
    placeholder?: string;
}

export function ChatInput({
    onSend,
    disabled = false,
    placeholder = 'Ask a question or describe what you need...',
}: ChatInputProps) {
    const [content, setContent] = useState('');
    const [isSending, setIsSending] = useState(false);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const handleSend = async () => {
        if (!content.trim() || isSending || disabled) return;

        setIsSending(true);
        try {
            await onSend(content.trim());
            setContent('');
            // Reset textarea height
            if (textareaRef.current) {
                textareaRef.current.style.height = 'auto';
            }
        } catch (error) {
            console.error('Failed to send message:', error);
        } finally {
            setIsSending(false);
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setContent(e.target.value);

        // Auto-resize textarea
        const textarea = e.target;
        textarea.style.height = 'auto';
        textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    };

    return (
        <div className="p-4 bg-zinc-900 border-t border-zinc-800">
            <div className="flex items-end space-x-2">
                {/* Upload Button */}
                <Button
                    variant="ghost"
                    size="md"
                    disabled={disabled || isSending}
                    className="flex-shrink-0"
                    title="Upload scan"
                >
                    <Paperclip className="h-5 w-5" />
                </Button>

                {/* Textarea */}
                <textarea
                    ref={textareaRef}
                    value={content}
                    onChange={handleTextareaChange}
                    onKeyDown={handleKeyDown}
                    placeholder={placeholder}
                    disabled={disabled || isSending}
                    rows={1}
                    className={classNames(
                        'flex-1 px-4 py-2 bg-zinc-800 border border-zinc-700 rounded-lg',
                        'text-sm text-white placeholder:text-zinc-500',
                        'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
                        'disabled:opacity-50 disabled:cursor-not-allowed',
                        'resize-none overflow-hidden'
                    )}
                    style={{ minHeight: '42px', maxHeight: '200px' }}
                />

                {/* Send Button */}
                <Button
                    variant="primary"
                    size="md"
                    onClick={handleSend}
                    disabled={!content.trim() || disabled || isSending}
                    isLoading={isSending}
                    className="flex-shrink-0"
                >
                    {isSending ? (
                        <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                        <Send className="h-5 w-5" />
                    )}
                </Button>
            </div>

            {/* Helper text */}
            <p className="mt-2 text-xs text-zinc-500">
                Press Enter to send, Shift+Enter for new line
            </p>
        </div>
    );
}

