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
import { Modal } from '../ui/Modal';
import { ScanUploadZone } from '../scans/ScanUploadZone';
import { classNames } from '../../lib/utils';
import type { Scan } from '../../lib/types/scan';

/**
 * ChatInput Component Props
 * @property chatId - The current chat ID (required for uploads)
 * @property onSend - Callback when user sends a message (required, async)
 * @property disabled - Whether input is disabled (default: false)
 * @property placeholder - Custom placeholder text
 */
interface ChatInputProps {
    /** The current chat ID (required for uploads) */
    chatId: string;
    /** Callback when user sends a message (async) */
    onSend: (content: string, scanIds?: string[]) => Promise<void>;
    /** Whether input is disabled */
    disabled?: boolean;
    /** Custom placeholder text */
    placeholder?: string;
}

export function ChatInput({
    chatId,
    onSend,
    disabled = false,
    placeholder = 'Ask a question or describe what you need...',
}: ChatInputProps) {
    const [content, setContent] = useState('');
    const [isSending, setIsSending] = useState(false);
    const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
    const [uploadedScanIds, setUploadedScanIds] = useState<string[]>([]);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const handleSend = async () => {
        if (!content.trim() || isSending || disabled) return;

        setIsSending(true);
        const scanIdsToSend = uploadedScanIds.length > 0 ? uploadedScanIds : undefined;
        console.log(`📤 Sending message with scan IDs:`, scanIdsToSend);

        try {
            // Pass uploaded scan IDs if any
            await onSend(content.trim(), scanIdsToSend);
            setContent('');
            setUploadedScanIds([]); // Clear uploaded scans after sending
            // Reset textarea height
            if (textareaRef.current) {
                textareaRef.current.style.height = 'auto';
            }
        } catch (error) {
            console.error('❌ Failed to send message:', error);
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

    const handleUploadComplete = (scans: Scan[]) => {
        setIsUploadModalOpen(false);
        // Store scan IDs to attach to next message
        const scanIds = scans.map(s => s.id);
        setUploadedScanIds(scanIds);
        console.log(`✅ Scans uploaded successfully:`, scanIds);
        console.log(`📎 Scans ready to attach:`, scans.map(s => ({ id: s.id, path: s.filePath })));
    };

    const handleUploadError = (error: string) => {
        console.error('Upload error:', error);
        // Error is already shown in the upload zone
    };

    return (
        <>
            <div className="p-4 bg-zinc-900 border-t border-zinc-800">
                <div className="flex items-end space-x-2">
                    {/* Upload Button */}
                    <Button
                        variant="ghost"
                        size="md"
                        disabled={disabled || isSending}
                        onClick={() => setIsUploadModalOpen(true)}
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
                    {uploadedScanIds.length > 0 && (
                        <span className="ml-2 text-blue-400">
                            • {uploadedScanIds.length} scan{uploadedScanIds.length > 1 ? 's' : ''} ready to attach
                        </span>
                    )}
                </p>
            </div>

            {/* Upload Modal */}
            <Modal
                isOpen={isUploadModalOpen}
                onClose={() => setIsUploadModalOpen(false)}
                title="Upload Medical Scans"
                size="lg"
            >
                <ScanUploadZone
                    chatId={chatId}
                    onUploadComplete={handleUploadComplete}
                    onUploadError={handleUploadError}
                />
            </Modal>
        </>
    );
}

