/**
 * Message Component
 * 
 * Individual message bubble with:
 * - Sender role (user/assistant)
 * - Content
 * - Activity section (tool executions inline)
 * - Attached scans
 * - Final generated images from tool executions
 */

'use client';

import { useState } from 'react';
import { User, Bot } from 'lucide-react';
import type { MessageWithDetails } from '../../lib/types/message';
import { MessageActivity } from './MessageActivity';
import { classNames, formatDateTime } from '../../lib/utils';
import { getImageUrl } from '../../lib/utils/image';
import { ImageModal } from '../ui/ImageModal';

/**
 * Extract the most relevant generated images from tool executions
 * Prioritizes: segmentation masks, grounding visualizations, generated x-rays
 */
function extractFinalImages(message: MessageWithDetails): string[] {
    if (!message.toolExecutions || message.toolExecutions.length === 0) {
        return [];
    }

    const finalImages: string[] = [];

    // Look through tool executions for image_paths (generated images)
    message.toolExecutions.forEach((execution) => {
        if (execution.imagePaths && Array.isArray(execution.imagePaths)) {
            // Add all non-input images
            execution.imagePaths.forEach((path) => {
                if (path && typeof path === 'string' && !path.toLowerCase().includes('input')) {
                    finalImages.push(path);
                }
            });
        }
    });

    // Return unique images, limited to 3 most recent
    return [...new Set(finalImages)].slice(-3);
}

/**
 * Message Component Props
 * @property message - The complete message data including attached scans and tool executions
 * @property onShowToolDetails - Optional callback triggered when user clicks to view detailed tool execution info
 */
interface MessageProps {
    /** The message to display with all its details (scans, tool executions) */
    message: MessageWithDetails;
    /** Optional callback when user clicks to view tool execution details for this message */
    onShowToolDetails?: () => void;
}

export function Message({ message, onShowToolDetails }: MessageProps) {
    const isUser = message.role === 'user';
    const isAssistant = message.role === 'assistant';

    // Extract final generated images for assistant messages
    const finalImages = isAssistant ? extractFinalImages(message) : [];

    // State for image modal
    const [modalImages, setModalImages] = useState<string[]>([]);
    const [modalInitialIndex, setModalInitialIndex] = useState(0);
    const [isModalOpen, setIsModalOpen] = useState(false);

    // Track which images failed to load
    const [failedImages, setFailedImages] = useState<Set<string>>(new Set());

    const openImageModal = (images: string[], index: number = 0) => {
        // Only open modal if we have valid images
        if (images.length === 0) return;
        setModalImages(images);
        setModalInitialIndex(index);
        setIsModalOpen(true);
    };

    const handleImageError = (imagePath: string) => {
        setFailedImages(prev => new Set(prev).add(imagePath));
    };

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

                        {/* Attached Scans (User messages) */}
                        {message.attachedScans && message.attachedScans.length > 0 && (
                            <div className="mt-3">
                                <p className="text-xs text-zinc-500 mb-2">Attached Scans:</p>
                                <div className="flex flex-wrap gap-3">
                                    {message.attachedScans.map((scan, idx) => {
                                        const scanUrl = getImageUrl(scan.displayPath);
                                        const hasFailed = failedImages.has(scanUrl);

                                        return (
                                            <div
                                                key={scan.id}
                                                className={classNames(
                                                    'relative group w-32 h-32 bg-zinc-800/50 rounded-lg border border-zinc-700 overflow-hidden flex items-center justify-center transition-colors',
                                                    hasFailed ? 'cursor-not-allowed' : 'cursor-pointer hover:border-blue-500'
                                                )}
                                                onClick={() => {
                                                    if (!hasFailed) {
                                                        openImageModal(
                                                            message.attachedScans!.map(s => getImageUrl(s.displayPath)),
                                                            idx
                                                        );
                                                    }
                                                }}
                                            >
                                                {hasFailed ? (
                                                    <div className="text-red-400 text-xs text-center p-2">
                                                        ⚠️ Failed to load
                                                    </div>
                                                ) : (
                                                    <>
                                                        {/* eslint-disable-next-line @next/next/no-img-element */}
                                                        <img
                                                            src={scanUrl}
                                                            alt="Medical Scan"
                                                            className="max-h-full max-w-full object-contain"
                                                            onError={() => handleImageError(scanUrl)}
                                                        />
                                                        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-opacity pointer-events-none" />
                                                        <div className="absolute bottom-1 right-1 bg-black/70 rounded px-2 py-1 text-[10px] text-white opacity-0 group-hover:opacity-100 transition-opacity">
                                                            Click to enlarge
                                                        </div>
                                                    </>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        {/* Final Generated Images (Assistant messages) */}
                        {finalImages.length > 0 && (
                            <div className="mt-3">
                                <p className="text-xs text-zinc-500 mb-2">Generated Result:</p>
                                <div className="flex flex-wrap gap-3">
                                    {finalImages.map((imagePath, idx) => {
                                        const imageUrl = getImageUrl(imagePath);
                                        const hasFailed = failedImages.has(imageUrl);

                                        return (
                                            <div
                                                key={idx}
                                                className={classNames(
                                                    'relative group w-48 h-48 bg-zinc-800/50 rounded-lg border overflow-hidden flex items-center justify-center transition-colors',
                                                    hasFailed
                                                        ? 'border-red-800 cursor-not-allowed'
                                                        : 'border-blue-500 cursor-zoom-in hover:border-blue-400'
                                                )}
                                                onClick={() => {
                                                    if (!hasFailed) {
                                                        openImageModal(
                                                            finalImages.map(path => getImageUrl(path)),
                                                            idx
                                                        );
                                                    }
                                                }}
                                            >
                                                {hasFailed ? (
                                                    <div className="text-red-400 text-xs text-center p-2">
                                                        ⚠️ Failed to load result
                                                    </div>
                                                ) : (
                                                    <>
                                                        {/* eslint-disable-next-line @next/next/no-img-element */}
                                                        <img
                                                            src={imageUrl}
                                                            alt={`Generated result ${idx + 1}`}
                                                            className="max-h-full max-w-full object-contain"
                                                            onError={() => handleImageError(imageUrl)}
                                                        />
                                                        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-opacity pointer-events-none" />
                                                        <div className="absolute bottom-1 right-1 bg-black/70 rounded px-2 py-1 text-[10px] text-white opacity-0 group-hover:opacity-100 transition-opacity">
                                                            Click to enlarge
                                                        </div>
                                                    </>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
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

            {/* Image Modal */}
            <ImageModal
                images={modalImages}
                initialIndex={modalInitialIndex}
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
            />
        </div>
    );
}

