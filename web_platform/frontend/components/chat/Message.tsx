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

"use client";

import { useState } from "react";
import { User, Bot } from "lucide-react";
import type { MessageWithDetails } from "../../lib/types/message";
import { MessageActivity } from "./MessageActivity";
import { classNames, formatDateTime } from "../../lib/utils";
import { getImageUrl } from "../../lib/utils/image";
import { ImageModal } from "../ui/ImageModal";
import { MarkdownRenderer } from "../ui/MarkdownRenderer";

/**
 * Extract final generated images from tool executions to display in the message.
 *
 * Strategy:
 * 1. Backend appends generated images to image_paths: [input_images, ...generated_images]
 * 2. Generated images are under medrax/generated/ (legacy: temp/)
 * 3. Input images are under medrax/uploads/ or have "input" in the name
 * 4. We want to show only the FINAL OUTPUT images (not inputs, not intermediate)
 *
 * @param message - The message with tool executions
 * @returns Array of image paths to display (up to 3 most recent)
 */
function extractFinalImages(message: MessageWithDetails): string[] {
  if (!message.toolExecutions || message.toolExecutions.length === 0) {
    return [];
  }

  const finalImages: string[] = [];

  // Look through tool executions for generated images
  message.toolExecutions.forEach((execution) => {
    if (execution.imagePaths && Array.isArray(execution.imagePaths)) {
      execution.imagePaths.forEach((path) => {
        if (!path || typeof path !== "string") return;

        const lowerPath = path.toLowerCase();

        // Exclude input images (uploads tree or have "input" in name)
        if (
          lowerPath.includes("uploads/") ||
          lowerPath.includes("medrax/uploads") ||
          lowerPath.includes("input")
        ) {
          return;
        }

        // Include generated images (medrax/generated/, legacy temp/, etc.)
        if (
          lowerPath.includes("medrax/generated") ||
          lowerPath.includes("temp/") ||
          lowerPath.includes("segmentation") ||
          lowerPath.includes("visualization") ||
          lowerPath.includes("mask") ||
          lowerPath.includes("output") ||
          lowerPath.includes("grounding") ||
          lowerPath.includes("generation")
        ) {
          finalImages.push(path);
        }
      });
    }
  });

  // Return unique images, limited to 3 most recent
  // This ensures we show the latest outputs without overwhelming the message
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
  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";

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
    setFailedImages((prev) => new Set(prev).add(imagePath));
  };

  return (
    <div className={classNames("flex", isUser ? "justify-end" : "justify-start")}>
      <div className={classNames("flex max-w-3xl", isUser ? "flex-row-reverse" : "flex-row")}>
        {/* Avatar */}
        <div
          className={classNames(
            "flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center",
            isUser ? "bg-blue-600 ml-3" : "bg-zinc-700 mr-3"
          )}
        >
          {isUser ? <User className="h-5 w-5 text-white" /> : <Bot className="h-5 w-5 text-white" />}
        </div>

        {/* Content */}
        <div className="flex-1">
          <div
            className={classNames(
              "rounded-lg px-4 py-3",
              isUser ? "bg-blue-600 text-white" : "bg-zinc-800 text-zinc-100"
            )}
          >
            {/* Render message content as Markdown for assistant, plain text for user */}
            {isUser ? (
              <p className="text-sm whitespace-pre-wrap">{message.content}</p>
            ) : (
              <div className="text-sm prose prose-invert max-w-none">
                <MarkdownRenderer content={message.content} />
              </div>
            )}

            {/* Attached Scans (User messages) */}
            {message.attachedScans && message.attachedScans.length > 0 && (
              <div className="mt-3">
                <p className="text-xs text-zinc-500 mb-2">Attached Scans:</p>
                <div className="flex flex-wrap gap-3">
                  {message.attachedScans.map((scan, idx) => {
                    const scanUrl = getImageUrl(scan.displayPath);
                    const hasFailed = !scanUrl || failedImages.has(scanUrl);

                    return (
                      <div
                        key={scan.id}
                        className={classNames(
                          "relative group w-32 h-32 bg-zinc-800/50 rounded-lg border border-zinc-700 overflow-hidden flex items-center justify-center transition-colors",
                          hasFailed ? "cursor-not-allowed" : "cursor-pointer hover:border-blue-500"
                        )}
                        onClick={() => {
                          if (!hasFailed && scanUrl && message.attachedScans) {
                            openImageModal(
                              message.attachedScans
                                .map((s) => getImageUrl(s.displayPath))
                                .filter((url): url is string => url !== null),
                              idx
                            );
                          }
                        }}
                      >
                        {hasFailed ? (
                          <div className="text-red-400 text-xs text-center p-2">⚠️ Failed to load</div>
                        ) : (
                          <>
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                              src={scanUrl}
                              alt="Medical Scan"
                              className="absolute inset-0 w-full h-full object-contain"
                              onError={() => handleImageError(scanUrl)}
                            />
                            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-opacity pointer-events-none" />
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
                    const hasFailed = !imageUrl || failedImages.has(imageUrl);

                    return (
                      <div
                        key={idx}
                        className={classNames(
                          "relative group w-48 h-48 bg-zinc-800/50 rounded-lg border overflow-hidden flex items-center justify-center transition-colors",
                          hasFailed
                            ? "border-red-800 cursor-not-allowed"
                            : "border-blue-500 cursor-zoom-in hover:border-blue-400"
                        )}
                        onClick={() => {
                          if (!hasFailed && imageUrl) {
                            openImageModal(
                              finalImages
                                .map((path) => getImageUrl(path))
                                .filter((url): url is string => url !== null),
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
                              className="absolute inset-0 w-full h-full object-contain"
                              onError={() => handleImageError(imageUrl)}
                            />
                            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-opacity pointer-events-none" />
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

          {/* Tool Activity */}
          {message.toolExecutions && message.toolExecutions.length > 0 && (
            <MessageActivity executions={message.toolExecutions} onShowDetails={onShowToolDetails} />
          )}

          {/* Timestamp */}
          <div
            className={classNames(
              "text-xs text-zinc-500 mt-1",
              isUser ? "text-right" : "text-left"
            )}
          >
            {formatDateTime(message.createdAt)}
          </div>
        </div>
      </div>

      {/* Image Modal */}
      <ImageModal
        key={`${isModalOpen}-${modalInitialIndex}-${modalImages.length}`}
        images={modalImages}
        initialIndex={modalInitialIndex}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </div>
  );
}
