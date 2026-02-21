/**
 * ChatInput Component
 *
 * Input area with:
 * - Textarea for message
 * - Upload button
 * - Send button
 */

"use client";

import { useState, useRef, KeyboardEvent, useEffect } from "react";
import { Send, Paperclip, Loader2, X } from "lucide-react";
import { Button } from "../ui/Button";
import { Modal } from "../ui/Modal";
import { ScanUploadZone } from "../scans/ScanUploadZone";
import { classNames } from "../../lib/utils";
import { getImageUrl } from "../../lib/utils/image";
import type { Scan } from "../../lib/types/scan";

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
  placeholder = "Ask a question or describe what you need...",
}: ChatInputProps) {
  const [content, setContent] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [uploadedScans, setUploadedScans] = useState<Scan[]>([]);
  const [sendError, setSendError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const prevChatIdRef = useRef<string>(chatId);

  // Clear uploaded scans when chatId changes (switching chats)
  useEffect(() => {
    if (prevChatIdRef.current !== chatId) {
      setUploadedScans([]);
      setSendError(null);
      prevChatIdRef.current = chatId;
    }
  }, [chatId]);

  const handleSend = async () => {
    if (!content.trim() || isSending || disabled) return;

    setIsSending(true);
    setSendError(null); // Clear previous errors
    const scanIdsToSend = uploadedScans.length > 0 ? uploadedScans.map((s) => s.id) : undefined;
    if (scanIdsToSend && scanIdsToSend.length > 0) {
      console.log(`📤 Sending message with ${scanIdsToSend.length} scan(s):`, scanIdsToSend);
    }

    try {
      // Pass uploaded scan IDs if any
      await onSend(content.trim(), scanIdsToSend);
      setContent("");
      setUploadedScans([]); // Clear uploaded scans after sending
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
        textareaRef.current.focus(); // Return focus to input
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : "Failed to send message";
      console.error("❌ Failed to send message:", error);
      setSendError(errorMsg);
      // Focus textarea so user can try again
      textareaRef.current?.focus();
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setContent(e.target.value);

    // Auto-resize textarea
    const textarea = e.target;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
  };

  const handleUploadComplete = (scans: Scan[]) => {
    setIsUploadModalOpen(false);
    // Store full scan objects to show preview and allow deletion
    setUploadedScans((prev) => [...prev, ...scans]);
    console.log(`✅ Scans uploaded successfully:`, scans.map((s) => s.id));
    console.log(
      `📎 Scans ready to attach:`,
      scans.map((s) => ({ id: s.id, path: s.filePath }))
    );
  };

  const handleRemoveScan = (scanId: string) => {
    setUploadedScans((prev) => prev.filter((s) => s.id !== scanId));
    console.log(`🗑️ Removed scan:`, scanId);
  };

  const handleUploadError = (error: string) => {
    console.error("Upload error:", error);
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
              "flex-1 px-4 py-2 bg-zinc-800 border border-zinc-700 rounded-lg",
              "text-sm text-white placeholder:text-zinc-500",
              "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              "resize-none overflow-hidden"
            )}
            style={{ minHeight: "42px", maxHeight: "200px" }}
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
            {isSending ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
          </Button>
        </div>

        {/* Error Message */}
        {sendError && (
          <div className="mt-2 p-2 bg-red-900/20 border border-red-800 rounded text-red-400 text-xs">
            {sendError}
          </div>
        )}

        {/* Helper text or Image Previews */}
        {uploadedScans.length === 0 ? (
          <p className="mt-2 text-xs text-zinc-500">
            Press Enter to send, Shift+Enter for new line
          </p>
        ) : (
          <div className="mt-3">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-zinc-400">
                {uploadedScans.length} scan{uploadedScans.length > 1 ? "s" : ""} ready to
                attach
              </p>
              <button
                onClick={() => setUploadedScans([])}
                disabled={isSending}
                className="text-xs text-red-400 hover:text-red-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Clear all
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {uploadedScans.map((scan) => {
                const imageUrl = getImageUrl(scan.displayPath);
                return (
                  <div key={scan.id} className="relative group">
                    {imageUrl ? (
                      <>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={imageUrl}
                          alt="Uploaded scan"
                          className="h-20 w-20 object-cover rounded-lg border border-zinc-700 bg-zinc-800"
                          onError={(e) => {
                            e.currentTarget.style.display = "none";
                            const container = e.currentTarget.parentElement;
                            if (container) {
                              const errorMsg = document.createElement("div");
                              errorMsg.className =
                                "h-20 w-20 flex items-center justify-center bg-red-900/20 border border-red-800 rounded-lg text-red-400 text-xs p-1 text-center";
                              errorMsg.textContent = "Failed to load";
                              container.insertBefore(errorMsg, e.currentTarget);
                            }
                          }}
                        />
                      </>
                    ) : (
                      <div className="h-20 w-20 flex items-center justify-center bg-yellow-900/20 border border-yellow-800 rounded-lg text-yellow-400 text-xs p-1 text-center">
                        No preview
                      </div>
                    )}
                    <button
                      onClick={() => handleRemoveScan(scan.id)}
                      disabled={isSending}
                      className="absolute -top-2 -right-2 p-1 bg-red-500 hover:bg-red-600 rounded-full text-white opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-50"
                      title="Remove scan"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}
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
