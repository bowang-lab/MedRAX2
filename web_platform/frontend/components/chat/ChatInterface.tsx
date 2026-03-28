/**
 * ChatInterface Component
 *
 * Main chat area with:
 * - Chat header (patient/chat name, actions)
 * - Message list (scrollable)
 * - Suggested questions (floating above input)
 * - Chat input
 */

"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { MoreHorizontal, FileImage } from "lucide-react";
import { Menu } from "@headlessui/react";
import { useAppStore } from "../../lib/store/appStore";
import { getMessages, streamChatResponse } from "../../lib/api/messages";
import { getChat, updateChat, deleteChat } from "../../lib/api/chats";
import { clearChatMemory, getChatMemoryStats, type MemoryStats } from "../../lib/api/memory";
import type { MessageWithDetails, SSEEvent } from "../../lib/types/message";
import { Message } from "./Message";
import { ChatInput } from "./ChatInput";
import { SuggestedQuestions } from "./SuggestedQuestions";
import { ScanGalleryDrawer } from "../scans/ScanGalleryDrawer";
import { Modal } from "../ui/Modal";
import { Input } from "../ui/Input";
import { Button } from "../ui/Button";
import { Spinner } from "../ui/Spinner";
import { classNames } from "../../lib/utils";
import type { Chat } from "../../lib/types/chat";
import type { SuggestedQuestion } from "../../lib/types/question";
import { ToolOutputsSidebar } from "../tool-outputs/ToolOutputsSidebar";

export function ChatInterface() {
  const {
    selectedChatId,
    messages,
    setMessages,
    addMessage,
    isSendingMessage,
    setSendingMessage,
    updateChat: updateChatInStore,
    removeChat,
  } = useAppStore();

  const [currentChat, setCurrentChat] = useState<Chat | null>(null);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isRenameModalOpen, setIsRenameModalOpen] = useState(false);
  const [isScanGalleryOpen, setIsScanGalleryOpen] = useState(false);
  const [isMemoryStatsModalOpen, setIsMemoryStatsModalOpen] = useState(false);
  const [memoryStats, setMemoryStats] = useState<MemoryStats | null>(null);
  const [isLoadingStats, setIsLoadingStats] = useState(false);
  const [chatNameInput, setChatNameInput] = useState("");
  const [isRenamingChat, setIsRenamingChat] = useState(false);
  const [toolOutputsMessageId, setToolOutputsMessageId] = useState<string | null>(null);
  const [isToolOutputsSidebarOpen, setIsToolOutputsSidebarOpen] = useState(false);

  // Store abort function for ongoing stream
  const abortStreamRef = useRef<(() => void) | null>(null);
  const currentStreamChatIdRef = useRef<string | null>(null);
  const lastStreamUserMessageIdRef = useRef<string | null>(null);
  const openedToolSidebarForThisStreamRef = useRef<boolean>(false);

  // Suggested questions (for now, hardcoded defaults)
  const [suggestedQuestions] = useState<SuggestedQuestion[]>([
    { id: "1", doctorId: "", question: "Is there pneumonia?", isDefault: true, displayOrder: 1, createdAt: "" },
    { id: "2", doctorId: "", question: "Measure heart size", isDefault: true, displayOrder: 2, createdAt: "" },
    { id: "3", doctorId: "", question: "What abnormalities do you see?", isDefault: true, displayOrder: 3, createdAt: "" },
    { id: "4", doctorId: "", question: "Generate a report", isDefault: true, displayOrder: 4, createdAt: "" },
  ]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatMessages = useMemo(() => {
    return selectedChatId ? messages[selectedChatId] || [] : [];
  }, [selectedChatId, messages]);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      // Use requestAnimationFrame to ensure DOM is updated
      requestAnimationFrame(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
      });
    }
  }, [chatMessages, chatMessages.length]);

  // Load chat and messages when chat is selected
  const loadChatData = useCallback(async (chatId: string) => {
    setIsLoadingMessages(true);
    setError(null);
    try {
      const [chat, msgs] = await Promise.all([getChat(chatId), getMessages(chatId)]);
      setCurrentChat(chat);
      setMessages(chatId, msgs);
      // Keep sidebar/chat list in sync (message/scan counts, name)
      updateChatInStore(chatId, chat);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load chat");
    } finally {
      setIsLoadingMessages(false);
    }
  }, [setMessages, updateChatInStore]);

  useEffect(() => {
    // Abort any ongoing stream when switching chats
    if (abortStreamRef.current && currentStreamChatIdRef.current !== selectedChatId) {
      console.log("🛑 Aborting stream due to chat switch");
      abortStreamRef.current();
      abortStreamRef.current = null;
      currentStreamChatIdRef.current = null;
      setSendingMessage(false);
    }

    if (selectedChatId) {
      loadChatData(selectedChatId);
    } else {
      setCurrentChat(null);
    }
    // Only re-run when chatId changes, not when data changes (prevents infinite loop)
  }, [loadChatData, selectedChatId, setSendingMessage]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortStreamRef.current) {
        console.log("🛑 Aborting stream on unmount");
        abortStreamRef.current();
      }
    };
  }, []);

  const handleSendMessage = async (content: string, scanIds?: string[]) => {
    if (!selectedChatId) return;

    // Store the chatId at the start of this request
    const requestChatId = selectedChatId;
    currentStreamChatIdRef.current = requestChatId;
    lastStreamUserMessageIdRef.current = null;
    openedToolSidebarForThisStreamRef.current = false;

    setSendingMessage(true);
    setError(null);

    // Add user message optimistically
    const tempUserMessage: MessageWithDetails = {
      id: `temp-${Date.now()}`,
      chatId: requestChatId,
      role: "user",
      content,
      createdAt: new Date().toISOString(),
      attachedScans: [],
      toolExecutions: [],
    };
    addMessage(requestChatId, tempUserMessage);

    // Add assistant message placeholder for real-time updates
    const tempAssistantMessage: MessageWithDetails = {
      id: `temp-assistant-${Date.now()}`,
      chatId: requestChatId,
      role: "assistant",
      content: "",
      createdAt: new Date().toISOString(),
      attachedScans: [],
      toolExecutions: [],
    };
    addMessage(requestChatId, tempAssistantMessage);

    let assistantContent = "";

    // Helper to clean up temp messages
    const cleanupTempMessages = () => {
      const currentMessages = messages[requestChatId] || [];
      const filteredMessages = currentMessages.filter(
        (msg) => msg.id !== tempUserMessage.id && msg.id !== tempAssistantMessage.id
      );
      setMessages(requestChatId, filteredMessages);
    };

    const handleToolEventOpen = (event: SSEEvent) => {
      // Auto-open the tool outputs sidebar the first time a tool runs for this stream
      const msgIdFromEvent = event.data.messageId || event.data.message_id;
      const targetMessageId: string | null =
        (typeof msgIdFromEvent === "string" ? msgIdFromEvent : null) || lastStreamUserMessageIdRef.current;
      if (!openedToolSidebarForThisStreamRef.current && targetMessageId) {
        setToolOutputsMessageId(targetMessageId);
        setIsToolOutputsSidebarOpen(true);
        openedToolSidebarForThisStreamRef.current = true;
      }
    };

    // Stream response and store abort function
    const abortFn = streamChatResponse(
      requestChatId,
      content,
      scanIds || [],
      (event) => {
        // Ignore events if chat has switched
        if (currentStreamChatIdRef.current !== requestChatId) {
          return;
        }

        // Handle streaming events
        if (event.type === "message_start") {
          console.log("Message started:", event.data.messageId);
          // Track the user message id for this stream
          lastStreamUserMessageIdRef.current = (event.data.messageId as string) || null;
        } else if (event.type === "content_chunk") {
          // Update assistant message content in real-time
          assistantContent += (event.data.content as string) || "";
          // Update the temp message
          const currentMessages = messages[requestChatId] || [];
          const updatedMessages = currentMessages.map((msg) =>
            msg.id === tempAssistantMessage.id ? { ...msg, content: assistantContent } : msg
          );
          setMessages(requestChatId, updatedMessages);
        } else if (event.type === "tool_start") {
          console.log("Tool started:", event.data);
          handleToolEventOpen(event);
        } else if (event.type === "tool_output") {
          console.log("Tool output:", event.data);
          handleToolEventOpen(event);
        } else if (event.type === "tool_done") {
          console.log("Tool completed:", event.data);
        }
      },
      () => {
        // On complete - reload to get final state with all tool executions
        abortStreamRef.current = null;
        currentStreamChatIdRef.current = null;
        setSendingMessage(false);
        loadChatData(requestChatId);
      },
      (err) => {
        console.error("Stream error:", err);
        cleanupTempMessages();
        setSendingMessage(false);
        setError(err.message || "Failed to send message");
      }
    );

    abortStreamRef.current = abortFn;
  };

  const handleQuestionClick = (question: string) => {
    if (!selectedChatId) return;
    handleSendMessage(question);
  };

  const handleShowToolDetails = (messageId: string) => {
    setToolOutputsMessageId(messageId);
    setIsToolOutputsSidebarOpen(true);
  };

  const handleRenameChat = async () => {
    if (!currentChat || !chatNameInput.trim()) return;
    setIsRenamingChat(true);
    try {
      const updated = await updateChat(currentChat.id, { name: chatNameInput.trim() });
      setCurrentChat(updated);
      updateChatInStore(updated.id, updated);
      setIsRenameModalOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to rename chat");
    } finally {
      setIsRenamingChat(false);
    }
  };

  const handleDeleteChat = async () => {
    if (!currentChat) return;
    if (!confirm("Are you sure you want to delete this chat?")) return;
    try {
      await deleteChat(currentChat.id);
      removeChat(currentChat.patientId, currentChat.id);
      setCurrentChat(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete chat");
    }
  };

  const openMemoryStats = async () => {
    if (!currentChat) return;
    setIsMemoryStatsModalOpen(true);
    setIsLoadingStats(true);
    try {
      const stats = await getChatMemoryStats(currentChat.id);
      setMemoryStats(stats);
    } catch (err) {
      setMemoryStats(null);
      setError(err instanceof Error ? err.message : "Failed to load memory stats");
    } finally {
      setIsLoadingStats(false);
    }
  };

  const handleClearMemory = async () => {
    if (!currentChat) return;
    if (!confirm("Clear conversation memory for this chat?")) return;
    try {
      await clearChatMemory(currentChat.id);
      await openMemoryStats();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to clear memory");
    }
  };

  return (
    <>
      <div className="flex-1 flex flex-col min-h-0">
        {/* Header */}
        {currentChat && (
          <div className="h-16 border-b border-zinc-800 flex items-center justify-between px-6 bg-zinc-900 flex-shrink-0">
            <div className="flex items-center space-x-3">
              <div className="flex flex-col">
                <h2 className="text-lg font-semibold text-white">{currentChat.name}</h2>
                <div className="text-xs text-zinc-500">
                  {currentChat.messageCount} messages • {currentChat.scanCount} scans
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={() => setIsScanGalleryOpen(true)}
                className="p-2 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-md transition-colors"
                title="View all scans"
              >
                <FileImage className="h-5 w-5" />
              </button>

              <Menu as="div" className="relative">
                <Menu.Button className="p-2 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-md transition-colors">
                  <MoreHorizontal className="h-5 w-5" />
                </Menu.Button>
                <Menu.Items className="absolute right-0 mt-2 w-48 origin-top-right bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl focus:outline-none z-50">
                  <div className="py-1">
                    <Menu.Item>
                      {({ active }) => (
                        <button
                          onClick={() => {
                            setChatNameInput(currentChat.name);
                            setIsRenameModalOpen(true);
                          }}
                          className={classNames(
                            "w-full text-left px-4 py-2 text-sm",
                            active ? "bg-zinc-800 text-white" : "text-zinc-300"
                          )}
                        >
                          Rename Chat
                        </button>
                      )}
                    </Menu.Item>
                    <Menu.Item>
                      {({ active }) => (
                        <button
                          onClick={() => setIsScanGalleryOpen(true)}
                          className={classNames(
                            "w-full text-left px-4 py-2 text-sm",
                            active ? "bg-zinc-800 text-white" : "text-zinc-300"
                          )}
                        >
                          View All Scans
                        </button>
                      )}
                    </Menu.Item>
                    <Menu.Item>
                      {({ active }) => (
                        <button
                          onClick={openMemoryStats}
                          className={classNames(
                            "w-full text-left px-4 py-2 text-sm",
                            active ? "bg-zinc-800 text-white" : "text-zinc-300"
                          )}
                        >
                          View Memory Stats
                        </button>
                      )}
                    </Menu.Item>
                    <Menu.Item>
                      {({ active }) => (
                        <button
                          onClick={handleClearMemory}
                          className={classNames(
                            "w-full text-left px-4 py-2 text-sm",
                            active ? "bg-zinc-800 text-white" : "text-zinc-300"
                          )}
                        >
                          Clear Memory
                        </button>
                      )}
                    </Menu.Item>
                    <Menu.Item>
                      {({ active }) => (
                        <button
                          onClick={handleDeleteChat}
                          className={classNames(
                            "w-full text-left px-4 py-2 text-sm",
                            active ? "bg-zinc-800 text-red-400" : "text-red-500"
                          )}
                        >
                          Delete Chat
                        </button>
                      )}
                    </Menu.Item>
                  </div>
                </Menu.Items>
              </Menu>
            </div>
          </div>
        )}

        {/* Messages Area - Scrollable */}
        <div className="flex-1 overflow-y-auto min-h-0">
          <div className="p-6 space-y-4">
            {isLoadingMessages ? (
              <div className="flex items-center justify-center min-h-[300px]">
                <Spinner size="lg" />
              </div>
            ) : error ? (
              <div className="flex items-center justify-center min-h-[300px]">
                <div className="text-red-400 text-sm">{error}</div>
              </div>
            ) : chatMessages.length > 0 ? (
              <>
                {chatMessages.map((message) => (
                  <Message
                    key={message.id}
                    message={message}
                    onShowToolDetails={() => handleShowToolDetails(message.id)}
                  />
                ))}
                <div ref={messagesEndRef} />
              </>
            ) : (
              <div className="flex items-center justify-center min-h-[300px] text-zinc-500 text-sm">
                No messages yet. Start the conversation below.
              </div>
            )}
          </div>
        </div>

        {/* Bottom Section - Fixed at bottom, never scrolls away */}
        <div className="flex-shrink-0 bg-zinc-950">
          {/* Suggested Questions */}
          <SuggestedQuestions questions={suggestedQuestions} onSelect={handleQuestionClick} />

          {/* Input Area */}
          {selectedChatId && (
            <ChatInput
              chatId={selectedChatId}
              onSend={handleSendMessage}
              disabled={isSendingMessage || isLoadingMessages}
            />
          )}
        </div>
      </div>

      {/* Rename Chat Modal */}
      <Modal
        isOpen={isRenameModalOpen}
        onClose={() => setIsRenameModalOpen(false)}
        title="Rename Chat"
        size="sm"
      >
        <div className="space-y-4">
          <Input
            label="Chat Name"
            value={chatNameInput}
            onChange={(e) => setChatNameInput(e.target.value)}
            placeholder="Enter chat name"
            autoFocus
          />
          <div className="flex items-center justify-end space-x-3">
            <Button
              variant="ghost"
              onClick={() => setIsRenameModalOpen(false)}
              disabled={isRenamingChat}
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={handleRenameChat}
              isLoading={isRenamingChat}
              disabled={isRenamingChat || !chatNameInput.trim()}
            >
              Rename
            </Button>
          </div>
        </div>
      </Modal>

      {/* Scan Gallery Drawer */}
      <ScanGalleryDrawer
        isOpen={isScanGalleryOpen}
        patientId={currentChat?.patientId || null}
        onClose={() => setIsScanGalleryOpen(false)}
      />

      {/* Tool Outputs Sidebar */}
      <ToolOutputsSidebar
        messageId={toolOutputsMessageId}
        isOpen={isToolOutputsSidebarOpen}
        onClose={() => {
          setIsToolOutputsSidebarOpen(false);
          setToolOutputsMessageId(null);
        }}
      />

      {/* Memory Stats Modal */}
      <Modal
        isOpen={isMemoryStatsModalOpen}
        onClose={() => setIsMemoryStatsModalOpen(false)}
        title="Chat Memory Statistics"
        size="sm"
      >
        {isLoadingStats ? (
          <div className="flex items-center justify-center py-8">
            <Spinner size="lg" />
          </div>
        ) : memoryStats ? (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-zinc-800 rounded-lg p-4">
                <div className="text-xs text-zinc-500 mb-1">Messages</div>
                <div className="text-2xl font-semibold text-white">{memoryStats.messageCount}</div>
              </div>
              <div className="bg-zinc-800 rounded-lg p-4">
                <div className="text-xs text-zinc-500 mb-1">Scans</div>
                <div className="text-2xl font-semibold text-white">{memoryStats.scanCount}</div>
              </div>
              <div className="bg-zinc-800 rounded-lg p-4">
                <div className="text-xs text-zinc-500 mb-1">Tool Runs</div>
                <div className="text-2xl font-semibold text-white">{memoryStats.toolExecutionCount}</div>
              </div>
            </div>
            <div className="bg-zinc-800 rounded-lg p-4">
              <div className="text-xs text-zinc-500 mb-1">Context Status</div>
              <div className="text-sm text-white">
                {memoryStats.hasContext ? (
                  <span className="text-green-400">✓ Active conversation context</span>
                ) : (
                  <span className="text-zinc-400">No context (fresh start)</span>
                )}
              </div>
            </div>
            <div className="text-xs text-zinc-500">
              Chat ID: <span className="font-mono text-zinc-400">{memoryStats.chatId.substring(0, 8)}...</span>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-zinc-500">Failed to load memory stats</div>
        )}
      </Modal>
    </>
  );
}
