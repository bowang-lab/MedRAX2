/**
 * ChatInterface Component
 * 
 * Main chat area with:
 * - Chat header (patient/chat name, actions)
 * - Message list (scrollable)
 * - Suggested questions (floating above input)
 * - Chat input
 */

'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import { MoreHorizontal, FileImage } from 'lucide-react';
import { Menu } from '@headlessui/react';
import { useAppStore } from '../../lib/store/appStore';
import { getMessages, streamChatResponse } from '../../lib/api/messages';
import { getChat, updateChat, deleteChat } from '../../lib/api/chats';
import { clearChatMemory, getChatMemoryStats, type MemoryStats } from '../../lib/api/memory';
import type { MessageWithDetails } from '../../lib/types/message';
import { Message } from './Message';
import { ChatInput } from './ChatInput';
import { SuggestedQuestions } from './SuggestedQuestions';
import { ScanGalleryDrawer } from '../scans/ScanGalleryDrawer';
import { Modal } from '../ui/Modal';
import { Input } from '../ui/Input';
import { Button } from '../ui/Button';
import { Spinner } from '../ui/Spinner';
import { classNames } from '../../lib/utils';
import type { Chat } from '../../lib/types/chat';
import type { SuggestedQuestion } from '../../lib/types/question';
import { ToolOutputsSidebar } from '../tool-outputs/ToolOutputsSidebar';

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
    const [chatNameInput, setChatNameInput] = useState('');
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
        { id: '1', doctorId: '', question: 'Is there pneumonia?', isDefault: true, displayOrder: 1, createdAt: '' },
        { id: '2', doctorId: '', question: 'Measure heart size', isDefault: true, displayOrder: 2, createdAt: '' },
        { id: '3', doctorId: '', question: 'What abnormalities do you see?', isDefault: true, displayOrder: 3, createdAt: '' },
        { id: '4', doctorId: '', question: 'Generate a report', isDefault: true, displayOrder: 4, createdAt: '' },
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
                messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
            });
        }
    }, [chatMessages, chatMessages.length]);

    // Load chat and messages when chat is selected
    const loadChatData = async (chatId: string) => {
        setIsLoadingMessages(true);
        setError(null);
        try {
            const [chat, msgs] = await Promise.all([
                getChat(chatId),
                getMessages(chatId),
            ]);
            setCurrentChat(chat);
            setMessages(chatId, msgs);
            // Keep sidebar/chat list in sync (message/scan counts, name)
            updateChatInStore(chatId, chat);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load chat');
        } finally {
            setIsLoadingMessages(false);
        }
    };

    useEffect(() => {
        // Abort any ongoing stream when switching chats
        if (abortStreamRef.current && currentStreamChatIdRef.current !== selectedChatId) {
            console.log('🛑 Aborting stream due to chat switch');
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
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedChatId]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (abortStreamRef.current) {
                console.log('🛑 Aborting stream on unmount');
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
            role: 'user',
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
            role: 'assistant',
            content: '',
            createdAt: new Date().toISOString(),
            attachedScans: [],
            toolExecutions: [],
        };
        addMessage(requestChatId, tempAssistantMessage);

        let assistantContent = '';

        // Helper to clean up temp messages
        const cleanupTempMessages = () => {
            const currentMessages = messages[requestChatId] || [];
            const filteredMessages = currentMessages.filter(
                msg => msg.id !== tempUserMessage.id && msg.id !== tempAssistantMessage.id
            );
            setMessages(requestChatId, filteredMessages);
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
                if (event.type === 'message_start') {
                    console.log('Message started:', event.data.messageId);
                    // Track the user message id for this stream
                    lastStreamUserMessageIdRef.current = event.data.messageId || null;
                } else if (event.type === 'content_chunk') {
                    // Update assistant message content in real-time
                    assistantContent += event.data.content || '';
                    // Update the temp message
                    const currentMessages = messages[requestChatId] || [];
                    const updatedMessages = currentMessages.map(msg =>
                        msg.id === tempAssistantMessage.id
                            ? { ...msg, content: assistantContent }
                            : msg
                    );
                    setMessages(requestChatId, updatedMessages);
                } else if (event.type === 'tool_start') {
                    console.log('Tool started:', event.data);
                    // Auto-open the tool outputs sidebar the first time a tool runs for this stream
                    // SSE events may have messageId or message_id (backend uses snake_case)
                    // The [key: string]: unknown in SSEEvent.data allows for both formats
                    const msgIdFromEvent = event.data.messageId || event.data.message_id;
                    const targetMessageId: string | null = 
                        (typeof msgIdFromEvent === 'string' ? msgIdFromEvent : null) || 
                        lastStreamUserMessageIdRef.current;
                    if (!openedToolSidebarForThisStreamRef.current && targetMessageId) {
                        setToolOutputsMessageId(targetMessageId);
                        setIsToolOutputsSidebarOpen(true);
                        openedToolSidebarForThisStreamRef.current = true;
                    }
                } else if (event.type === 'tool_done') {
                    console.log('Tool completed:', event.data);
                }
            },
            () => {
                // On complete - reload to get final state with all tool executions
                abortStreamRef.current = null;
                currentStreamChatIdRef.current = null;
                lastStreamUserMessageIdRef.current = null;
                openedToolSidebarForThisStreamRef.current = false;
                setSendingMessage(false);

                // Only reload if we're still on the same chat
                if (selectedChatId === requestChatId) {
                    // Small delay to ensure DB commits are complete
                    setTimeout(() => {
                        loadChatData(requestChatId);
                    }, 500);
                } else {
                    // Clean up temp messages if chat has switched
                    cleanupTempMessages();
                }
            },
            (err) => {
                // On error
                abortStreamRef.current = null;
                currentStreamChatIdRef.current = null;
                lastStreamUserMessageIdRef.current = null;
                openedToolSidebarForThisStreamRef.current = false;
                setError(err.message || 'Failed to send message');
                setSendingMessage(false);

                // Clean up temp messages on error
                cleanupTempMessages();
            }
        );

        // Store the abort function
        abortStreamRef.current = abortFn;
    };

    const handleQuestionClick = (question: string) => {
        if (!isSendingMessage) {
            handleSendMessage(question);
        }
    };

    const handleShowToolDetails = (messageId: string) => {
        setToolOutputsMessageId(messageId);
        setIsToolOutputsSidebarOpen(true);
    };

    const handleRenameChat = async () => {
        if (!selectedChatId || !currentChat) return;

        const newName = chatNameInput.trim();
        if (!newName || newName === currentChat.name) {
            setIsRenameModalOpen(false);
            return;
        }

        setIsRenamingChat(true);
        try {
            const updated = await updateChat(selectedChatId, { name: newName });
            updateChatInStore(selectedChatId, updated);
            setIsRenameModalOpen(false);
            setChatNameInput('');
        } catch (err) {
            const errorMsg = err instanceof Error ? err.message : 'Failed to rename chat';
            setError(errorMsg);
        } finally {
            setIsRenamingChat(false);
        }
    };

    const handleDeleteChat = async () => {
        if (!selectedChatId || !currentChat) return;

        if (!confirm(`Are you sure you want to delete "${currentChat.name}"? This will delete all messages and cannot be undone.`)) {
            return;
        }

        try {
            await deleteChat(selectedChatId);
            removeChat(currentChat.patientId, selectedChatId);
        } catch (err) {
            const errorMsg = err instanceof Error ? err.message : 'Failed to delete chat';
            setError(errorMsg);
        }
    };

    const openRenameModal = () => {
        if (currentChat) {
            setChatNameInput(currentChat.name);
            setIsRenameModalOpen(true);
        }
    };

    const openScanGallery = () => {
        setIsScanGalleryOpen(true);
    };

    const handleClearMemory = async () => {
        if (!selectedChatId || !currentChat) return;

        if (!confirm(`Clear conversation memory for "${currentChat.name}"? This will reset the AI's context but keep all messages.`)) {
            return;
        }

        try {
            const result = await clearChatMemory(selectedChatId);
            if (result.success) {
                const successMsg = 'Chat memory cleared successfully. The AI will start with fresh context.';
                setError(null);
                alert(successMsg);
            }
        } catch (err) {
            const errorMsg = err instanceof Error ? err.message : 'Failed to clear chat memory';
            setError(errorMsg);
        }
    };

    const openMemoryStats = async () => {
        if (!selectedChatId) return;

        setIsMemoryStatsModalOpen(true);
        setIsLoadingStats(true);
        setMemoryStats(null);

        try {
            const stats = await getChatMemoryStats(selectedChatId);
            setMemoryStats(stats);
        } catch (err) {
            const errorMsg = err instanceof Error ? err.message : 'Failed to load memory stats';
            setError(errorMsg);
        } finally {
            setIsLoadingStats(false);
        }
    };

    // No chat selected
    if (!selectedChatId) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center bg-zinc-950 p-8 text-center">
                <div className="max-w-md">
                    <h2 className="text-2xl font-semibold text-white mb-3">
                        Welcome to MedRAX
                    </h2>
                    <p className="text-zinc-400 mb-6">
                        Select a patient and chat from the sidebar to begin medical image analysis.
                    </p>
                    <div className="flex items-center justify-center space-x-6 text-sm text-zinc-500">
                        <div className="flex items-center space-x-2">
                            <FileImage className="h-5 w-5" />
                            <span>Upload Scans</span>
                        </div>
                        <div>
                            <span>Ask Questions</span>
                        </div>
                        <div>
                            <span>Get AI Analysis</span>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <>
            <div className="flex-1 flex flex-col bg-zinc-950 min-h-0">
                {/* Chat Header */}
                {currentChat && (
                    <div className="h-16 bg-zinc-900 border-b border-zinc-800 flex items-center justify-between px-6 flex-shrink-0">
                        <div>
                            <h2 className="text-lg font-semibold text-white">{currentChat.name}</h2>
                            <p className="text-xs text-zinc-500">
                                {currentChat.messageCount} messages · {currentChat.scanCount} scans
                            </p>
                        </div>

                        <Menu as="div" className="relative">
                            <Menu.Button className="p-2 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-md">
                                <MoreHorizontal className="h-5 w-5" />
                            </Menu.Button>
                            <Menu.Items className="absolute right-0 mt-1 w-48 bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl z-10">
                                <div className="py-1">
                                    <Menu.Item>
                                        {({ active }) => (
                                            <button
                                                onClick={openRenameModal}
                                                className={classNames(
                                                    'w-full text-left px-4 py-2 text-sm',
                                                    active ? 'bg-zinc-800 text-white' : 'text-zinc-300'
                                                )}
                                            >
                                                Rename Chat
                                            </button>
                                        )}
                                    </Menu.Item>
                                    <Menu.Item>
                                        {({ active }) => (
                                            <button
                                                onClick={openScanGallery}
                                                className={classNames(
                                                    'w-full text-left px-4 py-2 text-sm',
                                                    active ? 'bg-zinc-800 text-white' : 'text-zinc-300'
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
                                                    'w-full text-left px-4 py-2 text-sm',
                                                    active ? 'bg-zinc-800 text-white' : 'text-zinc-300'
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
                                                    'w-full text-left px-4 py-2 text-sm',
                                                    active ? 'bg-zinc-800 text-white' : 'text-zinc-300'
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
                                                    'w-full text-left px-4 py-2 text-sm',
                                                    active ? 'bg-zinc-800 text-red-400' : 'text-red-500'
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
                    <SuggestedQuestions
                        questions={suggestedQuestions}
                        onSelect={handleQuestionClick}
                    />

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
                    <div className="text-center py-8 text-zinc-500">
                        Failed to load memory stats
                    </div>
                )}
            </Modal>
        </>
    );
}
