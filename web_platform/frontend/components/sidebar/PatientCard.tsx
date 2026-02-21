/**
 * PatientCard Component
 *
 * Collapsible patient card with:
 * - Patient name (or "Anonymous")
 * - Stats (chats, scans, last activity)
 * - Expand/collapse for chat list
 * - Chat list when expanded
 */

"use client";

import { useEffect, useCallback, useMemo } from "react";
import { ChevronDown, ChevronRight, User, Plus, MoreHorizontal } from "lucide-react";
import { Menu } from "@headlessui/react";
import type { PatientWithStats } from "../../lib/types/patient";
import { ChatListItem } from "./ChatListItem";
import { classNames } from "../../lib/utils";
import { useAppStore } from "../../lib/store/appStore";
import { getChats, createChat } from "../../lib/api/chats";
import { Spinner } from "../ui/Spinner";

/**
 * PatientCard Component Props
 * @property patient - Patient data with stats (chats, scans, last activity) (required)
 * @property isExpanded - Whether patient's chat list is expanded (required)
 * @property selectedChatId - Currently selected chat ID (null = none selected)
 * @property onToggle - Callback when user clicks to expand/collapse (required)
 * @property onChatSelect - Callback when user selects a chat (required)
 * @property onRenamePatient - Callback when user wants to rename patient (required)
 * @property onDeletePatient - Callback when user wants to delete patient (required)
 */
interface PatientCardProps {
  /** Patient data with stats (chats, scans, last activity) */
  patient: PatientWithStats;
  /** Whether patient's chat list is expanded */
  isExpanded: boolean;
  /** Currently selected chat ID (null = none selected) */
  selectedChatId: string | null;
  /** Callback when user clicks to expand/collapse */
  onToggle: () => void;
  /** Callback when user selects a chat */
  onChatSelect: (chatId: string) => void;
  /** Callback when user wants to rename patient */
  onRenamePatient: () => void;
  /** Callback when user wants to delete patient */
  onDeletePatient: () => void;
}

export function PatientCard({
  patient,
  isExpanded,
  selectedChatId,
  onToggle,
  onChatSelect,
  onRenamePatient,
  onDeletePatient,
}: PatientCardProps) {
  const { chats, setChats, addChat, isLoadingChats, setLoadingChats, setChatsError, updatePatient } = useAppStore();

  const patientChats = useMemo(() => chats[patient.id] || [], [chats, patient.id]);
  const hasLoadedChats = chats[patient.id] !== undefined;
  const isLoading = isLoadingChats[patient.id] || false;

  // Load chats when expanded
  const loadChats = useCallback(async () => {
    setLoadingChats(patient.id, true);
    try {
      const fetchedChats = await getChats(patient.id);
      setChats(patient.id, fetchedChats);
    } catch (error) {
      const errorMessage = (error as { message?: string })?.message || "Failed to load chats";
      console.error("Failed to load chats:", errorMessage);
      setChatsError(patient.id, errorMessage);
    } finally {
      setLoadingChats(patient.id, false);
    }
  }, [patient.id, setChats, setLoadingChats, setChatsError]);

  useEffect(() => {
    if (isExpanded && !hasLoadedChats && !isLoading) {
      loadChats();
    }
  }, [isExpanded, hasLoadedChats, isLoading, loadChats]);

  const handleNewChat = async () => {
    try {
      const newChat = await createChat(patient.id, {});
      addChat(patient.id, newChat);
      updatePatient(patient.id, {
        chatCount: (patient.chatCount || 0) + 1,
      });
      onChatSelect(newChat.id);
    } catch (error) {
      const errorMessage = (error as { message?: string })?.message || "Unknown error";
      console.error("Failed to create chat:", errorMessage);
      alert(`Failed to create chat: ${errorMessage}`);
    }
  };

  const displayName = patient.name || "Anonymous Patient";

  return (
    <div className="border-b border-zinc-800">
      {/* Patient Header */}
      <div className="flex items-center justify-between p-3 hover:bg-zinc-800/50">
        <button onClick={onToggle} className="flex-1 flex items-center space-x-2 text-left">
          {isExpanded ? (
            <ChevronDown className="h-4 w-4 text-zinc-400 flex-shrink-0" />
          ) : (
            <ChevronRight className="h-4 w-4 text-zinc-400 flex-shrink-0" />
          )}
          <User className="h-4 w-4 text-zinc-400 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-white truncate">{displayName}</div>
            <div className="text-xs text-zinc-500 flex items-center gap-2">
              <span>
                {patient.chatCount || 0} {(patient.chatCount || 0) === 1 ? "chat" : "chats"}
              </span>
              {(patient.scanCount || 0) > 0 && (
                <>
                  <span>·</span>
                  <span>
                    {patient.scanCount} {patient.scanCount === 1 ? "scan" : "scans"}
                  </span>
                </>
              )}
            </div>
          </div>
        </button>

        {/* Actions Menu */}
        <Menu as="div" className="relative">
          <Menu.Button className="p-1 text-zinc-400 hover:text-white hover:bg-zinc-700 rounded">
            <MoreHorizontal className="h-4 w-4" />
          </Menu.Button>
          <Menu.Items className="absolute right-0 mt-1 w-48 bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl z-10">
            <div className="py-1">
              <Menu.Item>
                {({ active }) => (
                  <button
                    onClick={handleNewChat}
                    className={classNames(
                      "w-full text-left px-4 py-2 text-sm",
                      active ? "bg-zinc-800 text-white" : "text-zinc-300"
                    )}
                  >
                    <Plus className="inline h-4 w-4 mr-2" />
                    New Chat
                  </button>
                )}
              </Menu.Item>
              <Menu.Item>
                {({ active }) => (
                  <button
                    onClick={onRenamePatient}
                    className={classNames(
                      "w-full text-left px-4 py-2 text-sm",
                      active ? "bg-zinc-800 text-white" : "text-zinc-300"
                    )}
                  >
                    Rename Patient
                  </button>
                )}
              </Menu.Item>
              <Menu.Item>
                {({ active }) => (
                  <button
                    onClick={onDeletePatient}
                    className={classNames(
                      "w-full text-left px-4 py-2 text-sm",
                      active ? "bg-zinc-800 text-red-400" : "text-red-500"
                    )}
                  >
                    Delete Patient
                  </button>
                )}
              </Menu.Item>
            </div>
          </Menu.Items>
        </Menu>
      </div>

      {/* Chat List (when expanded) */}
      {isExpanded && (
        <div className="pb-2 px-2 space-y-1">
          {isLoading ? (
            <div className="flex items-center justify-center py-4">
              <Spinner size="sm" />
            </div>
          ) : patientChats.length > 0 ? (
            patientChats.map((chat) => (
              <ChatListItem
                key={chat.id}
                chat={chat}
                isSelected={selectedChatId === chat.id}
                onClick={() => onChatSelect(chat.id)}
              />
            ))
          ) : (
            <div className="text-xs text-zinc-500 text-center py-2">No chats yet</div>
          )}
        </div>
      )}
    </div>
  );
}
