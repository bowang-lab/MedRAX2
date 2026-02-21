/**
 * ChatListItem Component
 *
 * Individual chat item in the patient's chat list.
 */

"use client";

import { MessageCircle } from "lucide-react";
import type { Chat } from "../../lib/types/chat";
import { classNames } from "../../lib/utils";

/**
 * ChatListItem Component Props
 * @property chat - Chat data to display (required)
 * @property isSelected - Whether this chat is currently selected (required)
 * @property onClick - Callback when user clicks the chat item (required)
 */
interface ChatListItemProps {
  /** Chat data to display */
  chat: Chat;
  /** Whether this chat is currently selected */
  isSelected: boolean;
  /** Callback when user clicks the chat item */
  onClick: () => void;
}

export function ChatListItem({ chat, isSelected, onClick }: ChatListItemProps) {
  return (
    <button
      onClick={onClick}
      className={classNames(
        "w-full px-3 py-2 text-left rounded-md transition-colors",
        isSelected ? "bg-blue-600 text-white" : "text-zinc-300 hover:bg-zinc-800 hover:text-white"
      )}
    >
      <div className="flex items-start space-x-2">
        <MessageCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium truncate">{chat.name}</div>
          <div className={classNames("text-xs mt-0.5", isSelected ? "text-blue-200" : "text-zinc-500")}>
            {chat.messageCount} messages
            {chat.scanCount > 0 && ` · ${chat.scanCount} scans`}
          </div>
        </div>
      </div>
    </button>
  );
}
