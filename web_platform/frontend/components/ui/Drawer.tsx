/**
 * Drawer Component
 *
 * Slide-in panel from the side of the screen.
 */

"use client";

import React, { useEffect } from "react";
import { cn } from "../../lib/utils/helpers";

/**
 * Drawer Component Props
 * @property isOpen - Controls drawer visibility (required)
 * @property onClose - Callback when drawer should close (required)
 * @property title - Optional title displayed at top of drawer
 * @property children - Drawer content (required)
 * @property side - Which side drawer slides from (default: 'right')
 * @property size - Drawer width size (default: 'md')
 */
export interface DrawerProps {
  /** Controls drawer visibility */
  isOpen: boolean;
  /** Callback when drawer should close */
  onClose: () => void;
  /** Optional title displayed at top */
  title?: string;
  /** Drawer content (required) */
  children: React.ReactNode;
  /** Which side drawer slides from */
  side?: "left" | "right";
  /** Drawer width size */
  size?: "sm" | "md" | "lg";
}

export function Drawer({
  isOpen,
  onClose,
  title,
  children,
  side = "right",
  size = "md",
}: DrawerProps) {
  // Handle ESC key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
      document.body.style.overflow = "hidden";
    }

    return () => {
      document.removeEventListener("keydown", handleEscape);
      document.body.style.overflow = "unset";
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const sizeStyles = {
    sm: "w-80",
    md: "w-96",
    lg: "w-1/2",
  };

  const slideStyles = {
    left: "left-0",
    right: "right-0",
  };

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Overlay */}
      <div
        className="absolute inset-0 bg-black/40"
        onClick={onClose}
      />

      {/* Drawer */}
      <div
        className={cn(
          "absolute top-0 bottom-0 bg-zinc-900 shadow-2xl border flex flex-col",
          sizeStyles[size],
          slideStyles[side],
          side === "left" ? "border-r border-zinc-800" : "border-l border-zinc-800",
          "animate-in slide-in-from-right duration-300"
        )}
      >
        {/* Header */}
        {title && (
          <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
            <h2 className="text-xl font-semibold text-white">{title}</h2>
            <button
              onClick={onClose}
              className="p-1 hover:bg-zinc-800 rounded-lg transition-colors text-zinc-400 hover:text-white"
              aria-label="Close drawer"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">{children}</div>
      </div>
    </div>
  );
}
