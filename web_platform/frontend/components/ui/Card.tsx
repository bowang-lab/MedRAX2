/**
 * Card Component
 *
 * Container with border and background.
 */

"use client";

import React from "react";
import { cn } from "../../lib/utils/helpers";

/**
 * Card Component Props
 * @property children - Card content (required)
 * @property className - Additional CSS classes
 * @property padding - Internal padding size (default: 'md')
 * @property hoverable - Apply hover effect (default: false)
 * @property onClick - Optional click handler
 */
export interface CardProps {
  /** Card content (required) */
  children: React.ReactNode;
  /** Additional CSS classes */
  className?: string;
  /** Internal padding size */
  padding?: "none" | "sm" | "md" | "lg";
  /** Apply hover effect */
  hoverable?: boolean;
  /** Optional click handler */
  onClick?: () => void;
}

export function Card({
  children,
  className,
  padding = "md",
  hoverable = false,
  onClick,
}: CardProps) {
  const paddingStyles = {
    none: "",
    sm: "p-3",
    md: "p-4",
    lg: "p-6",
  };

  return (
    <div
      className={cn(
        "bg-zinc-900 border border-zinc-800 rounded-lg",
        paddingStyles[padding],
        hoverable && "transition-all duration-200 hover:border-zinc-700 hover:shadow-lg",
        onClick && "cursor-pointer",
        className
      )}
      onClick={onClick}
    >
      {children}
    </div>
  );
}
