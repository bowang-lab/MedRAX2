/**
 * Badge Component
 *
 * Small label or tag.
 */

"use client";

import React from "react";
import { cn } from "../../lib/utils/helpers";

/**
 * Badge Component Props
 * @property children - Badge content (required)
 * @property variant - Color/semantic variant (default: 'default')
 * @property size - Badge size (default: 'md')
 * @property className - Additional CSS classes
 */
export interface BadgeProps {
  /** Badge content (required) */
  children: React.ReactNode;
  /** Color/semantic variant */
  variant?: "default" | "success" | "warning" | "error" | "info";
  /** Badge size */
  size?: "sm" | "md";
  /** Additional CSS classes */
  className?: string;
}

export function Badge({
  children,
  variant = "default",
  size = "md",
  className,
}: BadgeProps) {
  const variantStyles = {
    default: "bg-zinc-800 text-zinc-300",
    success: "bg-green-900 text-green-300",
    warning: "bg-yellow-900 text-yellow-300",
    error: "bg-red-900 text-red-300",
    info: "bg-blue-900 text-blue-300",
  };

  const sizeStyles = {
    sm: "px-2 py-0.5 text-xs",
    md: "px-2.5 py-1 text-sm",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center font-medium rounded-full",
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
    >
      {children}
    </span>
  );
}
