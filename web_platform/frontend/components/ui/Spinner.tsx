/**
 * Spinner Component
 *
 * Loading spinner indicator.
 */

"use client";

import React from "react";
import { cn } from "../../lib/utils/helpers";

/**
 * Spinner Component Props
 * @property size - Spinner size (default: 'md')
 * @property className - Additional CSS classes
 */
export interface SpinnerProps {
  /** Spinner size */
  size?: "sm" | "md" | "lg";
  /** Additional CSS classes */
  className?: string;
}

export function Spinner({ size = "md", className }: SpinnerProps) {
  const sizeStyles = {
    sm: "w-4 h-4",
    md: "w-8 h-8",
    lg: "w-12 h-12",
  };

  return (
    <svg
      className={cn("animate-spin text-blue-500", sizeStyles[size], className)}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}
