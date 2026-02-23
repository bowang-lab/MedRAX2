/**
 * Button Component
 *
 * Reusable button with multiple variants and sizes.
 */

"use client";

import React from "react";
import { cn } from "../../lib/utils/helpers";

/**
 * Button Component Props
 * @property variant - Visual style variant (default: 'primary')
 * @property size - Button size (default: 'md')
 * @property isLoading - Shows loading spinner and disables interaction
 * @property leftIcon - Optional icon on the left side
 * @property rightIcon - Optional icon on the right side
 * @property children - Button content (required)
 * @extends React.ButtonHTMLAttributes<HTMLButtonElement> - All standard button props
 */
export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  /** Visual style variant */
  variant?: "primary" | "secondary" | "outline" | "ghost" | "danger";
  /** Button size */
  size?: "sm" | "md" | "lg";
  /** Shows loading spinner and disables interaction */
  isLoading?: boolean;
  /** Optional icon on the left side */
  leftIcon?: React.ReactNode;
  /** Optional icon on the right side */
  rightIcon?: React.ReactNode;
  /** Button content (required) */
  children: React.ReactNode;
}

export function Button({
  variant = "primary",
  size = "md",
  isLoading = false,
  leftIcon,
  rightIcon,
  children,
  className,
  disabled,
  ...props
}: ButtonProps) {
  const baseStyles =
    "inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed";

  const variantStyles = {
    primary: "bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500",
    secondary: "bg-zinc-700 hover:bg-zinc-600 text-white focus:ring-zinc-500",
    outline: "border-2 border-zinc-700 hover:bg-zinc-800 text-white focus:ring-zinc-500",
    ghost: "hover:bg-zinc-800 text-zinc-300 hover:text-white focus:ring-zinc-500",
    danger: "bg-red-600 hover:bg-red-700 text-white focus:ring-red-500",
  };

  const sizeStyles = {
    sm: "px-3 py-1.5 text-sm gap-1.5",
    md: "px-4 py-2 text-base gap-2",
    lg: "px-6 py-3 text-lg gap-2.5",
  };

  return (
    <button
      className={cn(baseStyles, variantStyles[variant], sizeStyles[size], className)}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <>
          <svg
            className="animate-spin h-4 w-4"
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
          <span>Loading...</span>
        </>
      ) : (
        <>
          {leftIcon && <span>{leftIcon}</span>}
          <span>{children}</span>
          {rightIcon && <span>{rightIcon}</span>}
        </>
      )}
    </button>
  );
}
