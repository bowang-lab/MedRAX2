/**
 * Input Component
 *
 * Reusable text input with label and error handling.
 */

"use client";

import React from "react";
import { cn } from "../../lib/utils/helpers";

/**
 * Input Component Props
 * @property label - Optional label text displayed above the input
 * @property error - Error message to display below input (also styles input red)
 * @property helperText - Helper text shown below input when no error
 * @property leftIcon - Optional icon/element shown on the left inside input
 * @property rightIcon - Optional icon/element shown on the right inside input
 * @extends React.InputHTMLAttributes<HTMLInputElement> - All standard input props
 */
export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  /** Optional label text displayed above the input */
  label?: string;
  /** Error message to display (also styles input red) */
  error?: string;
  /** Helper text shown when no error */
  helperText?: string;
  /** Optional icon/element on the left inside input */
  leftIcon?: React.ReactNode;
  /** Optional icon/element on the right inside input */
  rightIcon?: React.ReactNode;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, helperText, leftIcon, rightIcon, className, ...props }, ref) => {
    // Use React's useId for SSR-safe ID generation (matches server & client)
    const autoId = React.useId();
    const inputId = props.id || autoId;

    return (
      <div className="w-full">
        {label && (
          <label htmlFor={inputId} className="block text-sm font-medium text-zinc-300 mb-1.5">
            {label}
          </label>
        )}

        <div className="relative">
          {leftIcon && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500">
              {leftIcon}
            </div>
          )}

          <input
            ref={ref}
            id={inputId}
            className={cn(
              "w-full px-4 py-2 bg-zinc-800 border rounded-lg text-white placeholder-zinc-500 transition-all duration-200",
              "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              error ? "border-red-500 focus:ring-red-500" : "border-zinc-700 hover:border-zinc-600",
              leftIcon ? "pl-10" : "",
              rightIcon ? "pr-10" : "",
              className
            )}
            {...props}
          />

          {rightIcon && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500">
              {rightIcon}
            </div>
          )}
        </div>

        {error && <p className="mt-1.5 text-sm text-red-500">{error}</p>}

        {helperText && !error && (
          <p className="mt-1.5 text-sm text-zinc-500">{helperText}</p>
        )}
      </div>
    );
  }
);

Input.displayName = "Input";
