/**
 * Textarea Component
 * 
 * Reusable multiline text input.
 */

'use client';

import React from 'react';
import { cn } from '../../lib/utils/helpers';

/**
 * Textarea Component Props
 * @property label - Optional label text displayed above the textarea
 * @property error - Error message to display below textarea (also styles textarea red)
 * @property helperText - Helper text shown below textarea when no error
 * @extends React.TextareaHTMLAttributes<HTMLTextAreaElement> - All standard textarea props
 */
export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
    /** Optional label text displayed above the textarea */
    label?: string;
    /** Error message to display (also styles textarea red) */
    error?: string;
    /** Helper text shown when no error */
    helperText?: string;
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
    ({ label, error, helperText, className, ...props }, ref) => {
        // Use React's useId for SSR-safe ID generation (matches server & client)
        const autoId = React.useId();
        const textareaId = props.id || autoId;

        return (
            <div className="w-full">
                {label && (
                    <label
                        htmlFor={textareaId}
                        className="block text-sm font-medium text-zinc-300 mb-1.5"
                    >
                        {label}
                    </label>
                )}

                <textarea
                    ref={ref}
                    id={textareaId}
                    className={cn(
                        'w-full px-4 py-2 bg-zinc-800 border rounded-lg text-white placeholder-zinc-500 transition-all duration-200',
                        'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
                        'disabled:opacity-50 disabled:cursor-not-allowed resize-none',
                        error && 'border-red-500 focus:ring-red-500',
                        !error && 'border-zinc-700 hover:border-zinc-600',
                        className
                    )}
                    {...props}
                />

                {error && (
                    <p className="mt-1.5 text-sm text-red-500">{error}</p>
                )}

                {helperText && !error && (
                    <p className="mt-1.5 text-sm text-zinc-500">{helperText}</p>
                )}
            </div>
        );
    }
);

Textarea.displayName = 'Textarea';

