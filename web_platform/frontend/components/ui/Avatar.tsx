/**
 * Avatar Component
 *
 * User/patient avatar with initials fallback.
 */

"use client";

import React, { useState } from "react";
import { cn, getInitials } from "../../lib/utils/helpers";

/**
 * Avatar Component Props
 * @property name - Name to generate initials from (required, can be null for anonymous)
 * @property imageUrl - Optional image URL to display instead of initials
 * @property size - Avatar size (default: 'md')
 * @property className - Additional CSS classes
 */
export interface AvatarProps {
  /** Name to generate initials from (can be null for anonymous) */
  name: string | null;
  /** Optional image URL to display instead of initials */
  imageUrl?: string;
  /** Avatar size */
  size?: "sm" | "md" | "lg" | "xl";
  /** Additional CSS classes */
  className?: string;
}

export function Avatar({ name, imageUrl, size = "md", className }: AvatarProps) {
  const [imageError, setImageError] = useState(false);

  const sizeStyles = {
    sm: "w-8 h-8 text-xs",
    md: "w-10 h-10 text-sm",
    lg: "w-12 h-12 text-base",
    xl: "w-16 h-16 text-xl",
  };

  const initials = getInitials(name);

  // Show image if URL exists and no error, otherwise show initials
  const shouldShowImage = imageUrl && !imageError;

  return (
    <div
      className={cn(
        "relative inline-flex items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-blue-700 text-white font-semibold overflow-hidden",
        sizeStyles[size],
        className
      )}
    >
      {shouldShowImage ? (
        // eslint-disable-next-line @next/next/no-img-element -- Dynamic user-provided avatar URLs, not static assets
        <img
          src={imageUrl}
          alt={name || "Anonymous"}
          className="w-full h-full object-cover"
          onError={() => setImageError(true)}
        />
      ) : (
        <span>{initials}</span>
      )}
    </div>
  );
}
