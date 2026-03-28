/**
 * AuthGuard Component
 *
 * Protects routes that require authentication.
 * Redirects to login if not authenticated.
 */

"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "../../lib/store/authStore";
import { Spinner } from "../ui/Spinner";

/**
 * AuthGuard Component Props
 * @property children - Protected content to render when authenticated
 */
interface AuthGuardProps {
  /** Protected content to render when authenticated */
  children: React.ReactNode;
}

export function AuthGuard({ children }: AuthGuardProps) {
  const router = useRouter();
  const { isAuthenticated, isLoading, initialize } = useAuthStore();

  useEffect(() => {
    // Initialize auth state from localStorage
    initialize();
  }, [initialize]);

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!isLoading && !isAuthenticated) {
      router.push("/login");
    }
  }, [isLoading, isAuthenticated, router]);

  // Show loading spinner while checking auth
  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-zinc-950">
        <Spinner size="lg" />
      </div>
    );
  }

  // Don't render children if not authenticated
  if (!isAuthenticated) {
    return null;
  }

  return <>{children}</>;
}
