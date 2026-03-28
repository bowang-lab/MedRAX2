/**
 * Root Page
 *
 * Redirects to /app (main application).
 */

"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "../lib/store/authStore";
import { Spinner } from "../components/ui/Spinner";

export default function RootPage() {
  const router = useRouter();
  const { isAuthenticated, isLoading, initialize } = useAuthStore();

  useEffect(() => {
    initialize();
  }, [initialize]);

  useEffect(() => {
    if (!isLoading) {
      if (isAuthenticated) {
        router.push("/app");
      } else {
        router.push("/login");
      }
    }
  }, [isLoading, isAuthenticated, router]);

  return (
    <div className="h-screen w-screen flex items-center justify-center bg-zinc-950">
      <Spinner size="lg" />
    </div>
  );
}
