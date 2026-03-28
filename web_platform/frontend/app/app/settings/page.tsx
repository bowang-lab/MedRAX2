/**
 * Settings Page
 *
 * Main settings interface with tabs for:
 * - Profile (name, password)
 * - Tools Management (load/unload)
 * - Suggested Questions
 */

"use client";

import { useState } from "react";
import { ArrowLeft } from "lucide-react";
import { useRouter } from "next/navigation";
import { AuthGuard } from "../../../components/auth/AuthGuard";
import { ProfileSettings } from "../../../components/settings/ProfileSettings";
import { ToolsSettings } from "../../../components/settings/ToolsSettings";
import { QuestionsSettings } from "../../../components/settings/QuestionsSettings";
import { classNames } from "../../../lib/utils";

export default function SettingsPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<"profile" | "tools" | "questions">(
    "profile"
  );

  const tabs = [
    { id: "profile" as const, label: "Profile" },
    { id: "tools" as const, label: "Tools Management" },
    { id: "questions" as const, label: "Suggested Questions" },
  ];

  return (
    <AuthGuard>
      <div className="min-h-screen bg-zinc-950 flex flex-col">
        {/* Header */}
        <div className="h-16 bg-zinc-900 border-b border-zinc-800 flex items-center px-6">
          <button
            onClick={() => router.push("/app")}
            className="flex items-center space-x-2 text-zinc-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="h-5 w-5" />
            <span className="text-sm font-medium">Back to App</span>
          </button>
          <div className="ml-6 text-lg font-semibold text-white">Settings</div>
        </div>

        {/* Content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Sidebar Tabs */}
          <aside className="w-64 bg-zinc-900 border-r border-zinc-800 p-4">
            <nav className="space-y-1">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={classNames(
                    "w-full text-left px-4 py-2 rounded-md text-sm font-medium transition-colors",
                    activeTab === tab.id
                      ? "bg-blue-600 text-white"
                      : "text-zinc-300 hover:bg-zinc-800 hover:text-white"
                  )}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </aside>

          {/* Main Content */}
          <main className="flex-1 overflow-y-auto p-8">
            <div className="max-w-3xl mx-auto">
              {activeTab === "profile" && <ProfileSettings />}
              {activeTab === "tools" && <ToolsSettings />}
              {activeTab === "questions" && <QuestionsSettings />}
            </div>
          </main>
        </div>
      </div>
    </AuthGuard>
  );
}
