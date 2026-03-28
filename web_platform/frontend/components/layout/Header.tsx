/**
 * Header Component
 *
 * Top navigation bar with:
 * - App logo/title
 * - Profile dropdown (doctor name, logout)
 * - Settings button
 */

"use client";

import { useState, Fragment } from "react";
import { useRouter } from "next/navigation";
import { Menu, Transition } from "@headlessui/react";
import { Settings, LogOut, User, ChevronDown } from "lucide-react";
import { useAuthStore } from "../../lib/store/authStore";
import { logoutDoctor } from "../../lib/api/auth";
import { Avatar } from "../ui/Avatar";
import { classNames } from "../../lib/utils";

export function Header() {
  const router = useRouter();
  const { doctor, clearAuth } = useAuthStore();
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  const handleLogout = async () => {
    setIsLoggingOut(true);
    try {
      await logoutDoctor();
    } catch (error) {
      console.error("Logout error:", error);
    } finally {
      clearAuth();
      router.push("/login");
    }
  };

  const handleSettings = () => {
    router.push("/app/settings");
  };

  return (
    <header className="h-16 bg-zinc-900 border-b border-zinc-800 flex items-center justify-between px-6 flex-shrink-0">
      {/* Left: Logo/Title */}
      <div className="flex items-center space-x-3">
        <div className="text-2xl font-bold text-blue-500">MedRAX</div>
        <div className="text-sm text-zinc-500">Medical Reasoning Agent</div>
      </div>

      {/* Right: Profile + Settings */}
      <div className="flex items-center space-x-4">
        {/* Settings Button */}
        <button
          onClick={handleSettings}
          className="p-2 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-md transition-colors"
          title="Settings"
        >
          <Settings className="h-5 w-5" />
        </button>

        {/* Profile Dropdown */}
        <Menu as="div" className="relative">
          <Menu.Button className="flex items-center space-x-2 px-3 py-2 text-zinc-300 hover:text-white hover:bg-zinc-800 rounded-md transition-colors">
            <Avatar name={doctor?.name || null} size="sm" />
            <span className="text-sm font-medium">{doctor?.name || "User"}</span>
            <ChevronDown className="h-4 w-4" />
          </Menu.Button>

          <Transition
            as={Fragment}
            enter="transition ease-out duration-100"
            enterFrom="transform opacity-0 scale-95"
            enterTo="transform opacity-100 scale-100"
            leave="transition ease-in duration-75"
            leaveFrom="transform opacity-100 scale-100"
            leaveTo="transform opacity-0 scale-95"
          >
            <Menu.Items className="absolute right-0 mt-2 w-56 origin-top-right bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl focus:outline-none z-50">
              <div className="py-1">
                <Menu.Item>
                  {({ active }) => (
                    <button
                      onClick={handleSettings}
                      className={classNames(
                        "w-full flex items-center px-4 py-2 text-sm",
                        active ? "bg-zinc-800 text-white" : "text-zinc-300"
                      )}
                    >
                      <User className="mr-3 h-4 w-4" />
                      Profile Settings
                    </button>
                  )}
                </Menu.Item>

                <div className="border-t border-zinc-800 my-1"></div>

                <Menu.Item>
                  {({ active }) => (
                    <button
                      onClick={handleLogout}
                      disabled={isLoggingOut}
                      className={classNames(
                        "w-full flex items-center px-4 py-2 text-sm",
                        active ? "bg-zinc-800 text-red-400" : "text-red-500",
                        isLoggingOut && "opacity-50 cursor-not-allowed"
                      )}
                    >
                      <LogOut className="mr-3 h-4 w-4" />
                      {isLoggingOut ? "Logging out..." : "Logout"}
                    </button>
                  )}
                </Menu.Item>
              </div>
            </Menu.Items>
          </Transition>
        </Menu>
      </div>
    </header>
  );
}
