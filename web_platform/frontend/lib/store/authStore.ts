/**
 * Auth Store
 *
 * Zustand store for authentication state management.
 * Handles login, logout, and persisting auth data.
 */

import { create } from "zustand";
import { Doctor } from "../types/doctor";
import { AUTH_CONFIG } from "../config/app";

interface AuthState {
  doctor: Doctor | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  // Actions
  setAuth: (doctor: Doctor, token: string) => void;
  clearAuth: () => void;
  setLoading: (loading: boolean) => void;
  initialize: () => void;
}

export const useAuthStore = create<AuthState>()((set) => ({
  doctor: null,
  token: null,
  isAuthenticated: false,
  isLoading: true,

  setAuth: (doctor, token) => {
    // Save to localStorage
    localStorage.setItem(AUTH_CONFIG.tokenKey, token);
    localStorage.setItem(AUTH_CONFIG.doctorKey, JSON.stringify(doctor));

    set({
      doctor,
      token,
      isAuthenticated: true,
      isLoading: false,
    });
  },

  clearAuth: () => {
    // Clear localStorage
    localStorage.removeItem(AUTH_CONFIG.tokenKey);
    localStorage.removeItem(AUTH_CONFIG.doctorKey);

    set({
      doctor: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
    });
  },

  setLoading: (loading) => set({ isLoading: loading }),

  initialize: () => {
    // Try to restore auth from localStorage
    const token = localStorage.getItem(AUTH_CONFIG.tokenKey);
    const doctorData = localStorage.getItem(AUTH_CONFIG.doctorKey);

    if (token && doctorData) {
      try {
        const doctor = JSON.parse(doctorData);
        set({
          doctor,
          token,
          isAuthenticated: true,
          isLoading: false,
        });
      } catch {
        // Invalid data, clear it
        localStorage.removeItem(AUTH_CONFIG.tokenKey);
        localStorage.removeItem(AUTH_CONFIG.doctorKey);
        set({ isLoading: false });
      }
    } else {
      set({ isLoading: false });
    }
  },
}));
