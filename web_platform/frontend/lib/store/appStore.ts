/**
 * App Store
 *
 * Zustand store for main application state:
 * - Patients, chats, messages
 * - UI state (selected patient, chat, etc.)
 * - Loading states
 */

import { create } from "zustand";
import type { PatientWithStats } from "../types/patient";
import type { Chat } from "../types/chat";
import type { MessageWithDetails } from "../types/message";
import type { Scan } from "../types/scan";

interface AppState {
  // Data
  patients: PatientWithStats[];
  chats: Record<string, Chat[]>; // patientId -> Chat[]
  messages: Record<string, MessageWithDetails[]>; // chatId -> MessageWithDetails[]
  scans: Record<string, Scan[]>; // chatId -> Scan[]

  // UI State
  selectedChatId: string | null;
  expandedPatientIds: string[]; // IDs of patients with expanded chat lists
  isSidebarCollapsed: boolean;
  isToolPanelOpen: boolean;
  selectedToolExecutionId: string | null;

  // Loading states
  isLoadingPatients: boolean;
  isLoadingChats: Record<string, boolean>; // patientId -> loading
  isLoadingMessages: Record<string, boolean>; // chatId -> loading
  isSendingMessage: boolean;

  // Error states
  patientsError: string | null;
  chatsError: Record<string, string | null>; // patientId -> error
  messagesError: Record<string, string | null>; // chatId -> error
  messageError: string | null;

  // Actions: Data
  setPatients: (patients: PatientWithStats[]) => void;
  addPatient: (patient: PatientWithStats) => void;
  updatePatient: (id: string, patient: Partial<PatientWithStats>) => void;
  removePatient: (id: string) => void;

  setChats: (patientId: string, chats: Chat[]) => void;
  addChat: (patientId: string, chat: Chat) => void;
  updateChat: (chatId: string, chat: Partial<Chat>) => void;
  removeChat: (patientId: string, chatId: string) => void;

  setMessages: (chatId: string, messages: MessageWithDetails[]) => void;
  addMessage: (chatId: string, message: MessageWithDetails) => void;

  setScans: (chatId: string, scans: Scan[]) => void;
  addScan: (chatId: string, scan: Scan) => void;

  // Actions: UI
  selectChat: (chatId: string | null) => void;
  togglePatientExpanded: (patientId: string) => void;
  toggleSidebar: () => void;
  openToolPanel: (executionId: string) => void;
  closeToolPanel: () => void;

  // Actions: Loading
  setLoadingPatients: (loading: boolean) => void;
  setLoadingChats: (patientId: string, loading: boolean) => void;
  setLoadingMessages: (chatId: string, loading: boolean) => void;
  setSendingMessage: (sending: boolean) => void;

  // Actions: Errors
  setPatientsError: (error: string | null) => void;
  setChatsError: (patientId: string, error: string | null) => void;
  setMessagesError: (chatId: string, error: string | null) => void;
  setMessageError: (error: string | null) => void;
}

export const useAppStore = create<AppState>()((set) => ({
  // Initial Data
  patients: [],
  chats: {},
  messages: {},
  scans: {},

  // Initial UI State
  selectedChatId: null,
  expandedPatientIds: [],
  isSidebarCollapsed: false,
  isToolPanelOpen: false,
  selectedToolExecutionId: null,

  // Initial Loading
  isLoadingPatients: false,
  isLoadingChats: {},
  isLoadingMessages: {},
  isSendingMessage: false,

  // Initial Errors
  patientsError: null,
  chatsError: {},
  messagesError: {},
  messageError: null,

  // Data Actions
  setPatients: (patients) => set({ patients }),
  addPatient: (patient) =>
    set((state) => ({ patients: [...(state.patients || []), patient] })),
  updatePatient: (id, update) =>
    set((state) => ({
      patients: (state.patients || []).map((p) =>
        p.id === id ? { ...p, ...update } : p
      ),
    })),
  removePatient: (id) =>
    set((state) => {
      const { [id]: removedChats, ...restChats } = state.chats;
      const { [id]: removedLoading, ...restLoadingChats } = state.isLoadingChats;
      const { [id]: removedError, ...restChatsError } = state.chatsError;
      void removedChats;
      void removedLoading;
      void removedError;

      // Also clear messages and scans for all chats of this patient
      const restMessages = { ...state.messages };
      const restScans = { ...state.scans };
      const patientChatIds = state.chats[id]?.map((c) => c.id) || [];
      patientChatIds.forEach((chatId) => {
        delete restMessages[chatId];
        delete restScans[chatId];
      });

      const shouldClearSelected = patientChatIds.includes(state.selectedChatId ?? "");

      return {
        patients: (state.patients || []).filter((p) => p.id !== id),
        chats: restChats,
        isLoadingChats: restLoadingChats,
        chatsError: restChatsError,
        messages: restMessages,
        scans: restScans,
        selectedChatId: shouldClearSelected ? null : state.selectedChatId,
      };
    }),

  setChats: (patientId, chats) =>
    set((state) => ({ chats: { ...state.chats, [patientId]: chats } })),
  addChat: (patientId, chat) =>
    set((state) => ({
      chats: {
        ...state.chats,
        [patientId]: [...(state.chats[patientId] || []), chat],
      },
    })),
  updateChat: (chatId, update) =>
    set((state) => {
      const newChats = { ...state.chats };
      Object.keys(newChats).forEach((patientId) => {
        newChats[patientId] = (newChats[patientId] || []).map((c) =>
          c.id === chatId ? { ...c, ...update } : c
        );
      });
      return { chats: newChats };
    }),
  removeChat: (patientId, chatId) =>
    set((state) => {
      const { [chatId]: removedMessages, ...restMessages } = state.messages;
      const { [chatId]: removedScans, ...restScans } = state.scans;
      void removedMessages;
      void removedScans;
      return {
        chats: {
          ...state.chats,
          [patientId]: state.chats[patientId]?.filter((c) => c.id !== chatId) || [],
        },
        messages: restMessages,
        scans: restScans,
        // CRITICAL: Clear selectedChatId if we're deleting the selected chat
        selectedChatId: state.selectedChatId === chatId ? null : state.selectedChatId,
      };
    }),

  setMessages: (chatId, messages) =>
    set((state) => ({ messages: { ...state.messages, [chatId]: messages } })),
  addMessage: (chatId, message) =>
    set((state) => ({
      messages: {
        ...state.messages,
        [chatId]: [...(state.messages[chatId] || []), message],
      },
    })),

  setScans: (chatId, scans) =>
    set((state) => ({ scans: { ...state.scans, [chatId]: scans } })),
  addScan: (chatId, scan) =>
    set((state) => ({
      scans: {
        ...state.scans,
        [chatId]: [...(state.scans[chatId] || []), scan],
      },
    })),

  // UI Actions
  selectChat: (chatId) => set({ selectedChatId: chatId }),
  togglePatientExpanded: (patientId) =>
    set((state) => ({
      expandedPatientIds: (state.expandedPatientIds || []).includes(patientId)
        ? (state.expandedPatientIds || []).filter((id) => id !== patientId)
        : [...(state.expandedPatientIds || []), patientId],
    })),
  toggleSidebar: () =>
    set((state) => ({ isSidebarCollapsed: !state.isSidebarCollapsed })),
  openToolPanel: (executionId) =>
    set({ isToolPanelOpen: true, selectedToolExecutionId: executionId }),
  closeToolPanel: () => set({ isToolPanelOpen: false, selectedToolExecutionId: null }),

  // Loading Actions
  setLoadingPatients: (loading) => set({ isLoadingPatients: loading }),
  setLoadingChats: (patientId, loading) =>
    set((state) => ({
      isLoadingChats: { ...state.isLoadingChats, [patientId]: loading },
    })),
  setLoadingMessages: (chatId, loading) =>
    set((state) => ({
      isLoadingMessages: { ...state.isLoadingMessages, [chatId]: loading },
    })),
  setSendingMessage: (sending) => set({ isSendingMessage: sending }),

  // Error Actions
  setPatientsError: (error) => set({ patientsError: error }),
  setChatsError: (patientId, error) =>
    set((state) => ({
      chatsError: { ...state.chatsError, [patientId]: error },
    })),
  setMessagesError: (chatId, error) =>
    set((state) => ({
      messagesError: { ...state.messagesError, [chatId]: error },
    })),
  setMessageError: (error) => set({ messageError: error }),
}));
