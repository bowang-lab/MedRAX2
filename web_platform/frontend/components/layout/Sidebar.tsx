/**
 * Sidebar Component
 *
 * Left sidebar containing:
 * - Search bar
 * - New Patient button
 * - List of patients (collapsible) with their chats
 */

"use client";

import { useState, useEffect, useCallback } from "react";
import { Search, Plus } from "lucide-react";
import { useAppStore } from "../../lib/store/appStore";
import {
  getPatients,
  createPatient,
  updatePatient,
  deletePatient,
} from "../../lib/api/patients";
import { getChats } from "../../lib/api/chats";
import { PatientCard } from "../sidebar/PatientCard";
import { NewPatientModal } from "../sidebar/NewPatientModal";
import { RenamePatientModal } from "../sidebar/RenamePatientModal";
import { Spinner } from "../ui/Spinner";
import { Button } from "../ui/Button";
import type { Patient } from "../../lib/types/patient";

export function Sidebar() {
  const {
    patients,
    setPatients,
    addPatient,
    setChats,
    updatePatient: updatePatientInStore,
    removePatient,
    selectedChatId,
    expandedPatientIds,
    togglePatientExpanded,
    selectChat,
    isLoadingPatients,
    setLoadingPatients,
    setPatientsError,
  } = useAppStore();

  const [searchQuery, setSearchQuery] = useState("");
  const [isNewPatientModalOpen, setIsNewPatientModalOpen] = useState(false);
  const [isRenameModalOpen, setIsRenameModalOpen] = useState(false);
  const [patientToRename, setPatientToRename] = useState<Patient | null>(null);

  const loadPatients = useCallback(async () => {
    setLoadingPatients(true);
    try {
      const fetchedPatients = await getPatients();
      setPatients(fetchedPatients);
    } catch (error) {
      const errorMessage =
        (error as { message?: string })?.message || "Failed to load patients";
      console.error("Failed to load patients:", errorMessage);
      setPatientsError(errorMessage);
    } finally {
      setLoadingPatients(false);
    }
  }, [setLoadingPatients, setPatients, setPatientsError]);

  // Load patients on mount
  useEffect(() => {
    loadPatients();
  }, [loadPatients]);

  const handleCreatePatient = async (name: string | null) => {
    const newPatient = await createPatient({ name });
    addPatient(newPatient);

    if (!expandedPatientIds.includes(newPatient.id)) {
      togglePatientExpanded(newPatient.id);
    }

    try {
      // Backend already creates the initial chat; fetch chats and select it
      const chats = await getChats(newPatient.id);
      setChats(newPatient.id, chats);
      if (chats && chats.length > 0) {
        selectChat(chats[0].id);
      }
    } catch (error) {
      console.error("Failed to load initial consultation chat:", error);
    }
  };

  const handleRenamePatient = async (patientId: string, name: string | null) => {
    const updatedPatient = await updatePatient(patientId, { name });
    updatePatientInStore(patientId, updatedPatient);
  };

  const handleDeletePatient = async (patientId: string) => {
    if (!confirm("Are you sure you want to delete this patient and all their chats?")) {
      return;
    }

    try {
      await deletePatient(patientId);
      removePatient(patientId);
    } catch (error) {
      const errorMessage =
        (error as { message?: string })?.message || "Unknown error";
      console.error("Failed to delete patient:", errorMessage);
      alert(`Failed to delete patient: ${errorMessage}`);
    }
  };

  // Filter patients by search query
  const filteredPatients = (patients || []).filter((patient) => {
    if (!searchQuery) return true;
    const name = patient.name || "Anonymous Patient";
    return name.toLowerCase().includes(searchQuery.toLowerCase());
  });

  return (
    <aside className="w-80 bg-zinc-900 border-r border-zinc-800 flex flex-col">
      {/* Header with Search and New Patient Button */}
      <div className="p-4 space-y-3 border-b border-zinc-800">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold text-white">Patients</h2>
          <Button
            size="sm"
            variant="primary"
            onClick={() => setIsNewPatientModalOpen(true)}
            className="flex items-center space-x-1.5 px-3 py-1.5"
          >
            <Plus className="h-4 w-4" />
            <span className="text-sm font-medium">New Patient</span>
          </Button>
        </div>

        {/* Search Bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-zinc-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search patients..."
            className="w-full pl-9 pr-3 py-2 bg-zinc-800 border border-zinc-700 rounded-md text-sm text-white placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Patient List */}
      <div className="flex-1 overflow-y-auto">
        {isLoadingPatients ? (
          <div className="flex items-center justify-center py-8">
            <Spinner size="md" />
          </div>
        ) : filteredPatients.length > 0 ? (
          filteredPatients.map((patient) => (
            <PatientCard
              key={patient.id}
              patient={patient}
              isExpanded={expandedPatientIds.includes(patient.id)}
              selectedChatId={selectedChatId}
              onToggle={() => togglePatientExpanded(patient.id)}
              onChatSelect={(chatId) => selectChat(chatId)}
              onRenamePatient={() => {
                setPatientToRename(patient);
                setIsRenameModalOpen(true);
              }}
              onDeletePatient={() => handleDeletePatient(patient.id)}
            />
          ))
        ) : (
          <div className="flex flex-col items-center justify-center py-12 px-6 text-center">
            <p className="text-zinc-500 text-sm mb-3">
              {searchQuery ? "No patients found" : "No patients yet"}
            </p>
            {!searchQuery && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => setIsNewPatientModalOpen(true)}
              >
                Create First Patient
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Modals */}
      <NewPatientModal
        isOpen={isNewPatientModalOpen}
        onClose={() => setIsNewPatientModalOpen(false)}
        onSubmit={handleCreatePatient}
      />

      <RenamePatientModal
        isOpen={isRenameModalOpen}
        patient={patientToRename}
        onClose={() => {
          setIsRenameModalOpen(false);
          setPatientToRename(null);
        }}
        onSubmit={handleRenamePatient}
      />
    </aside>
  );
}
