/**
 * RenamePatientModal Component
 *
 * Modal for renaming a patient.
 */

"use client";

import { useState, useEffect } from "react";
import { Modal } from "../ui/Modal";
import { Input } from "../ui/Input";
import { Button } from "../ui/Button";
import type { Patient } from "../../lib/types/patient";

/**
 * RenamePatientModal Component Props
 * @property isOpen - Controls modal visibility (required)
 * @property patient - Patient to rename (null = no patient selected)
 * @property onClose - Callback when modal should close (required)
 * @property onSubmit - Callback when user submits new name (required, async)
 */
interface RenamePatientModalProps {
  /** Controls modal visibility */
  isOpen: boolean;
  /** Patient to rename (null = no patient selected) */
  patient: Patient | null;
  /** Callback when modal should close */
  onClose: () => void;
  /** Callback when user submits new name (async) */
  onSubmit: (patientId: string, name: string | null) => Promise<void>;
}

export function RenamePatientModal({ isOpen, patient, onClose, onSubmit }: RenamePatientModalProps) {
  const [name, setName] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setName(patient?.name || "");
    } else {
      setName("");
    }
  }, [isOpen, patient]);

  const handleSubmit = async () => {
    if (!patient) return;

    setIsSubmitting(true);
    try {
      const patientName = name.trim() || null;
      await onSubmit(patient.id, patientName);
      onClose();
    } catch (error) {
      console.error("Failed to rename patient:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Rename Patient" size="sm">
      <div className="space-y-4">
        <Input
          label="Patient Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Leave blank for anonymous"
          autoFocus
          helperText="Leave blank to make patient anonymous."
        />

        <div className="flex items-center justify-end space-x-3">
          <Button variant="ghost" onClick={onClose} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleSubmit} isLoading={isSubmitting} disabled={isSubmitting}>
            Rename
          </Button>
        </div>
      </div>
    </Modal>
  );
}
