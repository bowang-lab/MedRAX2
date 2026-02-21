/**
 * NewPatientModal Component
 *
 * Modal for creating a new patient.
 * Name is optional (can be anonymous).
 */

"use client";

import { useState, useEffect } from "react";
import { Modal } from "../ui/Modal";
import { Input } from "../ui/Input";
import { Button } from "../ui/Button";

/**
 * NewPatientModal Component Props
 * @property isOpen - Controls modal visibility (required)
 * @property onClose - Callback when modal should close (required)
 * @property onSubmit - Callback when user submits new patient (required, async)
 */
interface NewPatientModalProps {
  /** Controls modal visibility */
  isOpen: boolean;
  /** Callback when modal should close */
  onClose: () => void;
  /** Callback when user submits new patient (async) */
  onSubmit: (name: string | null) => Promise<void>;
}

export function NewPatientModal({ isOpen, onClose, onSubmit }: NewPatientModalProps) {
  const [name, setName] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!isOpen) {
      setName("");
    }
  }, [isOpen]);

  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      const patientName = name.trim() || null;
      await onSubmit(patientName);
      setName("");
      onClose();
    } catch (error) {
      console.error("Failed to create patient:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="New Patient" size="sm">
      <div className="space-y-4">
        <Input
          label="Patient Name (Optional)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Leave blank for anonymous"
          autoFocus
          helperText="Patient name is optional. Leave blank to create an anonymous patient."
        />

        <div className="flex items-center justify-end space-x-3">
          <Button variant="ghost" onClick={onClose} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleSubmit} isLoading={isSubmitting} disabled={isSubmitting}>
            Create Patient
          </Button>
        </div>
      </div>
    </Modal>
  );
}
