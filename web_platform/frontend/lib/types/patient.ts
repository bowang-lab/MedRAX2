/**
 * Patient Types
 *
 * Patients are managed by doctors.
 * Name is optional - null means anonymous patient.
 *
 * PatientWithStats is used for list operations where we need
 * aggregate counts (chats, scans) from database joins.
 * Base Patient is used for simple CRUD operations.
 */

/**
 * Base patient structure
 * Used in single patient operations
 */
export interface Patient {
  id: string;
  doctorId: string;
  name: string | null; // null = anonymous patient
  createdAt: string;
  lastActivityAt: string | null;
}

/**
 * Patient with aggregate statistics
 * Used in list operations with database joins
 * This is the primary type used in the frontend
 */
export interface PatientWithStats extends Patient {
  chatCount: number;
  scanCount: number;
  lastActivityAt: string | null;
}
