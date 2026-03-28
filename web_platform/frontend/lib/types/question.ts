/**
 * Suggested Question Types
 *
 * Doctors can have personalized suggested questions.
 * Default system questions + doctor's custom questions.
 * Shared across all chats and patients for that doctor.
 */

export interface SuggestedQuestion {
  id: string;
  doctorId: string | null; // null for default system questions
  question: string;
  isDefault: boolean; // System default vs user-added
  displayOrder: number;
  createdAt: string;
}

export interface QuestionCreate {
  question: string;
  displayOrder?: number;
}

export interface QuestionUpdate {
  question?: string;
  displayOrder?: number;
}

// Default questions to seed database
export const DEFAULT_QUESTIONS = ["Is there pneumonia?", "Measure heart size"];
