/**
 * SuggestedQuestions Component
 *
 * Floating chips above the chat input showing:
 * - Default predefined questions
 * - Doctor's custom frequent questions
 */

"use client";

import type { SuggestedQuestion } from "../../lib/types/question";

/**
 * SuggestedQuestions Component Props
 * @property questions - Array of suggested questions to display (required)
 * @property onSelect - Callback when user selects a question (required)
 * @property onManage - Optional callback to open questions management settings
 */
interface SuggestedQuestionsProps {
  /** Array of suggested questions to display */
  questions: SuggestedQuestion[];
  /** Callback when user selects a question */
  onSelect: (question: string) => void;
  /** Optional callback to open questions management settings */
  onManage?: () => void;
}

export function SuggestedQuestions({
  questions,
  onSelect,
  onManage,
}: SuggestedQuestionsProps) {
  if (!questions || questions.length === 0) return null;

  return (
    <div className="px-4 pb-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-zinc-400">Suggested Questions</span>
        {onManage && (
          <button
            onClick={onManage}
            className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
          >
            Manage
          </button>
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        {(questions || []).map((question) => (
          <button
            key={question.id}
            onClick={() => onSelect(question.question)}
            className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 hover:text-white rounded-full text-sm transition-colors border border-zinc-700 hover:border-zinc-600"
          >
            {question.question}
          </button>
        ))}
      </div>
    </div>
  );
}
