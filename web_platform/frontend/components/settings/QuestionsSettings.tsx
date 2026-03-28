/**
 * QuestionsSettings Component
 *
 * Manage suggested questions:
 * - View all questions (default + custom)
 * - Add custom questions
 * - Remove custom questions
 * - Reorder questions
 */

"use client";

import { useState, useEffect, useCallback } from "react";
import { Plus, X } from "lucide-react";
import { getQuestions, createQuestion, deleteQuestion } from "../../lib/api/questions";
import { Card } from "../ui/Card";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { Spinner } from "../ui/Spinner";
import { Badge } from "../ui/Badge";
import type { SuggestedQuestion } from "../../lib/types/question";

export function QuestionsSettings() {
  const [questions, setQuestions] = useState<SuggestedQuestion[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [newQuestion, setNewQuestion] = useState("");
  const [isAdding, setIsAdding] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);

  const loadQuestions = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const fetchedQuestions = await getQuestions();
      setQuestions(fetchedQuestions);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load questions");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadQuestions();
  }, [loadQuestions]);

  const handleAddQuestion = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!newQuestion.trim()) {
      setAddError("Question cannot be empty");
      return;
    }

    setIsAdding(true);
    setAddError(null);

    try {
      const question = await createQuestion({ question: newQuestion.trim() });
      setQuestions((prev) => [...prev, question]);
      setNewQuestion("");
    } catch (err) {
      setAddError(err instanceof Error ? err.message : "Failed to add question");
    } finally {
      setIsAdding(false);
    }
  };

  const handleDeleteQuestion = async (id: string) => {
    if (!confirm("Are you sure you want to delete this question?")) return;

    try {
      await deleteQuestion(id);
      setQuestions((prev) => prev.filter((q) => q.id !== id));
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete question");
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Spinner size="lg" />
      </div>
    );
  }

  if (error) {
    return <div className="text-red-400 text-sm text-center py-12">{error}</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">Suggested Questions</h2>
        <p className="text-sm text-zinc-400">
          Manage the questions that appear as quick options in the chat interface
        </p>
        <p className="text-xs text-zinc-500 mt-1">Reordering will be added once backend ordering is available.</p>
      </div>

      {/* Add New Question */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Add New Question</h3>
        <form onSubmit={handleAddQuestion} className="flex items-start space-x-3">
          <div className="flex-1">
            <Input
              value={newQuestion}
              onChange={(e) => setNewQuestion(e.target.value)}
              placeholder="e.g., Is there any sign of infection?"
              disabled={isAdding}
              error={addError || undefined}
            />
          </div>
          <Button type="submit" variant="primary" isLoading={isAdding} disabled={isAdding || !newQuestion.trim()}>
            <Plus className="h-4 w-4 mr-1" />
            Add
          </Button>
        </form>
      </Card>

      {/* Questions List */}
      <div className="space-y-3">
        {(questions || []).map((question) => (
          <Card key={question.id} className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3 flex-1">
                <span className="text-sm text-zinc-300 flex-1">{question.question}</span>
                {question.isDefault && <Badge variant="default">Default</Badge>}
              </div>

              {!question.isDefault && (
                <button
                  onClick={() => handleDeleteQuestion(question.id)}
                  className="ml-4 p-2 text-zinc-400 hover:text-red-400 transition-colors"
                  title="Delete question"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
          </Card>
        ))}
      </div>

      {(!questions || questions.length === 0) && (
        <div className="text-center py-12 text-zinc-500">
          No questions added yet. Add your first question above!
        </div>
      )}
    </div>
  );
}
