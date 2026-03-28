/**
 * Suggested Questions API Functions
 *
 * API calls for managing suggested questions.
 */

import { openapiClient, authHeaders } from "../openapi/client";
import type { SuggestedQuestion } from "../types/question";
import type { ApiQuestionResponse, ApiQuestionCreate } from "../types/api";
import { toUiQuestion } from "../openapi/transformers";

/**
 * Get all suggested questions for the current doctor
 * Backend always returns List[QuestionResponse] (never null)
 */
export async function getQuestions(): Promise<SuggestedQuestion[]> {
  const { data, error } = await openapiClient.GET("/api/questions", {
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!data) throw new Error("No data returned from server");
  return data.map((q: ApiQuestionResponse) => toUiQuestion(q));
}

/**
 * Create a new suggested question
 */
export async function createQuestion(data: { question: string }): Promise<SuggestedQuestion> {
  const requestBody: ApiQuestionCreate = {
    question: data.question,
    display_order: 0, // Default order, backend will handle proper ordering
  };

  const { data: response, error } = await openapiClient.POST("/api/questions", {
    body: requestBody,
    headers: authHeaders(),
  });
  if (error) throw error;
  if (!response) throw new Error("Failed to create question");
  return toUiQuestion(response);
}

/**
 * Delete a suggested question
 */
export async function deleteQuestion(id: string): Promise<void> {
  const { error } = await openapiClient.DELETE("/api/questions/{question_id}", {
    params: { path: { question_id: id } },
    headers: authHeaders(),
  });
  if (error) throw error;
}
