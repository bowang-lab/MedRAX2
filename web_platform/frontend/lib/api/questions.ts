/**
 * Suggested Questions API Functions
 * 
 * API calls for managing suggested questions.
 */

import apiClient from './client';
import { API_ENDPOINTS } from '../config/api';
import type { SuggestedQuestion } from '../types/question';

/**
 * Get all suggested questions for the current doctor
 */
export async function getQuestions(): Promise<SuggestedQuestion[]> {
    const response = await apiClient.get<SuggestedQuestion[]>(
        API_ENDPOINTS.QUESTIONS
    );
    return response.data;
}

/**
 * Create a new suggested question
 */
export async function createQuestion(data: {
    question: string;
}): Promise<SuggestedQuestion> {
    const response = await apiClient.post<SuggestedQuestion>(
        API_ENDPOINTS.QUESTIONS,
        data
    );
    return response.data;
}

/**
 * Delete a suggested question
 */
export async function deleteQuestion(id: string): Promise<void> {
    await apiClient.delete(API_ENDPOINTS.QUESTION_DETAIL(id));
}

