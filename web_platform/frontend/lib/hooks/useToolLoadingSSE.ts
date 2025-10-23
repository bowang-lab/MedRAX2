/**
 * Tool Loading SSE Hook
 * 
 * Uses Server-Sent Events for real-time tool loading progress.
 * No polling needed - server pushes updates to client.
 */

import { useEffect, useRef, useState } from 'react';
import { API_CONFIG } from '../config/api';
import { AUTH_CONFIG } from '../config/app';

export interface ToolLoadingProgress {
    status: 'loading' | 'loaded' | 'error';
    progress: number;  // 0-100
    message: string;
    tool?: any;  // Tool data when loaded
}

export interface UseToolLoadingSSEOptions {
    toolId: string;
    onProgress?: (progress: ToolLoadingProgress) => void;
    onComplete?: (tool: any) => void;
    onError?: (error: string) => void;
}

/**
 * Hook for streaming tool loading progress via SSE
 * 
 * @example
 * ```tsx
 * const { start, stop, progress, isLoading, error } = useToolLoadingSSE({
 *   toolId: 'chexagent_vqa',
 *   onComplete: (tool) => console.log('Tool loaded:', tool),
 *   onError: (error) => console.error('Loading failed:', error)
 * });
 * 
 * // Start loading
 * start();
 * 
 * // Show progress
 * {isLoading && <Progress value={progress.progress} message={progress.message} />}
 * ```
 */
export function useToolLoadingSSE({
    toolId,
    onProgress,
    onComplete,
    onError,
}: UseToolLoadingSSEOptions) {
    const [progress, setProgress] = useState<ToolLoadingProgress>({
        status: 'loading',
        progress: 0,
        message: 'Initializing...',
    });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    const eventSourceRef = useRef<EventSource | null>(null);
    const isStartedRef = useRef(false);

    const stop = () => {
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        setIsLoading(false);
        isStartedRef.current = false;
    };

    const start = () => {
        if (isStartedRef.current) {
            console.warn('Tool loading already in progress');
            return;
        }

        isStartedRef.current = true;
        setIsLoading(true);
        setError(null);
        setProgress({
            status: 'loading',
            progress: 0,
            message: 'Connecting to server...',
        });

        // Get auth token from localStorage using correct key
        const token = localStorage.getItem(AUTH_CONFIG.tokenKey);
        if (!token) {
            const errorMsg = 'Authentication required - please log in again';
            setError(errorMsg);
            setIsLoading(false);
            onError?.(errorMsg);
            isStartedRef.current = false;
            return;
        }

        // Create SSE connection
        // Note: EventSource doesn't support custom headers, so we pass token as query param
        const url = `${API_CONFIG.BASE_URL}/api/tools/${toolId}/load-stream?token=${encodeURIComponent(token)}`;
        const eventSource = new EventSource(url);

        eventSource.onopen = () => {
            console.log('SSE connection opened for tool loading:', toolId);
        };

        eventSource.onmessage = (event) => {
            try {
                const data: ToolLoadingProgress = JSON.parse(event.data);
                
                setProgress(data);
                onProgress?.(data);

                if (data.status === 'loaded') {
                    console.log('Tool loaded successfully:', toolId);
                    onComplete?.(data.tool);
                    stop();
                } else if (data.status === 'error') {
                    console.error('Tool loading error:', data.message);
                    setError(data.message);
                    onError?.(data.message);
                    stop();
                }
            } catch (err) {
                console.error('Failed to parse SSE message:', err);
            }
        };

        eventSource.onerror = (err) => {
            console.error('SSE connection error:', err);
            const errorMsg = 'Connection error. Please try again.';
            setError(errorMsg);
            onError?.(errorMsg);
            stop();
        };

        eventSourceRef.current = eventSource;
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stop();
        };
    }, []);

    return {
        start,
        stop,
        progress,
        isLoading,
        error,
    };
}

