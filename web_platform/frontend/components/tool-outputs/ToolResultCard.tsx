/**
 * ToolResultCard Component
 * 
 * Displays tool-specific results based on the tool type.
 * Supports generic JSON display and extracts generated images.
 */

'use client';

import type { ToolExecutionResult } from '../../lib/types/tool';

/**
 * ToolResultCard Component Props
 * @property toolName - Name of the tool that produced this result (required)
 * @property result - The execution result data to display (required)
 */
interface ToolResultCardProps {
    /** Name of the tool that produced this result */
    toolName: string;
    /** The execution result data to display */
    result: ToolExecutionResult;
}

export function ToolResultCard({ result }: ToolResultCardProps) {
    const data = result.resultData;

    // Extract image paths from result data
    const imagePaths: string[] = [];
    if (data && typeof data === 'object') {
        Object.entries(data).forEach(([key, value]) => {
            if (
                (key.toLowerCase().includes('image_path') ||
                    key.toLowerCase().includes('visualization') ||
                    key.toLowerCase().includes('mask_image')) &&
                typeof value === 'string' &&
                value
            ) {
                imagePaths.push(value);
            }
        });
    }

    return (
        <div className="space-y-4">
            {/* Generated Images */}
            {imagePaths.length > 0 && (
                <div className="space-y-2">
                    <h4 className="text-sm font-medium text-zinc-300">Generated Images:</h4>
                    <div className="flex flex-wrap gap-3">
                        {imagePaths.map((imagePath, idx) => (
                            <div key={idx} className="relative group">
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                    src={imagePath}
                                    alt={`Tool output ${idx + 1}`}
                                    className="h-48 w-auto max-w-full object-contain rounded-lg border border-zinc-700 bg-zinc-900 hover:border-blue-500 transition-colors cursor-zoom-in"
                                />
                                <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-opacity rounded-lg pointer-events-none" />
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Result Data */}
            <div className="space-y-2">
                <h4 className="text-sm font-medium text-zinc-300">Result Data:</h4>
                <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 max-h-96 overflow-y-auto">
                    <pre className="text-xs text-zinc-400 whitespace-pre-wrap overflow-x-auto">
                        {JSON.stringify(data, null, 2)}
                    </pre>
                </div>
            </div>

        </div>
    );
}
