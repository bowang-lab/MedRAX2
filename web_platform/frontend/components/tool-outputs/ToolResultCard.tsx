/**
 * ToolResultCard Component
 * 
 * Displays tool-specific results based on the tool type.
 * Supports:
 * - Classification results (pathology predictions, probabilities)
 * - Segmentation results (mask images, metrics)
 * - VQA results (text answers)
 * - Report generation (structured reports)
 * - Grounding results (bounding boxes, visualizations)
 */

'use client';

import { useState } from 'react';
import type { ToolExecutionResult } from '../../lib/types/tool';
import { getImageUrl } from '../../lib/utils/image';

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

export function ToolResultCard({ toolName, result }: ToolResultCardProps) {
    const [failedImages, setFailedImages] = useState<Set<string>>(new Set());
    const data = result.resultData;

    const handleImageError = (imagePath: string) => {
        setFailedImages(prev => new Set(prev).add(imagePath));
    };

    // Extract image paths from result data (generated images, visualizations, masks)
    const imagePaths: string[] = [];
    if (data && typeof data === 'object') {
        Object.entries(data).forEach(([key, value]) => {
            if (
                (key.toLowerCase().includes('image_path') ||
                    key.toLowerCase().includes('visualization') ||
                    key.toLowerCase().includes('mask') ||
                    key.toLowerCase().includes('output_image') ||
                    key.toLowerCase().includes('grounding_image') ||
                    key.toLowerCase().includes('segmentation_image')) &&
                typeof value === 'string' &&
                value &&
                !value.includes('input')  // Exclude input image paths
            ) {
                imagePaths.push(value);
            }
        });
    }

    // Tool-specific rendering helpers
    const isClassificationTool = toolName.includes('torchxrayvision') ||
        toolName.includes('arcplus') ||
        toolName.includes('classifier');

    const isSegmentationTool = toolName.includes('medsam') ||
        toolName.includes('segmentation');

    const isVQATool = toolName.includes('chexagent') ||
        toolName.includes('llava') ||
        toolName.includes('medgemma') ||
        toolName.includes('vqa');

    const isReportTool = toolName.includes('report');

    const isGroundingTool = toolName.includes('grounding');

    // Render classification results in a formatted way
    const renderClassificationResults = () => {
        if (!isClassificationTool || !data) return null;

        const predictions = data.predictions || data.pathology_predictions || data.results;
        if (!predictions || typeof predictions !== 'object') return null;

        return (
            <div className="space-y-2">
                <h4 className="text-sm font-medium text-zinc-300">Pathology Predictions:</h4>
                <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 space-y-2">
                    {Object.entries(predictions).map(([pathology, probability]) => {
                        const prob = typeof probability === 'number' ? probability : 0;
                        const percentage = (prob * 100).toFixed(1);
                        const isHighProbability = prob > 0.5;

                        return (
                            <div key={pathology} className="flex items-center justify-between">
                                <span className={`text-sm ${isHighProbability ? 'text-yellow-400 font-medium' : 'text-zinc-400'}`}>
                                    {pathology.replace(/_/g, ' ')}
                                </span>
                                <div className="flex items-center gap-2">
                                    <div className="w-32 h-2 bg-zinc-800 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full ${isHighProbability ? 'bg-yellow-500' : 'bg-blue-500'}`}
                                            style={{ width: `${percentage}%` }}
                                        />
                                    </div>
                                    <span className="text-xs text-zinc-500 w-12 text-right">{percentage}%</span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>
        );
    };

    // Render VQA answer prominently
    const renderVQAAnswer = () => {
        if (!isVQATool || !data) return null;

        const answer = data.answer || data.response || data.text;
        if (!answer || typeof answer !== 'string') return null;

        return (
            <div className="space-y-2">
                <h4 className="text-sm font-medium text-zinc-300">Answer:</h4>
                <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3">
                    <p className="text-sm text-zinc-100 leading-relaxed">{answer}</p>
                </div>
            </div>
        );
    };

    // Render report sections
    const renderReport = () => {
        if (!isReportTool || !data) return null;

        const findings = data.findings || data.Findings;
        const impression = data.impression || data.Impression;

        if (!findings && !impression) return null;

        return (
            <div className="space-y-3">
                {findings && (
                    <div>
                        <h4 className="text-sm font-medium text-zinc-300 mb-2">Findings:</h4>
                        <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3">
                            <p className="text-sm text-zinc-100 leading-relaxed whitespace-pre-wrap">{findings}</p>
                        </div>
                    </div>
                )}
                {impression && (
                    <div>
                        <h4 className="text-sm font-medium text-zinc-300 mb-2">Impression:</h4>
                        <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3">
                            <p className="text-sm text-zinc-100 leading-relaxed whitespace-pre-wrap">{impression}</p>
                        </div>
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="space-y-4">
            {/* Tool-specific formatted results */}
            {renderClassificationResults()}
            {renderVQAAnswer()}
            {renderReport()}

            {/* Generated Images / Visualizations */}
            {imagePaths.length > 0 && (
                <div className="space-y-2">
                    <h4 className="text-sm font-medium text-zinc-300">
                        {isSegmentationTool ? 'Segmentation Masks:' :
                            isGroundingTool ? 'Grounding Visualizations:' :
                                'Generated Images:'}
                    </h4>
                    <div className="flex flex-wrap gap-3">
                        {imagePaths.map((imagePath, idx) => {
                            const imageUrl = getImageUrl(imagePath);
                            const hasFailed = failedImages.has(imageUrl);

                            return (
                                <div key={idx} className="relative group">
                                    {hasFailed ? (
                                        <div className="h-48 w-48 flex items-center justify-center bg-red-900/20 border border-red-800 rounded-lg text-red-400 text-xs p-4 text-center">
                                            ⚠️ Failed to load result image
                                        </div>
                                    ) : (
                                        <>
                                            {/* eslint-disable-next-line @next/next/no-img-element */}
                                            <img
                                                src={imageUrl}
                                                alt={`Tool output ${idx + 1}`}
                                                className="h-48 w-auto max-w-full object-contain rounded-lg border border-zinc-700 bg-zinc-900 hover:border-blue-500 transition-colors cursor-zoom-in"
                                                onError={() => handleImageError(imageUrl)}
                                            />
                                            <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-opacity rounded-lg pointer-events-none" />
                                        </>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Raw Result Data (collapsible for debugging) */}
            {data && (
                <details className="space-y-2">
                    <summary className="text-sm font-medium text-zinc-400 cursor-pointer hover:text-zinc-300">
                        Raw Data (click to expand)
                    </summary>
                    <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 max-h-96 overflow-y-auto">
                        <pre className="text-xs text-zinc-400 whitespace-pre-wrap overflow-x-auto">
                            {JSON.stringify(data, null, 2)}
                        </pre>
                    </div>
                </details>
            )}
        </div>
    );
}
