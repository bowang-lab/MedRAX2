/**
 * ToolResultCard Component
 * 
 * Displays tool-specific results based on the tool type.
 * Different visualizations for:
 * - Classification
 * - Segmentation
 * - Report Generation
 * - VQA
 * - Phrase Grounding
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

export function ToolResultCard({ toolName, result }: ToolResultCardProps) {
    const data = result.resultData;

    // Classification Result
    if (toolName.includes('classification') || toolName.includes('classifier')) {
        return (
            <div className="p-4 bg-zinc-800 rounded-lg">
                <h4 className="text-sm font-semibold text-white mb-3">Classification Results</h4>
                {data.predictions && Array.isArray(data.predictions) ? (
                    <div className="space-y-2">
                        {/* Tool-specific result format - structure varies by tool */}
                        {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                        {data.predictions.map((pred: any, idx: number) => (
                            <div key={idx} className="flex items-center justify-between">
                                <span className="text-sm text-zinc-300">{pred.label || pred.class}</span>
                                <div className="flex items-center space-x-2">
                                    <div className="w-24 h-2 bg-zinc-700 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-blue-500"
                                            style={{ width: `${(pred.probability || pred.score) * 100}%` }}
                                        />
                                    </div>
                                    <span className="text-xs text-zinc-400 w-12 text-right">
                                        {((pred.probability || pred.score) * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <pre className="text-xs text-zinc-400 whitespace-pre-wrap">
                        {JSON.stringify(data, null, 2)}
                    </pre>
                )}
            </div>
        );
    }

    // Segmentation Result
    if (toolName.includes('segmentation') || toolName.includes('segment')) {
        return (
            <div className="p-4 bg-zinc-800 rounded-lg">
                <h4 className="text-sm font-semibold text-white mb-3">Segmentation Results</h4>
                {data.mask_image_url && (
                    // eslint-disable-next-line @next/next/no-img-element -- Dynamic tool-generated medical images, not static assets
                    <img
                        src={data.mask_image_url}
                        alt="Segmentation mask"
                        className="w-full rounded-lg mb-2"
                    />
                )}
                {data.regions && (
                    <div className="text-xs text-zinc-400">
                        <p>Regions detected: {data.regions.length}</p>
                    </div>
                )}
                {!data.mask_image_url && !data.regions && (
                    <pre className="text-xs text-zinc-400 whitespace-pre-wrap">
                        {JSON.stringify(data, null, 2)}
                    </pre>
                )}
            </div>
        );
    }

    // Report Generation Result
    if (toolName.includes('report') || toolName.includes('generation')) {
        return (
            <div className="p-4 bg-zinc-800 rounded-lg">
                <h4 className="text-sm font-semibold text-white mb-3">Generated Report</h4>
                {data.report || data.text ? (
                    <p className="text-sm text-zinc-300 whitespace-pre-wrap">
                        {data.report || data.text}
                    </p>
                ) : (
                    <pre className="text-xs text-zinc-400 whitespace-pre-wrap">
                        {JSON.stringify(data, null, 2)}
                    </pre>
                )}
            </div>
        );
    }

    // VQA Result
    if (toolName.includes('vqa') || toolName.includes('question')) {
        return (
            <div className="p-4 bg-zinc-800 rounded-lg">
                <h4 className="text-sm font-semibold text-white mb-3">Answer</h4>
                {data.answer ? (
                    <p className="text-sm text-zinc-300">{data.answer}</p>
                ) : (
                    <pre className="text-xs text-zinc-400 whitespace-pre-wrap">
                        {JSON.stringify(data, null, 2)}
                    </pre>
                )}
                {data.confidence && (
                    <p className="text-xs text-zinc-500 mt-2">
                        Confidence: {(data.confidence * 100).toFixed(1)}%
                    </p>
                )}
            </div>
        );
    }

    // Grounding Result
    if (toolName.includes('grounding') || toolName.includes('phrase')) {
        return (
            <div className="p-4 bg-zinc-800 rounded-lg">
                <h4 className="text-sm font-semibold text-white mb-3">Phrase Grounding</h4>
                {data.bounding_boxes && Array.isArray(data.bounding_boxes) ? (
                    <div className="space-y-2">
                        {/* Tool-specific result format - structure varies by tool */}
                        {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                        {(data.bounding_boxes || []).map((box: any, idx: number) => (
                            <div key={idx} className="text-xs text-zinc-300 p-2 bg-zinc-900 rounded">
                                <p>Phrase: {box.phrase || box.text}</p>
                                <p className="text-zinc-500">
                                    Box: [{box.x}, {box.y}, {box.width}, {box.height}]
                                </p>
                            </div>
                        ))}
                    </div>
                ) : (
                    <pre className="text-xs text-zinc-400 whitespace-pre-wrap">
                        {JSON.stringify(data, null, 2)}
                    </pre>
                )}
            </div>
        );
    }

    // Generic Result (fallback)
    return (
        <div className="p-4 bg-zinc-800 rounded-lg">
            <h4 className="text-sm font-semibold text-white mb-3">Tool Output</h4>
            <pre className="text-xs text-zinc-400 whitespace-pre-wrap overflow-x-auto">
                {JSON.stringify(data, null, 2)}
            </pre>
        </div>
    );
}

