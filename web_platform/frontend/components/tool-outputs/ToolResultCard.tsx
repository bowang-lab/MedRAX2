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
    const [showRawData, setShowRawData] = useState(false);

    // Parse result data if it's stringified JSON in 'raw' field
    let data = result.resultData;
    if (data && typeof data === 'object' && 'raw' in data && typeof data.raw === 'string') {
        try {
            const parsed = JSON.parse(data.raw);
            // If parsed is an array, use the first item (which typically has the main results)
            // Backend often returns [actual_results, metadata], we want the first item
            data = Array.isArray(parsed) ? parsed[0] : parsed;
        } catch {
            // Keep original data if parsing fails
        }
    }

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

    const isSearchTool = toolName.includes('duckduckgo') ||
        toolName.includes('web_browser') ||
        toolName.includes('search');

    // Optional metadata for richer summaries
    const metadata: any = result.resultMetadata || null;

    // Render classification results in a formatted way
    const renderClassificationResults = () => {
        if (!isClassificationTool || !data) return null;

        // Classification tools return the predictions directly as the data object
        // Format: { "Atelectasis": 0.123, "Cardiomegaly": 0.456, ... }
        const predictions = data.predictions || data.pathology_predictions || data;

        // Check if this looks like classification data (object with numeric values)
        if (!predictions || typeof predictions !== 'object') return null;

        // Filter to only numeric probability values
        const pathologies = Object.entries(predictions).filter(([key, value]) =>
            typeof value === 'number' && !key.includes('_') && key !== 'error' && key !== 'image_path'
        );

        if (pathologies.length === 0) return null;

        return (
            <div className="space-y-2">
                <h4 className="text-sm font-medium text-zinc-300">Pathology Predictions:</h4>
                <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 space-y-2">
                    {pathologies.map(([pathology, probability]) => {
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

        const findingsRaw = data.findings || data.Findings;
        const impressionRaw = data.impression || data.Impression;

        // Convert to strings safely
        const findings = typeof findingsRaw === 'string' ? findingsRaw : null;
        const impression = typeof impressionRaw === 'string' ? impressionRaw : null;

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

    // Render search results (DuckDuckGo, Web Browser)
    const renderSearchResults = () => {
        if (!isSearchTool || !data) return null;

        // Handle error state
        if (data.error || data.error_details) {
            const errorMsg = data.error || data.error_details;
            const queryText = typeof data.query === 'string' ? data.query : '';
            return (
                <div className="bg-red-900/20 border border-red-800 rounded-lg p-3">
                    <h4 className="text-sm font-medium text-red-400 mb-2">❌ Search Failed</h4>
                    <p className="text-sm text-red-300">{typeof errorMsg === 'string' ? errorMsg : 'An error occurred during search'}</p>
                    {queryText && <p className="text-xs text-red-400 mt-2">Query: {queryText}</p>}
                </div>
            );
        }

        // Extract results array
        const results = data.results || [];
        if (!Array.isArray(results) || results.length === 0) {
            const queryText = typeof data.query === 'string' ? data.query : '';
            return (
                <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3">
                    <p className="text-sm text-zinc-400">No search results found.</p>
                    {queryText && <p className="text-xs text-zinc-500 mt-2">Query: {queryText}</p>}
                </div>
            );
        }

        const messageText = typeof data.message === 'string' ? data.message : '';
        return (
            <div className="space-y-3">
                <h4 className="text-sm font-medium text-zinc-300">
                    🔍 Search Results {messageText && <span className="text-zinc-500 font-normal text-xs">({messageText})</span>}
                </h4>
                <div className="space-y-2">
                    {results.map((result: { title?: string; link?: string; url?: string; snippet?: string; source?: string }, idx: number) => (
                        <div key={idx} className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 hover:border-blue-600 transition-colors">
                            <a
                                href={result.link || result.url || '#'}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="block space-y-1"
                            >
                                <h5 className="text-sm font-medium text-blue-400 hover:text-blue-300">
                                    {result.title || 'Untitled'}
                                </h5>
                                {result.snippet && (
                                    <p className="text-xs text-zinc-400 leading-relaxed">
                                        {result.snippet}
                                    </p>
                                )}
                                {(result.link || result.url) && (
                                    <p className="text-xs text-zinc-600 truncate">
                                        {result.source || (result.link || result.url ? new URL(result.link || result.url || '').hostname : '')}
                                    </p>
                                )}
                            </a>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    return (
        <div className="space-y-4">
            {/* Tool-specific formatted results */}
            {renderClassificationResults()}
            {renderVQAAnswer()}
            {renderReport()}
            {renderSearchResults()}

            {/* Generated Images / Visualizations */}
            {imagePaths.length > 0 && (
                <div className="space-y-2">
                    <h4 className="text-sm font-medium text-zinc-300">
                        {isSegmentationTool ? 'Segmentation Masks:' :
                            isGroundingTool ? 'Grounding Visualizations:' :
                                'Generated Images:'}
                    </h4>
                    {/* Segmentation metadata summary if available */}
                    {isSegmentationTool && metadata && (
                        <div className="text-xs text-zinc-400 space-y-1">
                            {Array.isArray(metadata.requested_organs) && metadata.requested_organs.length > 0 && (
                                <div>Requested: <span className="text-zinc-300">{metadata.requested_organs.join(', ')}</span></div>
                            )}
                            {Array.isArray(metadata.processed_organs) && (
                                <div>Detected: <span className="text-zinc-300">{metadata.processed_organs.length > 0 ? metadata.processed_organs.join(', ') : 'none'}</span></div>
                            )}
                            {typeof metadata.pixel_spacing_mm === 'number' && (
                                <div>Pixel spacing: <span className="text-zinc-300">{metadata.pixel_spacing_mm} mm</span></div>
                            )}
                            {typeof metadata.threshold_used === 'number' && (
                                <div>Threshold: <span className="text-zinc-300">{metadata.threshold_used}</span></div>
                            )}
                        </div>
                    )}
                    <div className="flex flex-wrap gap-3">
                        {imagePaths.map((imagePath, idx) => {
                            const imageUrl = getImageUrl(imagePath);
                            const hasFailed = !imageUrl || failedImages.has(imageUrl);

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
                                            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-opacity rounded-lg pointer-events-none" />
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
                <div className="space-y-2">
                    <button
                        onClick={() => setShowRawData(!showRawData)}
                        className="text-xs font-medium text-zinc-500 hover:text-zinc-300 underline decoration-dotted"
                    >
                        {showRawData ? '▼ Hide Raw Data' : '▶ Show Raw Data (for debugging)'}
                    </button>
                    {/* Segmentation-specific summary (metrics or mask summary) */}
                    {isSegmentationTool && (
                        <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 space-y-2">
                            {'metrics' in (data as any) && typeof (data as any).metrics === 'object' ? (
                                <>
                                    <h4 className="text-sm font-medium text-zinc-300">Segmentation Metrics:</h4>
                                    {Object.keys((data as any).metrics).length > 0 ? (
                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                            {Object.entries((data as any).metrics).map(([organ, metrics]) => {
                                                const m = metrics as Record<string, unknown>;
                                                const areaCm2 = typeof m.area_cm2 === 'number' ? m.area_cm2.toFixed(2) : undefined;
                                                const conf = typeof m.confidence_score === 'number' ? (m.confidence_score * 100).toFixed(1) : undefined;
                                                const mean = typeof m.mean_intensity === 'number' ? m.mean_intensity.toFixed(1) : undefined;
                                                return (
                                                    <div key={organ} className="rounded-md border border-zinc-700 p-2">
                                                        <div className="text-xs text-zinc-400 mb-1">{organ}</div>
                                                        <div className="text-xs text-zinc-300 space-x-2">
                                                            {areaCm2 && <span>Area: {areaCm2} cm²</span>}
                                                            {conf && <span>Conf: {conf}%</span>}
                                                            {mean && <span>Mean Intensity: {mean}</span>}
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    ) : (
                                        <div className="space-y-2">
                                            <div className="text-xs text-zinc-500">
                                                No organ masks detected at the current threshold. The overlay image above may look unchanged.
                                            </div>
                                            <div className="text-xs p-2 bg-blue-500/10 border border-blue-500/20 rounded">
                                                <span className="font-semibold text-blue-400">💡 Recommendation:</span>
                                                <span className="text-zinc-300"> Use MedSAM2 instead - it's more robust and works with a wider variety of X-ray images. MedSAM2 uses advanced segmentation that doesn't rely on pre-trained organ detection.</span>
                                            </div>
                                        </div>
                                    )}
                                </>
                            ) : ('mask_summary' in (data as any) || 'confidence_scores' in (data as any)) ? (
                                <>
                                    <h4 className="text-sm font-medium text-zinc-300">Segmentation Summary:</h4>
                                    <div className="text-xs text-zinc-300 space-y-1">
                                        {'mask_summary' in (data as any) && (data as any).mask_summary?.total_masks != null && (
                                            <div>Total Masks: {(data as any).mask_summary.total_masks}</div>
                                        )}
                                        {'best_mask_score' in (data as any) && typeof (data as any).best_mask_score === 'number' && (
                                            <div>Best Mask Score: {((data as any).best_mask_score * 100).toFixed(1)}%</div>
                                        )}
                                    </div>
                                </>
                            ) : (
                                <div className="text-xs text-zinc-500">No structured segmentation metrics available.</div>
                            )}
                        </div>
                    )}
                    {showRawData && (
                        <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 max-h-96 overflow-y-auto">
                            <pre className="text-xs text-zinc-400 whitespace-pre-wrap overflow-x-auto">
                                {JSON.stringify(result.resultData, null, 2)}
                            </pre>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
