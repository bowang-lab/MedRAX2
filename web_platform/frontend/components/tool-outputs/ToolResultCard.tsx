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

"use client";

import { useMemo, useState } from "react";
import type { ToolExecutionResult } from "../../lib/types/tool";
import { getImageUrl } from "../../lib/utils/image";

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

// Basic check for plain objects (avoids arrays/Date/etc.)
const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);

// Convert python-ish output (with np.float32, tuples, single quotes) into JSON-parsable text
const toJsonishString = (raw: string) => {
  let cleaned = raw.trim();
  // Drop np.float32/np.float64 wrappers that break JSON
  cleaned = cleaned.replace(
    /np\.float(?:16|32|64)?\(\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\s*\)/g,
    "$1"
  );
  cleaned = cleaned.replace(/np\.int(?:8|16|32|64)?\(\s*([+\-]?\d+)\s*\)/g, "$1");
  // Normalise booleans/nulls
  cleaned = cleaned
    .replace(/\bNone\b/g, "null")
    .replace(/\bTrue\b/g, "true")
    .replace(/\bFalse\b/g, "false");
  // Convert leading/trailing tuple to array so JSON.parse can handle it
  if (cleaned.startsWith("(") && cleaned.endsWith(")")) {
    cleaned = `[${cleaned.slice(1, -1)}]`;
  }
  // Switch single quotes to double quotes for JSON compatibility
  cleaned = cleaned.replace(/'/g, '"');
  return cleaned;
};

// Try to coerce raw string into structured data
const parseRawResult = (raw: string) => {
  const attempts = [() => JSON.parse(raw), () => JSON.parse(toJsonishString(raw))];
  for (const attempt of attempts) {
    try {
      return attempt();
    } catch {
      // try next strategy
    }
  }
  return null;
};

export function ToolResultCard({ toolName, result }: ToolResultCardProps) {
  const [failedImages, setFailedImages] = useState<Set<string>>(new Set());
  const [showRawData, setShowRawData] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  // Normalize raw tool output into structured data + metadata
  const { data, parsedMetadata } = useMemo(() => {
    let normalized: unknown = result.resultData;
    let metadata: Record<string, unknown> | null = null;

    if (isPlainObject(normalized) && "raw" in normalized && typeof normalized.raw === "string") {
      const raw = normalized.raw;
      const parsed = parseRawResult(raw);

      if (parsed !== null) {
        if (Array.isArray(parsed) && parsed.length > 0) {
          normalized = parsed[0];
          if (parsed.length > 1 && isPlainObject(parsed[1])) {
            metadata = parsed[1];
          }
        } else if (isPlainObject(parsed)) {
          normalized = parsed;
        } else {
          // Fallback: wrap primitive into object so downstream logic still works
          normalized = { value: parsed };
        }
      }
    }

    return { data: normalized as Record<string, unknown> | null, parsedMetadata: metadata };
  }, [result.resultData]);

  const handleImageError = (imagePath: string) => {
    setFailedImages((prev) => new Set(prev).add(imagePath));
  };

  // Extract image paths from result data (generated images, visualizations, masks)
  const imagePaths: string[] = [];
  if (isPlainObject(data)) {
    Object.entries(data).forEach(([key, value]) => {
      if (
        (key.toLowerCase().includes("image_path") ||
          key.toLowerCase().includes("visualization") ||
          key.toLowerCase().includes("mask") ||
          key.toLowerCase().includes("output_image") ||
          key.toLowerCase().includes("grounding_image") ||
          key.toLowerCase().includes("segmentation_image")) &&
        typeof value === "string" &&
        value &&
        !value.includes("input") // Exclude input image paths
      ) {
        imagePaths.push(value);
      }
    });
  }

  // Tool-specific rendering helpers
  const isClassificationTool =
    toolName.includes("torchxrayvision") || toolName.includes("arcplus") || toolName.includes("classifier");

  const isSegmentationTool = toolName.includes("medsam") || toolName.includes("segmentation");

  const isVQATool =
    toolName.includes("chexagent") ||
    toolName.includes("llava") ||
    toolName.includes("medgemma") ||
    toolName.includes("vqa");

  const isReportTool = toolName.includes("report");

  const isGroundingTool = toolName.includes("grounding");

  const isSearchTool =
    toolName.includes("duckduckgo") || toolName.includes("web_browser") || toolName.includes("search");

  const isRagTool = toolName.includes("rag") || toolName.includes("retrieval");

  // Optional metadata for richer summaries
  const metadata = parsedMetadata || result.resultMetadata || null;

  // Render classification results in a formatted way
  const renderClassificationResults = () => {
    if (!isClassificationTool || !data) return null;

    // Classification tools return the predictions directly as the data object
    // Format: { "Atelectasis": 0.123, "Cardiomegaly": 0.456, ... }
    const dataObj = isPlainObject(data) ? data : null;
    if (!dataObj) return null;

    const predictions = dataObj.predictions || dataObj.pathology_predictions || dataObj;

    // Check if this looks like classification data (object with numeric values)
    if (!predictions || typeof predictions !== "object") return null;

    // Filter to only numeric probability values
    const pathologies = Object.entries(predictions).filter(
      ([key, value]) => typeof value === "number" && !key.includes("_") && key !== "error" && key !== "image_path"
    );

    if (pathologies.length === 0) return null;

    const sortedPathologies = [...pathologies].sort(([, a], [, b]) => {
      const aNum = typeof a === "number" ? a : Number(a) || 0;
      const bNum = typeof b === "number" ? b : Number(b) || 0;
      return bNum - aNum;
    });

    return (
      <div className="space-y-2">
        <h4 className="text-sm font-medium text-zinc-300">Pathology Predictions:</h4>
        <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 space-y-2">
          {sortedPathologies.map(([pathology, probability]) => {
            const prob = typeof probability === "number" ? probability : 0;
            const percentage = (prob * 100).toFixed(1);
            const isHighProbability = prob > 0.5;

            return (
              <div key={pathology} className="flex items-center justify-between">
                <span className={`text-sm ${isHighProbability ? "text-yellow-400 font-medium" : "text-zinc-400"}`}>
                  {pathology.replace(/_/g, " ")}
                </span>
                <div className="flex items-center gap-2">
                  <div className="w-32 h-2 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${isHighProbability ? "bg-yellow-500" : "bg-blue-500"}`}
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

    const dataObj = isPlainObject(data) ? data : null;
    if (!dataObj) return null;

    const answer = dataObj.answer || dataObj.response || dataObj.text;
    if (!answer || typeof answer !== "string") return null;

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

    const dataObj = isPlainObject(data) ? data : null;
    if (!dataObj) return null;

    const findingsRaw = dataObj.findings || dataObj.Findings;
    const impressionRaw = dataObj.impression || dataObj.Impression;

    // Convert to strings safely
    const findings = typeof findingsRaw === "string" ? findingsRaw : null;
    const impression = typeof impressionRaw === "string" ? impressionRaw : null;

    if (!findings && !impression) return null;

    return (
      <div className="space-y-3">
        {findings && (
          <div>
            <h4 className="text-sm font-medium text-zinc-300 mb-1">Findings:</h4>
            <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3">
              <p className="text-sm text-zinc-100 whitespace-pre-wrap">{findings}</p>
            </div>
          </div>
        )}

        {impression && (
          <div>
            <h4 className="text-sm font-medium text-zinc-300 mb-1">Impression:</h4>
            <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3">
              <p className="text-sm text-zinc-100 whitespace-pre-wrap">{impression}</p>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Render grounding results
  const renderGroundingResults = () => {
    if (!isGroundingTool || !data) return null;

    const dataObj = isPlainObject(data) ? data : null;
    if (!dataObj) return null;

    const predictions = dataObj.predictions;
    if (!Array.isArray(predictions)) return null;

    return (
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-zinc-300">Detected Phrases:</h4>
        {predictions.length === 0 ? (
          <div className="text-sm text-zinc-500">No findings detected</div>
        ) : (
          <div className="space-y-2">
            {predictions.map((prediction, idx) => (
              <div key={idx} className="bg-zinc-900 border border-zinc-700 rounded-lg p-3">
                <div className="text-sm text-zinc-200 font-medium mb-1">
                  {prediction?.phrase || "Unknown"}
                </div>
                {prediction?.bounding_boxes?.image_coordinates && (
                  <div className="text-xs text-zinc-400">
                    {prediction.bounding_boxes.image_coordinates.length} bounding box(es)
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  // Render search results
  const renderSearchResults = () => {
    if (!isSearchTool || !data) return null;

    const dataObj = isPlainObject(data) ? data : null;
    if (!dataObj) return null;

    const results = dataObj.results;
    if (!Array.isArray(results)) return null;

    return (
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-zinc-300">Search Results:</h4>
        {results.length === 0 ? (
          <div className="text-sm text-zinc-500">No results found</div>
        ) : (
          <div className="space-y-2">
            {results.map((result, idx) => (
              <div key={idx} className="bg-zinc-900 border border-zinc-700 rounded-lg p-3">
                <div className="text-sm text-zinc-200 font-medium">{result.title || "Untitled"}</div>
                {result.link && (
                  <a
                    href={result.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-blue-400 hover:text-blue-300 underline"
                  >
                    {result.link}
                  </a>
                )}
                {result.snippet && <p className="text-xs text-zinc-400 mt-1">{result.snippet}</p>}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  const renderRagResults = () => {
    if (!isRagTool || !data) return null;

    const dataObj = isPlainObject(data) ? data : null;
    if (!dataObj) return null;

    const answer = dataObj.answer;
    const sources = dataObj.source_documents;

    if (typeof answer !== "string") return null;

    return (
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-zinc-300">Answer:</h4>
        <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3">
          <p className="text-sm text-zinc-100 whitespace-pre-wrap">{answer}</p>
        </div>

        {Array.isArray(sources) && sources.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-zinc-300">Sources:</h4>
            <div className="space-y-2">
              {sources.map((source, idx) => {
                const sourceObj = isPlainObject(source) ? source : null;
                const content =
                  sourceObj && typeof sourceObj.content === "string" ? sourceObj.content : "Source content unavailable";
                const metadata = sourceObj && isPlainObject(sourceObj.metadata) ? sourceObj.metadata : null;
                const title =
                  metadata && typeof metadata.title === "string"
                    ? metadata.title
                    : metadata && typeof metadata.source === "string"
                      ? metadata.source
                      : `Source ${idx + 1}`;

                return (
                  <div key={idx} className="bg-zinc-900 border border-zinc-700 rounded-lg p-3">
                    <div className="text-xs text-zinc-400 mb-1">{title}</div>
                    <div className="text-xs text-zinc-300 whitespace-pre-wrap">
                      {content.length > 400 ? `${content.slice(0, 400)}…` : content}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {renderClassificationResults()}
      {renderVQAAnswer()}
      {renderReport()}
      {renderGroundingResults()}
      {renderSearchResults()}
      {renderRagResults()}

      {/* Image Outputs */}
      {imagePaths.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-zinc-300">Output Images:</h4>
            {metadata && (
              <div className="text-xs text-zinc-500">
                {Array.isArray(metadata.requested_organs) && metadata.requested_organs.length > 0 && (
                  <div>
                    Requested: <span className="text-zinc-300">{metadata.requested_organs.join(", ")}</span>
                  </div>
                )}
                {Array.isArray(metadata.processed_organs) && (
                  <div>
                    Detected:{" "}
                    <span className="text-zinc-300">
                      {metadata.processed_organs.length > 0 ? metadata.processed_organs.join(", ") : "none"}
                    </span>
                  </div>
                )}
                {typeof metadata.pixel_spacing_mm === "number" && (
                  <div>
                    Pixel spacing: <span className="text-zinc-300">{metadata.pixel_spacing_mm} mm</span>
                  </div>
                )}
                {typeof metadata.threshold_used === "number" && (
                  <div>
                    Threshold: <span className="text-zinc-300">{metadata.threshold_used}</span>
                  </div>
                )}
              </div>
            )}
          </div>
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
                        onClick={() => imageUrl && setPreviewImage(imageUrl)}
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
            {showRawData ? "▼ Hide Raw Data" : "▶ Show Raw Data (for debugging)"}
          </button>
          {/* Segmentation-specific summary (metrics or mask summary) */}
          {isSegmentationTool && isPlainObject(data) && (
            <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 space-y-2">
              {"metrics" in data && isPlainObject(data.metrics) ? (
                <>
                  <h4 className="text-sm font-medium text-zinc-300">Segmentation Metrics:</h4>
                  {Object.keys(data.metrics).length > 0 ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {Object.entries(data.metrics).map(([organ, metrics]) => {
                        const m = isPlainObject(metrics) ? metrics : {};
                        const areaCm2 = typeof m.area_cm2 === "number" ? m.area_cm2.toFixed(2) : undefined;
                        const conf =
                          typeof m.confidence_score === "number" ? (m.confidence_score * 100).toFixed(1) : undefined;
                        const mean = typeof m.mean_intensity === "number" ? m.mean_intensity.toFixed(1) : undefined;
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
                        <span className="text-zinc-300">
                          {" "}
                          Use MedSAM2 instead - it&apos;s more robust and works with a wider variety of X-ray images.
                          MedSAM2 uses advanced segmentation that doesn&apos;t rely on pre-trained organ detection.
                        </span>
                      </div>
                    </div>
                  )}
                </>
              ) : "mask_summary" in data || "confidence_scores" in data ? (
                <>
                  <h4 className="text-sm font-medium text-zinc-300">Segmentation Summary:</h4>
                  <div className="text-xs text-zinc-300 space-y-1">
                    {"mask_summary" in data &&
                      isPlainObject(data.mask_summary) &&
                      typeof data.mask_summary.total_masks === "number" && (
                        <div>Total Masks: {data.mask_summary.total_masks}</div>
                      )}
                    {"best_mask_score" in data && typeof data.best_mask_score === "number" && (
                      <div>Best Mask Score: {(data.best_mask_score * 100).toFixed(1)}%</div>
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
                {JSON.stringify(data ?? result.resultData, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* Image preview modal */}
      {previewImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
          onClick={() => setPreviewImage(null)}
        >
          <div
            className="relative max-h-[90vh] max-w-5xl w-full bg-zinc-950 border border-zinc-800 rounded-lg shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setPreviewImage(null)}
              className="absolute top-2 right-2 rounded-md bg-black/50 hover:bg-black/70 text-zinc-200 text-sm px-2 py-1 border border-zinc-700"
            >
              Close
            </button>
            <div className="flex items-center justify-center p-4 bg-black">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={previewImage} alt="Preview" className="max-h-[80vh] w-auto object-contain" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
