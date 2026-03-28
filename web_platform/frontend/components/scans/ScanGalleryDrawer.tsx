/**
 * ScanGalleryDrawer Component
 *
 * Drawer/modal showing all scans for a patient across all chats.
 * Allows viewing and managing scans.
 */

"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { X, Trash2, Download, ZoomIn } from "lucide-react";
import { Drawer } from "../ui/Drawer";
import { Spinner } from "../ui/Spinner";
import { getPatientScans, deleteScan } from "../../lib/api/scans";
import { formatDateTime, classNames } from "../../lib/utils";
import { getImageUrl } from "../../lib/utils/image";
import type { Scan } from "../../lib/types/scan";
import { useAppStore } from "../../lib/store/appStore";

/**
 * ScanGalleryDrawer Component Props
 * @property isOpen - Controls drawer visibility (required)
 * @property patientId - Patient ID to load scans for (null = no patient selected)
 * @property onClose - Callback when drawer should close (required)
 */
interface ScanGalleryDrawerProps {
  /** Controls drawer visibility */
  isOpen: boolean;
  /** Patient ID to load scans for (null = no patient selected) */
  patientId: string | null;
  /** Callback when drawer should close */
  onClose: () => void;
}

export function ScanGalleryDrawer({ isOpen, patientId, onClose }: ScanGalleryDrawerProps) {
  const [scans, setScans] = useState<Scan[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedScan, setSelectedScan] = useState<Scan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [failedImages, setFailedImages] = useState<Set<string>>(new Set());
  const { patients, updatePatient } = useAppStore();
  const currentPatient = useMemo(
    () => patients.find((p) => p.id === patientId) || null,
    [patients, patientId]
  );

  const handleImageError = (imagePath: string) => {
    setFailedImages((prev) => new Set(prev).add(imagePath));
  };

  const loadScans = useCallback(async () => {
    if (!patientId) return;

    setIsLoading(true);
    setError(null);
    try {
      const fetchedScans = await getPatientScans(patientId);
      setScans(fetchedScans);
      updatePatient(patientId, { scanCount: fetchedScans.length });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load scans");
    } finally {
      setIsLoading(false);
    }
  }, [patientId, updatePatient]);

  useEffect(() => {
    if (isOpen && patientId) {
      setSelectedScan(null);
      setFailedImages(new Set());
      setError(null);
      loadScans();
    }
  }, [isOpen, patientId, loadScans]);

  const handleDeleteScan = async (scanId: string) => {
    if (!confirm("Are you sure you want to delete this scan?")) return;

    try {
      await deleteScan(scanId);
      setScans((prev) => prev.filter((s) => s.id !== scanId));
      if (currentPatient) {
        updatePatient(currentPatient.id, {
          scanCount: Math.max(0, (currentPatient.scanCount || 0) - 1),
        });
      }
      if (selectedScan?.id === scanId) {
        setSelectedScan(null);
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete scan");
    }
  };

  return (
    <Drawer isOpen={isOpen} onClose={onClose} title="Patient Scans" size="lg">
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Spinner size="lg" />
        </div>
      ) : error ? (
        <div className="text-red-400 text-sm text-center py-12">{error}</div>
      ) : scans.length > 0 ? (
        <div className="space-y-4">
          {/* Scan Grid */}
          <div className="grid grid-cols-2 gap-4">
            {scans.map((scan) => {
              const imageUrl = getImageUrl(scan.displayPath);
              return (
                <div
                  key={scan.id}
                  className={classNames(
                    "relative group rounded-lg overflow-hidden border-2 transition-colors cursor-pointer",
                    selectedScan?.id === scan.id
                      ? "border-blue-500"
                      : "border-zinc-800 hover:border-zinc-700"
                  )}
                  onClick={() => setSelectedScan(scan)}
                >
                  {!imageUrl || failedImages.has(imageUrl) ? (
                    <div className="w-full h-48 flex items-center justify-center bg-red-900/20 border border-red-800 text-red-400 text-xs p-4 text-center">
                      ⚠️ Failed to load scan
                    </div>
                  ) : (
                    /* eslint-disable-next-line @next/next/no-img-element -- Dynamic medical images from backend */
                    <img
                      src={imageUrl}
                      alt="Scan"
                      className="w-full h-48 object-cover"
                      onError={() => handleImageError(imageUrl)}
                    />
                  )}

                  {/* Overlay on hover */}
                  <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center space-x-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedScan(scan);
                      }}
                      className="p-2 bg-zinc-800 rounded-md hover:bg-zinc-700"
                      title="View"
                    >
                      <ZoomIn className="h-5 w-5 text-white" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteScan(scan.id);
                      }}
                      className="p-2 bg-red-600 rounded-md hover:bg-red-700"
                      title="Delete"
                    >
                      <Trash2 className="h-5 w-5 text-white" />
                    </button>
                  </div>

                  {/* Scan Info */}
                  <div className="p-2 bg-zinc-900">
                    <p className="text-xs text-zinc-400 truncate">{formatDateTime(scan.uploadedAt)}</p>
                    <p className="text-xs text-zinc-500">{scan.fileType.toUpperCase()}</p>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Selected Scan Detail */}
          {selectedScan &&
            (() => {
              const selectedImageUrl = getImageUrl(selectedScan.displayPath);
              return (
                <div className="mt-6 p-4 bg-zinc-900 rounded-lg border border-zinc-800">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-medium text-white">Scan Details</h3>
                    <button onClick={() => setSelectedScan(null)} className="text-zinc-400 hover:text-white">
                      <X className="h-4 w-4" />
                    </button>
                  </div>

                  {!selectedImageUrl || failedImages.has(selectedImageUrl) ? (
                    <div className="w-full h-64 flex items-center justify-center bg-red-900/20 border border-red-800 rounded-lg text-red-400 text-sm p-4 mb-3">
                      ⚠️ Failed to load scan image
                    </div>
                  ) : (
                    /* eslint-disable-next-line @next/next/no-img-element -- Dynamic medical images from backend */
                    <img
                      src={selectedImageUrl}
                      alt="Selected Scan"
                      className="w-full rounded-lg mb-3"
                      onError={() => handleImageError(selectedImageUrl)}
                    />
                  )}

                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-zinc-400">Uploaded:</span>
                      <span className="text-zinc-300">{formatDateTime(selectedScan.uploadedAt)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-400">Format:</span>
                      <span className="text-zinc-300">{selectedScan.fileType.toUpperCase()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-400">Type:</span>
                      <span className="text-zinc-300">{selectedScan.fileType.toUpperCase()}</span>
                    </div>
                  </div>

                  <div className="mt-4 flex items-center space-x-2">
                    <a
                      href={selectedImageUrl || "#"}
                      download
                      className="flex-1 py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium text-center transition-colors"
                      {...(!selectedImageUrl && { onClick: (e) => e.preventDefault() })}
                    >
                      <Download className="inline h-4 w-4 mr-2" />
                      Download
                    </a>
                    <button
                      onClick={() => handleDeleteScan(selectedScan.id)}
                      className="flex-1 py-2 px-4 bg-red-600 hover:bg-red-700 text-white rounded-md text-sm font-medium transition-colors"
                    >
                      <Trash2 className="inline h-4 w-4 mr-2" />
                      Delete
                    </button>
                  </div>
                </div>
              );
            })()}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <p className="text-zinc-500 text-sm">No scans uploaded yet</p>
        </div>
      )}
    </Drawer>
  );
}
