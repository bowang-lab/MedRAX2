/**
 * ScanUploadZone Component
 *
 * Drag-and-drop zone for uploading medical scans.
 * Supports DICOM, JPG, PNG files.
 */

"use client";

import { useState, DragEvent } from "react";
import { Upload, X, FileImage, Loader2 } from "lucide-react";
import { uploadScans } from "../../lib/api/scans";
import { classNames } from "../../lib/utils";
import { UI_CONFIG } from "../../lib/config/app";
import { validation } from "../../lib/utils/validation";
import type { Scan } from "../../lib/types/scan";

/**
 * ScanUploadZone Component Props
 * @property chatId - Chat ID to associate uploaded scans with (required)
 * @property onUploadComplete - Callback when scans are successfully uploaded (required)
 * @property onUploadError - Optional callback when upload fails with error message
 */
interface ScanUploadZoneProps {
  /** Chat ID to associate uploaded scans with */
  chatId: string;
  /** Callback when scans are successfully uploaded */
  onUploadComplete: (scans: Scan[]) => void;
  /** Optional callback when upload fails with error message */
  onUploadError?: (error: string) => void;
}

export function ScanUploadZone({ chatId, onUploadComplete, onUploadError }: ScanUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleDragEnter = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    handleFiles(files);
  };

  const handleFiles = (files: File[]) => {
    setError(null);

    // Validate files
    const validFiles: File[] = [];
    for (const file of files) {
      const validationError = validation.scanFile(file, UI_CONFIG.maxFileSize, UI_CONFIG.allowedFileTypes);
      if (validationError) {
        setError(validationError);
        onUploadError?.(validationError);
        return;
      }
      validFiles.push(file);
    }

    setSelectedFiles(validFiles);
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;

    setIsUploading(true);
    setError(null);

    try {
      const uploadedScans = await uploadScans(chatId, selectedFiles);
      onUploadComplete(uploadedScans);
      setSelectedFiles([]);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to upload scans";
      setError(errorMsg);
      onUploadError?.(errorMsg);
    } finally {
      setIsUploading(false);
    }
  };

  const handleRemoveFile = (index: number) => {
    setSelectedFiles((files) => files.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <div
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={classNames(
          "border-2 border-dashed rounded-lg p-6 transition-colors",
          isDragging ? "border-blue-500 bg-blue-500/10" : "border-zinc-700 bg-zinc-900/50",
          "hover:border-zinc-600 hover:bg-zinc-900/70"
        )}
      >
        <div className="flex flex-col items-center justify-center text-center">
          <Upload className={classNames("h-10 w-10 mb-3", isDragging ? "text-blue-500" : "text-zinc-500")} />
          <p className="text-sm text-zinc-300 mb-1">Drag and drop scans here, or click to browse</p>
          <p className="text-xs text-zinc-500">
            Supports DICOM, JPG, PNG (max {UI_CONFIG.maxFileSize / (1024 * 1024)}MB)
          </p>

          <input
            type="file"
            multiple
            accept={UI_CONFIG.allowedFileExtensions.join(",")}
            onChange={handleFileInput}
            className="hidden"
            id="scan-upload-input"
          />
          <label
            htmlFor="scan-upload-input"
            className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm cursor-pointer transition-colors"
          >
            Browse Files
          </label>
        </div>
      </div>

      {/* Selected Files */}
      {selectedFiles.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-zinc-300">Selected Files ({selectedFiles.length})</p>
            <button onClick={() => setSelectedFiles([])} className="text-xs text-red-400 hover:text-red-300">
              Clear All
            </button>
          </div>

          <div className="space-y-2 max-h-60 overflow-y-auto">
            {selectedFiles.map((file, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-zinc-800 rounded-lg">
                <div className="flex items-center space-x-3 flex-1 min-w-0">
                  <FileImage className="h-5 w-5 text-zinc-400 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-zinc-300 truncate">{file.name}</p>
                    <p className="text-xs text-zinc-500">{(file.size / 1024).toFixed(1)} KB</p>
                  </div>
                </div>
                <button
                  onClick={() => handleRemoveFile(index)}
                  className="p-1 text-zinc-400 hover:text-red-400 transition-colors"
                  disabled={isUploading}
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>

          {/* Upload Button */}
          <button
            onClick={handleUpload}
            disabled={isUploading}
            className={classNames(
              "w-full py-2 px-4 rounded-md font-medium text-sm transition-colors",
              isUploading ? "bg-zinc-700 text-zinc-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700 text-white"
            )}
          >
            {isUploading ? (
              <span className="flex items-center justify-center">
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Uploading...
              </span>
            ) : (
              `Upload ${selectedFiles.length} ${selectedFiles.length === 1 ? "File" : "Files"}`
            )}
          </button>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="p-3 bg-red-900/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}
    </div>
  );
}
