"use client";

/**
 * Full-Screen Image Modal Component
 *
 * Displays images in full-screen with zoom, pan, and navigation controls.
 */

import { useEffect, useCallback, useState } from "react";
import { Button } from "./Button";

export interface ImageModalProps {
  /** Array of image URLs to display */
  images: string[];
  /** Index of initially selected image */
  initialIndex?: number;
  /** Callback when modal is closed */
  onClose: () => void;
  /** Whether the modal is open */
  isOpen: boolean;
}

/**
 * Full-screen image modal with zoom and navigation controls.
 */
export function ImageModal({ images, initialIndex = 0, onClose, isOpen }: ImageModalProps) {
  const clampedInitialIndex = Math.max(0, Math.min(initialIndex, images.length - 1));
  const [currentIndex, setCurrentIndex] = useState(clampedInitialIndex);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Lock body scroll when modal is open
  useEffect(() => {
    if (!isOpen) return;
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, [isOpen]);

  // Handle keyboard shortcuts
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!isOpen || images.length === 0) return;

      switch (e.key) {
        case "Escape":
          onClose();
          break;
        case "ArrowLeft":
          if (images.length > 1) {
            setCurrentIndex((prev) => (prev > 0 ? prev - 1 : images.length - 1));
          }
          break;
        case "ArrowRight":
          if (images.length > 1) {
            setCurrentIndex((prev) => (prev < images.length - 1 ? prev + 1 : 0));
          }
          break;
        case "+":
        case "=":
          setZoom((prev) => Math.min(prev + 0.5, 5));
          break;
        case "-":
        case "_":
          setZoom((prev) => Math.max(prev - 0.5, 0.5));
          break;
        case "0":
          setZoom(1);
          setPan({ x: 0, y: 0 });
          break;
      }
    },
    [isOpen, images.length, onClose]
  );

  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // Handle mouse drag for panning
  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom > 1) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Handle wheel zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom((prev) => Math.max(0.5, Math.min(5, prev + delta)));
  };

  const handleNext = () => {
    if (images.length > 1) {
      setCurrentIndex((prev) => (prev < images.length - 1 ? prev + 1 : 0));
      setZoom(1);
      setPan({ x: 0, y: 0 });
    }
  };

  const handlePrevious = () => {
    if (images.length > 1) {
      setCurrentIndex((prev) => (prev > 0 ? prev - 1 : images.length - 1));
      setZoom(1);
      setPan({ x: 0, y: 0 });
    }
  };

  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev + 0.5, 5));
  };

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(prev - 0.5, 0.5));
  };

  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Don't render if not open or no images
  if (!isOpen || images.length === 0) return null;

  // Ensure currentIndex is valid (defensive programming)
  const safeIndex = Math.max(0, Math.min(currentIndex, images.length - 1));

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-90"
      onClick={onClose}
    >
      <div className="relative w-full h-full flex items-center justify-center" onClick={(e) => e.stopPropagation()}>
        {/* Close Button */}
        <Button variant="secondary" size="sm" onClick={onClose} className="absolute top-4 right-4 z-10">
          ✕ Close
        </Button>

        {/* Navigation Controls */}
        {images.length > 1 && (
          <>
            <Button
              variant="secondary"
              size="sm"
              onClick={handlePrevious}
              className="absolute left-4 top-1/2 -translate-y-1/2 z-10"
            >
              ← Previous
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={handleNext}
              className="absolute right-4 top-1/2 -translate-y-1/2 z-10"
            >
              Next →
            </Button>
          </>
        )}

        {/* Zoom Controls */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-zinc-800 rounded-lg p-2 z-10">
          <Button variant="secondary" size="sm" onClick={handleZoomOut}>
            −
          </Button>
          <span className="text-white text-sm min-w-[60px] text-center">
            {Math.round(zoom * 100)}%
          </span>
          <Button variant="secondary" size="sm" onClick={handleZoomIn}>
            +
          </Button>
          <Button variant="secondary" size="sm" onClick={handleReset}>
            Reset
          </Button>
        </div>

        {/* Image Counter */}
        {images.length > 1 && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-zinc-800 rounded-lg px-4 py-2 text-white text-sm z-10">
            {safeIndex + 1} / {images.length}
          </div>
        )}

        {/* Image */}
        <div
          className="w-full h-full flex items-center justify-center overflow-hidden cursor-move"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={images[safeIndex]}
            alt={`Image ${safeIndex + 1}`}
            className="max-w-full max-h-full object-contain select-none"
            style={{
              transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
              transition: isDragging ? "none" : "transform 0.2s ease-out",
              cursor: zoom > 1 ? (isDragging ? "grabbing" : "grab") : "default",
            }}
            draggable={false}
            onError={(e) => {
              e.currentTarget.style.display = "none";
              const container = e.currentTarget.parentElement;
              if (container) {
                const errorMsg = document.createElement("div");
                errorMsg.className = "text-red-400 text-center p-8";
                errorMsg.textContent = `⚠️ Failed to load image: ${images[safeIndex]}`;
                container.appendChild(errorMsg);
              }
            }}
          />
        </div>

        {/* Help Text */}
        <div className="absolute bottom-16 left-1/2 -translate-x-1/2 text-zinc-400 text-xs text-center">
          ← → Navigate • + − Zoom • 0 Reset • Esc Close
        </div>
      </div>
    </div>
  );
}
