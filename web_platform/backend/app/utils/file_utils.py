"""
File Utilities

File handling and upload utilities.
"""

import logging
import uuid
from pathlib import Path

import aiofiles
import numpy as np
import pydicom
from PIL import Image
from fastapi import UploadFile

from ..config import settings

logger = logging.getLogger(__name__)


def get_file_extension(filename: str | None) -> str:
    """
    Get file extension from filename.

    Args:
        filename: The filename

    Returns:
        File extension without dot (empty string if no extension or None filename)
    """
    if not filename:
        return ''
    return Path(filename).suffix.lstrip('.').lower()


def is_allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed.

    Args:
        filename: The filename to check

    Returns:
        True if allowed, False otherwise
    """
    ext = get_file_extension(filename)
    return ext in settings.ALLOWED_EXTENSIONS


def _apply_windowing(img: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply basic window/level adjustment."""
    img_min = center - width / 2
    img_max = center + width / 2
    img = np.clip(img, img_min, img_max)
    denom = width if width != 0 else (img_max - img_min) or 1
    img = ((img - img_min) / denom * 255).astype(np.uint8)
    return img


def convert_dicom_to_png(dicom_path: Path) -> Path | None:
    """
    Convert a DICOM file to a PNG for display.

    Returns:
        Path to the generated PNG or None if conversion fails.
    """
    try:
        dcm = pydicom.dcmread(dicom_path)
        img = dcm.pixel_array.astype(float)

        # Apply rescale slope/intercept if available
        slope = getattr(dcm, "RescaleSlope", 1)
        intercept = getattr(dcm, "RescaleIntercept", 0)
        img = img * slope + intercept

        center = getattr(dcm, "WindowCenter", None)
        width = getattr(dcm, "WindowWidth", None)

        # Handle multi-value fields
        if isinstance(center, (list, tuple)):
            center = center[0]
        if isinstance(width, (list, tuple)):
            width = width[0]

        if center is not None and width is not None:
            img = _apply_windowing(img, float(center), float(width))
        else:
            img_min, img_max = np.min(img), np.max(img)
            if img_max == img_min:
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        png_path = dicom_path.with_suffix(".png")
        Image.fromarray(img).save(png_path)
        logger.info(f"Converted DICOM to PNG: {dicom_path} -> {png_path}")
        return png_path
    except Exception as e:
        logger.warning(f"Failed to convert DICOM {dicom_path} to PNG: {e}")
        return None


async def save_upload_file(file: UploadFile, subdirectory: str = "") -> tuple[str, str]:
    """
    Save an uploaded file to disk.

    Returns:
        Tuple of (file_path, display_path)
    """
    if not file.filename:
        raise ValueError("File must have a valid filename")

    upload_path = Path(settings.UPLOAD_DIR)
    if subdirectory:
        upload_path = upload_path / subdirectory
    upload_path.mkdir(parents=True, exist_ok=True)

    ext = get_file_extension(file.filename)
    if ext:
        unique_filename = f"{uuid.uuid4()}.{ext}"
    else:
        unique_filename = str(uuid.uuid4())

    file_path = upload_path / unique_filename

    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    display_path = f"/uploads/{subdirectory}/{unique_filename}" if subdirectory else f"/uploads/{unique_filename}"

    if ext in {"dcm", "dicom"}:
        png_path = convert_dicom_to_png(file_path)
        if png_path and png_path.exists():
            display_path = f"/{png_path.as_posix()}"
        else:
            logger.warning(f"Using original DICOM for display; PNG conversion failed for {file_path}")

    return str(file_path), display_path


def delete_file(file_path: str) -> bool:
    """Delete a file from disk."""
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            return True
        return False
    except Exception as e:
        logger.debug(f"Failed to delete file {file_path}: {e}")
        return False
