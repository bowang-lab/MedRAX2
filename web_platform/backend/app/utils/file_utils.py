"""
File Utilities

File handling and upload utilities.
"""

import os
import aiofiles
from pathlib import Path
from fastapi import UploadFile
import uuid

from ..config import settings


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


async def save_upload_file(file: UploadFile, subdirectory: str = "") -> tuple[str, str]:
    """
    Save an uploaded file to disk.
    
    Args:
        file: The uploaded file
        subdirectory: Optional subdirectory within upload dir
        
    Returns:
        Tuple of (file_path, display_path)
        
    Raises:
        ValueError: If filename is None or empty
    """
    # Validate filename
    if not file.filename:
        raise ValueError("File must have a valid filename")
    
    # Create upload directory if it doesn't exist
    upload_path = Path(settings.UPLOAD_DIR)
    if subdirectory:
        upload_path = upload_path / subdirectory
    upload_path.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    ext = get_file_extension(file.filename)
    if ext:
        unique_filename = f"{uuid.uuid4()}.{ext}"
    else:
        # If no extension, just use UUID (shouldn't happen with validation, but defensive)
        unique_filename = str(uuid.uuid4())
    
    file_path = upload_path / unique_filename
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Generate display path (URL path for frontend)
    display_path = f"/uploads/{subdirectory}/{unique_filename}" if subdirectory else f"/uploads/{unique_filename}"
    
    return str(file_path), display_path


def delete_file(file_path: str) -> bool:
    """
    Delete a file from disk.
    
    Args:
        file_path: Path to the file to delete
        
    Returns:
        True if deleted, False if file doesn't exist
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            return True
        return False
    except Exception:
        return False




