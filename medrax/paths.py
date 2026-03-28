"""
Canonical on-disk layout for MedRAX web + tools.

All user uploads and tool-generated images live under a single data root
(default: ``web_platform/backend/medrax_data/``) with:

- ``uploads/`` — chat attachments and user files
- ``generated/`` — tool outputs (DICOM previews, segmentations, etc.)

Override with env ``MEDRAX_DATA_ROOT`` (absolute path recommended).
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    # medrax/paths.py -> medrax -> repo root
    return Path(__file__).resolve().parents[1]


def resolve_medrax_data_root() -> Path:
    env = os.getenv("MEDRAX_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (_repo_root() / "web_platform" / "backend" / "medrax_data").resolve()


def resolve_uploads_dir() -> Path:
    return resolve_medrax_data_root() / "uploads"


def resolve_generated_dir() -> Path:
    return resolve_medrax_data_root() / "generated"
