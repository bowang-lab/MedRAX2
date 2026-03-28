"""
Application Configuration

Loads settings from environment variables with validation.
"""

import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List, Union, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # Application
    APP_NAME: str = "MedRAX API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "sqlite:///./medrax.db"

    # Security
    # REQUIRED: Must be set in .env file - no default value for security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 3600

    # API Access Control - Shared secret between frontend and backend
    # All API requests must include this in X-API-Secret header
    # REQUIRED: Must be set in .env file - no default value for security
    API_SECRET_KEY: str
    REQUIRE_API_SECRET: bool = True

    # CORS
    CORS_ORIGINS: Union[str, List[str]] = "http://localhost:3000,http://127.0.0.1:3000"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS_ORIGINS from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        if isinstance(v, list):
            return v
        return ["http://localhost:3000"]

    # On-disk data root: ``medrax_data/uploads`` and ``medrax_data/generated`` (see resolve_* helpers).
    # Set to an absolute path to relocate all MedRAX file storage.
    MEDRAX_DATA_ROOT: Optional[str] = None

    # File Upload — optional override for uploads directory only (default: MEDRAX_DATA_ROOT/uploads)
    UPLOAD_DIR: Optional[str] = None
    MAX_UPLOAD_SIZE: int = 104857600  # 100MB
    ALLOWED_EXTENSIONS: set = {"jpg", "jpeg", "png", "gif", "dcm", "dicom"}

    # AI/ML API Keys
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    GOOGLE_API_KEY: str = ""
    GOOGLE_SEARCH_API_KEY: str = ""
    GOOGLE_SEARCH_ENGINE_ID: str = ""
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    COHERE_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    XAI_API_KEY: str = ""

    # Model Caching Configuration
    MODEL_CACHE_DIR: str = "./model_cache"
    HUGGINGFACE_CACHE_DIR: str = "~/.cache/huggingface"
    TORCH_CACHE_DIR: str = "~/.cache/torch"

    # Model Weights Directory (for large pretrained models like ArcPlus)
    MODELWEIGHTS: str = "/model-weights"

    # Model Download Settings
    ALLOW_MODEL_DOWNLOADS: bool = True
    MAX_MODEL_DOWNLOAD_SIZE: int = 10737418240  # 10GB

    # Tool Configuration
    TOOL_TIMEOUT: int = 300  # 5 minutes
    MAX_CONCURRENT_TOOLS: int = 3
    AUTO_UNLOAD_TOOLS: bool = False

    # Device Configuration for Medical Imaging Tools
    # Options: "cuda", "cpu", "auto" (auto-detect)
    DEVICE: str = "auto"
    FORCE_CPU: bool = False

    # Multi-GPU tool loading (web backend): pick cuda:i from free VRAM + physical order.
    # TOOL_GPU_STRATEGY=auto: use TOOL_GPU_PHYSICAL_ORDER + TOOL_GPU_MIN_FREE_MIB (needs nvidia-smi).
    # TOOL_GPU_STRATEGY=fixed: use TOOL_DEVICE if set, else DEVICE (non-auto).
    TOOL_GPU_STRATEGY: str = "auto"
    # Physical GPU indices to try in order (comma-separated). Example: 2,1 = prefer GPU 2, else GPU 1.
    TOOL_GPU_PHYSICAL_ORDER: str = "2,1"
    # Minimum free VRAM (MiB) on a GPU before we prefer placing a new tool there.
    TOOL_GPU_MIN_FREE_MIB: int = 2048
    # When TOOL_GPU_STRATEGY=fixed, device string for all tools (e.g. cuda:0, cuda:1).
    TOOL_DEVICE: str = ""

    @field_validator("UPLOAD_DIR", mode="before")
    @classmethod
    def coerce_legacy_upload_dir(cls, v):
        """Treat bare ``uploads`` as unset so we use MEDRAX_DATA_ROOT/uploads."""
        if v in (None, "", "uploads"):
            return None
        return v


def resolve_medrax_data_root() -> Path:
    """Root directory for MedRAX files (uploads + generated)."""
    if settings.MEDRAX_DATA_ROOT:
        return Path(settings.MEDRAX_DATA_ROOT).expanduser().resolve()
    # Keep in sync with medrax.paths.resolve_medrax_data_root default (no import: app may load before medrax on path)
    backend_root = Path(__file__).resolve().parent.parent
    return (backend_root / "medrax_data").resolve()


def resolve_upload_dir() -> Path:
    """User uploads (chat attachments, etc.)."""
    if settings.UPLOAD_DIR:
        return Path(settings.UPLOAD_DIR).expanduser().resolve()
    return resolve_medrax_data_root() / "uploads"


def resolve_generated_dir() -> Path:
    """
    Tool-generated images (DICOM previews, segmentations, etc.).

    If ``MEDRAX_TEMP_DIR`` is set and ``MEDRAX_DATA_ROOT`` is not, uses the legacy temp directory only.
    """
    if settings.MEDRAX_DATA_ROOT or not os.getenv("MEDRAX_TEMP_DIR"):
        return resolve_medrax_data_root() / "generated"
    return Path(os.environ["MEDRAX_TEMP_DIR"]).expanduser().resolve()


settings = Settings()

# Expose canonical data root to medrax.tools (they read os.environ / medrax.paths only)
os.environ["MEDRAX_DATA_ROOT"] = str(resolve_medrax_data_root())
