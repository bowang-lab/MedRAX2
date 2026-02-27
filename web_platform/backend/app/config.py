"""
Application Configuration

Loads settings from environment variables with validation.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List, Union


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

    # File Upload
    UPLOAD_DIR: str = "uploads"
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


settings = Settings()
