"""
Scan Schemas

Pydantic schemas for scan-related operations.
"""

from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime


class ScanBase(BaseModel):
    """Base scan schema."""

    file_type: str = Field(..., alias="fileType")
    file_size: int = Field(..., alias="fileSize")

    model_config = ConfigDict(populate_by_name=True)


class ScanResponse(ScanBase):
    """Schema for scan response."""

    id: str
    chat_id: str = Field(..., alias="chatId")
    file_path: str = Field(..., alias="filePath")
    display_path: str = Field(..., alias="displayPath")
    uploaded_at: datetime = Field(..., alias="uploadedAt")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
