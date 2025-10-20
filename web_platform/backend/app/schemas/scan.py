"""
Scan Schemas

Pydantic schemas for scan-related operations.
"""

from pydantic import BaseModel
from datetime import datetime


class ScanBase(BaseModel):
    """Base scan schema."""
    file_type: str
    file_size: int


class ScanResponse(ScanBase):
    """Schema for scan response."""
    id: str
    chat_id: str
    file_path: str
    display_path: str
    uploaded_at: datetime
    
    class Config:
        from_attributes = True




