"""
Chat Schemas

Pydantic schemas for chat-related operations.
"""

from pydantic import BaseModel, Field
from datetime import datetime


class ChatBase(BaseModel):
    """Base chat schema."""

    name: str | None = Field(None, max_length=255)


class ChatCreate(ChatBase):
    """Schema for creating a chat."""

    pass


class ChatUpdate(ChatBase):
    """Schema for updating a chat."""

    pass


class ChatResponse(BaseModel):
    """Schema for chat response."""

    id: str
    patient_id: str
    name: str
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None = None
    message_count: int = 0
    scan_count: int = 0

    class Config:
        from_attributes = True
