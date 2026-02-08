"""
Message Schemas

Pydantic schemas for message-related operations.
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List, Optional

from .scan import ScanResponse
from .tool import ToolExecutionResponse


class MessageBase(BaseModel):
    """Base message schema."""

    content: str = Field(..., min_length=1, max_length=10000, description="Message content")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate and sanitize message content."""
        # Strip whitespace
        v = v.strip()

        # Check if empty after stripping
        if not v:
            raise ValueError("Message content cannot be empty or whitespace only")

        # Check maximum length
        if len(v) > 10000:
            raise ValueError("Message content cannot exceed 10000 characters")

        return v


class MessageCreate(MessageBase):
    """Schema for creating a message."""

    scan_ids: List[str] = []


class MessageResponse(BaseModel):
    """Schema for message response - allows empty content for failed assistant messages."""

    id: str
    chat_id: str
    role: str  # 'user', 'assistant', 'system'
    content: str = Field(..., max_length=10000, description="Message content")
    created_at: datetime

    class Config:
        from_attributes = True


class MessageWithDetails(MessageResponse):
    """Schema for message with attached scans and tool executions."""

    attached_scans: List[ScanResponse] = []
    tool_executions: List[ToolExecutionResponse] = []


class StreamRequest(MessageCreate):
    """Schema for streaming request."""

    pass
