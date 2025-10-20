"""
Message Schemas

Pydantic schemas for message-related operations.
"""

from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

from .scan import ScanResponse
from .tool import ToolExecutionResponse


class MessageBase(BaseModel):
    """Base message schema."""
    content: str


class MessageCreate(MessageBase):
    """Schema for creating a message."""
    scan_ids: List[str] = []


class MessageResponse(MessageBase):
    """Schema for message response."""
    id: str
    chat_id: str
    role: str  # 'user', 'assistant', 'system'
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




