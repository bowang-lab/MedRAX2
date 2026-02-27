"""
Memory Management Schemas

Pydantic models for memory management API responses.
"""

from pydantic import BaseModel, Field


class MemoryStatsResponse(BaseModel):
    """Memory statistics for a chat"""

    chat_id: str = Field(..., description="Chat ID")
    message_count: int = Field(..., description="Number of messages in the chat")
    scan_count: int = Field(..., description="Number of scans in the chat")
    tool_execution_count: int = Field(..., description="Number of tool executions in the chat")
    has_context: bool = Field(..., description="Whether the chat has any context/messages")

    class Config:
        json_schema_extra = {
            "example": {
                "chat_id": "abc123",
                "message_count": 10,
                "scan_count": 3,
                "tool_execution_count": 5,
                "has_context": True,
            }
        }


class ClearMemoryResponse(BaseModel):
    """Response from clearing chat memory"""

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable message about the operation")
    chat_id: str = Field(..., description="Chat ID that was cleared")

    class Config:
        json_schema_extra = {
            "example": {"success": True, "message": "Memory cleared for chat abc123", "chat_id": "abc123"}
        }


class SystemCleanupStatsData(BaseModel):
    """Statistics from system cleanup"""

    checkpoints_cleared: int = Field(..., description="Number of checkpoints cleared")
    memory_freed_mb: float = Field(..., description="Amount of memory freed in MB")


class SystemCleanupStatsResponse(BaseModel):
    """Response from system memory cleanup"""

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable message about the operation")
    stats: SystemCleanupStatsData = Field(..., description="Cleanup statistics")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "System memory cleanup completed. Cleared 10 checkpoints.",
                "stats": {"checkpoints_cleared": 10, "memory_freed_mb": 45.3},
            }
        }
