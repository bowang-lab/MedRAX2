"""
Tool Schemas

Pydantic schemas for tool execution and management.
"""

from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any, Optional


class ToolExecutionLogResponse(BaseModel):
    """Schema for tool execution log."""

    id: str
    execution_id: str
    log_level: str  # 'info', 'warning', 'error'
    message: str
    timestamp: datetime

    class Config:
        from_attributes = True


class ToolExecutionResultResponse(BaseModel):
    """Schema for tool execution result."""

    id: str
    execution_id: str
    result_data: Dict[str, Any]
    result_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ToolExecutionResponse(BaseModel):
    """Schema for tool execution."""

    id: str
    message_id: str
    request_id: Optional[str] = None
    tool_name: str
    tool_display_name: str = ""
    status: str  # 'pending', 'running', 'completed', 'failed'
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[int] = None
    image_paths: Optional[List[str]] = None

    class Config:
        from_attributes = True


class ToolExecutionDetailResponse(BaseModel):
    """Schema for detailed tool execution info."""

    execution: ToolExecutionResponse
    logs: List[ToolExecutionLogResponse]
    result: Optional[ToolExecutionResultResponse] = None


class ToolInfo(BaseModel):
    """Schema for tool information from tool_manager.get_all_tools()."""

    id: str
    name: str
    description: str
    category: str
    status: str  # 'available', 'unavailable', 'loaded', 'unloaded', 'error', 'loading'
    dependencies: List[str] = []
    requires_gpu: bool
    error_message: Optional[str] = None
    loaded_at: Optional[str] = None  # ISO format datetime string


class ToolLoadRequest(BaseModel):
    """Schema for loading/unloading tools."""

    pass  # Empty body, tool ID comes from path


class ToolHistoryQuery(BaseModel):
    """Schema for tool history query parameters."""

    filter_by_request: Optional[str] = None
    filter_by_tool: Optional[str] = None
    filter_by_image: Optional[str] = None
    latest_only: bool = False


class ToolBulkLoadRequest(BaseModel):
    """Schema for bulk loading tools."""

    tool_ids: Optional[List[str]] = None  # If None and load_all is True, load all available
    load_all: bool = False


class ToolBulkLoadResult(BaseModel):
    """Result for each tool in a bulk load operation."""

    id: str
    success: bool
    status: str
    message: Optional[str] = None


class ToolBulkLoadResponse(BaseModel):
    """Response for bulk load operation."""

    results: List[ToolBulkLoadResult]
