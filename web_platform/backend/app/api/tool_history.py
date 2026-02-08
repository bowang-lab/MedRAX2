"""
Tool History API Endpoints

Provides endpoints for querying tool execution history.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional

from ..database.session import get_db
from ..dependencies import get_current_doctor
from ..models.doctor import Doctor
from ..models.patient import Patient
from ..models.chat import Chat
from ..models.message import Message
from ..models.tool_execution import ToolExecution, ToolExecutionLog, ToolExecutionResult
from ..schemas.tool import ToolExecutionResponse, ToolHistoryQuery
from ..utils.logging_config import logger


router = APIRouter()


def enrich_tool_execution(execution: ToolExecution) -> dict:
    """Enrich tool execution with computed fields."""
    # Calculate execution time if completed
    execution_time_ms = None
    if execution.completed_at and execution.started_at:
        delta = execution.completed_at - execution.started_at
        execution_time_ms = int(delta.total_seconds() * 1000)

    # Get display name from tool registry
    tool_display_name = execution.tool_name
    try:
        from ..services.tool_manager import tool_manager

        tool_info = tool_manager.get_tool(execution.tool_name)
        if tool_info:
            # Normalize to .name across APIs for consistency
            tool_display_name = tool_info.name
    except:
        pass

    return {
        "id": execution.id,
        "message_id": execution.message_id,
        "request_id": execution.request_id,
        "tool_name": execution.tool_name,
        "tool_display_name": tool_display_name,
        "status": execution.status,
        "started_at": execution.started_at,
        "completed_at": execution.completed_at,
        "execution_time_ms": execution_time_ms,
        "image_paths": execution.image_paths,
    }


@router.get("/chats/{chat_id}/tool-history", response_model=List[ToolExecutionResponse])
async def get_tool_history(
    chat_id: str,
    filter_by_request: Optional[str] = Query(None, description="Filter by request ID"),
    filter_by_tool: Optional[str] = Query(None, description="Filter by tool name"),
    latest_only: bool = Query(False, description="Return only latest execution per tool"),
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """
    Get tool execution history for a chat.

    Args:
        chat_id: Chat ID
        filter_by_request: Only return executions from this request
        filter_by_tool: Only return executions from this tool
        latest_only: Only return latest execution per tool

    Returns:
        List of tool execution records
    """
    # Verify chat belongs to doctor
    chat = db.query(Chat).join(Patient).filter(Chat.id == chat_id, Patient.doctor_id == current_doctor.id).first()

    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    # Build query
    query = db.query(ToolExecution).join(Message).filter(Message.chat_id == chat_id)

    if filter_by_request:
        query = query.filter(ToolExecution.request_id == filter_by_request)

    if filter_by_tool:
        query = query.filter(ToolExecution.tool_name == filter_by_tool)

    query = query.order_by(ToolExecution.started_at.desc())

    executions = query.all()

    # If latest_only, keep only most recent per tool
    if latest_only:
        seen_tools = set()
        filtered = []
        for execution in executions:
            if execution.tool_name not in seen_tools:
                filtered.append(execution)
                seen_tools.add(execution.tool_name)
        executions = filtered

    logger.info(
        f"tool_history_fetched chat_id={chat_id[:8]} count={len(executions)} "
        f"filter_request={filter_by_request is not None} filter_tool={filter_by_tool is not None} latest_only={latest_only}"
    )

    # Convert to response format with computed fields
    return [ToolExecutionResponse(**enrich_tool_execution(execution)) for execution in executions]


@router.get("/messages/{message_id}/tool-history", response_model=List[ToolExecutionResponse])
async def get_message_tool_history(
    message_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """
    Get tool execution history for a specific message.

    This is useful for the "show me tool history for this message" feature.

    Args:
        message_id: Message ID

    Returns:
        List of tool executions for this message
    """
    # Verify message belongs to doctor's patient
    message = db.query(Message).join(Chat).join(Patient).filter(Message.id == message_id, Patient.doctor_id == current_doctor.id).first()

    if not message:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

    # Get tool executions for this message
    executions = db.query(ToolExecution).filter(ToolExecution.message_id == message_id).order_by(ToolExecution.started_at.asc()).all()

    logger.info(f"message_tool_history_fetched message_id={message_id[:8]} count={len(executions)}")

    return [ToolExecutionResponse(**enrich_tool_execution(execution)) for execution in executions]


@router.get("/tool-executions/{execution_id}", response_model=ToolExecutionResponse)
async def get_tool_execution(
    execution_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """
    Get detailed information about a specific tool execution.

    Args:
        execution_id: Tool execution ID

    Returns:
        Detailed tool execution information
    """
    # Verify execution belongs to doctor's patient
    execution = (
        db.query(ToolExecution)
        .join(Message)
        .join(Chat)
        .join(Patient)
        .filter(ToolExecution.id == execution_id, Patient.doctor_id == current_doctor.id)
        .first()
    )

    if not execution:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool execution not found")

    return ToolExecutionResponse(**enrich_tool_execution(execution))
