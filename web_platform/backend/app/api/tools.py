"""
Tool API Routes

Endpoints for tool management and execution details.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..database import get_db
from ..models import Doctor, Patient, ToolExecution, ToolExecutionLog, ToolExecutionResult
from ..schemas.tool import (
    ToolExecutionDetailResponse,
    ToolExecutionResponse,
    ToolExecutionLogResponse,
    ToolExecutionResultResponse,
    ToolLoadRequest,
)
from ..dependencies import get_current_doctor
from ..services.tool_manager import tool_manager
from ..utils.logging_config import logger

router = APIRouter()


@router.get("")
def list_tools(current_doctor: Doctor = Depends(get_current_doctor)):
    """List all available tools with their current status."""
    logger.debug(f"Doctor {current_doctor.id} requesting tool list")
    tools = tool_manager.get_all_tools()
    available_count = sum(1 for t in tools if t.get('status') == 'available')
    logger.info(f"Returning {len(tools)} tools ({available_count} available)")
    return tools


@router.post("/{tool_id}/load")
def load_tool(
    tool_id: str,
    current_doctor: Doctor = Depends(get_current_doctor)
):
    """Load/activate a tool."""
    logger.info(f"Doctor {current_doctor.id} loading tool: {tool_id}")
    
    result = tool_manager.load_tool(tool_id)
    
    if not result["success"]:
        logger.error(f"Failed to load tool {tool_id}: {result.get('error')}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to load tool")
        )
    
    tool_info = tool_manager.get_tool(tool_id)
    return {
        "message": result["message"],
        "tool": {
            "id": tool_info.id,
            "name": tool_info.name,
            "status": tool_info.status,
            "loaded_at": tool_info.loaded_at.isoformat() if tool_info.loaded_at else None
        }
    }


@router.post("/{tool_id}/unload")
def unload_tool(
    tool_id: str,
    current_doctor: Doctor = Depends(get_current_doctor)
):
    """Unload/deactivate a tool."""
    logger.info(f"Doctor {current_doctor.id} unloading tool: {tool_id}")
    
    result = tool_manager.unload_tool(tool_id)
    
    if not result["success"]:
        logger.error(f"Failed to unload tool {tool_id}: {result.get('error')}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to unload tool")
        )
    
    tool_info = tool_manager.get_tool(tool_id)
    return {
        "message": result["message"],
        "tool": {
            "id": tool_info.id,
            "name": tool_info.name,
            "status": tool_info.status
        }
    }


@router.get("/executions/{execution_id}", response_model=ToolExecutionDetailResponse)
def get_execution_detail(
    execution_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Get detailed information about a tool execution."""
    
    # Verify execution belongs to doctor
    execution = db.query(ToolExecution).join(ToolExecution.message).join(
        ToolExecution.message.property.mapper.class_.chat
    ).join(Patient).filter(
        ToolExecution.id == execution_id,
        Patient.doctor_id == current_doctor.id
    ).first()
    
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool execution not found"
        )
    
    # Build detailed response
    execution_data = ToolExecutionResponse.model_validate(execution)
    logs_data = [ToolExecutionLogResponse.model_validate(log) for log in execution.logs]
    result_data = ToolExecutionResultResponse.model_validate(execution.result) if execution.result else None
    
    return ToolExecutionDetailResponse(
        execution=execution_data,
        logs=logs_data,
        result=result_data
    )




