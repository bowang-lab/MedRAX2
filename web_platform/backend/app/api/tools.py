"""
Tool API Routes

Endpoints for tool management and execution details.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
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
    ToolBulkLoadRequest,
    ToolBulkLoadResponse,
    ToolBulkLoadResult,
)
from ..dependencies import get_current_doctor
from ..services.tool_manager import tool_manager
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
            tool_display_name = tool_info.display_name
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


@router.get("")
def list_tools(current_doctor: Doctor = Depends(get_current_doctor)):
    """List all available tools with their current status."""
    logger.debug(f"Doctor {current_doctor.id} requesting tool list")
    tools = tool_manager.get_all_tools()
    available_count = sum(1 for t in tools if t.get('status') == 'available')
    logger.info(f"Returning {len(tools)} tools ({available_count} available)")
    return tools


@router.post("/{tool_id}/load")
async def load_tool(
    tool_id: str,
    background_tasks: BackgroundTasks,
    current_doctor: Doctor = Depends(get_current_doctor)
):
    """Load/activate a tool (starts loading in background for large models)."""
    logger.info(f"Doctor {current_doctor.id} loading tool: {tool_id}")
    
    # Initiate loading (returns immediately)
    result = tool_manager.load_tool(tool_id)
    
    if not result["success"]:
        logger.error(f"Failed to load tool {tool_id}: {result.get('error')}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to load tool")
        )
    
    # Add background task to actually load the tool
    tool_info = tool_manager.get_tool(tool_id)
    if tool_info.status == "loading":
        background_tasks.add_task(tool_manager.load_tool_in_background, tool_id)
        logger.info(f"Added background task to load {tool_id}")
    
    return {
        "message": result["message"],
        "tool": {
            "id": tool_info.id,
            "name": tool_info.name,
            "status": tool_info.status,
            "loaded_at": tool_info.loaded_at.isoformat() if tool_info.loaded_at else None
        }
    }


@router.post("/bulk-load", response_model=ToolBulkLoadResponse)
async def bulk_load_tools(
    bulk: ToolBulkLoadRequest,
    background_tasks: BackgroundTasks,
    current_doctor: Doctor = Depends(get_current_doctor),
):
    """Bulk load tools. If load_all is true, load all available tools unless tool_ids provided."""
    tools = tool_manager.get_all_tools()
    target_ids = set()
    if bulk.load_all and (not bulk.tool_ids or len(bulk.tool_ids) == 0):
        target_ids = {t['id'] for t in tools if t.get('status') in ('available', 'unloaded')}
    else:
        target_ids = set(bulk.tool_ids or [])

    results: list[ToolBulkLoadResult] = []

    for tool_id in target_ids:
        tm_tool = tool_manager.get_tool(tool_id)
        if not tm_tool:
            results.append(ToolBulkLoadResult(id=tool_id, success=False, status="error", message="Tool not found"))
            continue
        # Skip if already loaded
        if tm_tool.status == "loaded":
            results.append(ToolBulkLoadResult(id=tool_id, success=True, status="loaded", message="Already loaded"))
            continue

        # Initiate loading
        result = tool_manager.load_tool(tool_id)
        if not result.get("success"):
            results.append(ToolBulkLoadResult(id=tool_id, success=False, status="error", message=result.get("error")))
            continue

        # Schedule background loading if in loading state
        if tm_tool.status == "loading":
            background_tasks.add_task(tool_manager.load_tool_in_background, tool_id)
            results.append(ToolBulkLoadResult(id=tool_id, success=True, status="loading", message="Loading started"))
        else:
            # If status changed synchronously, reflect it
            results.append(ToolBulkLoadResult(id=tool_id, success=True, status=tm_tool.status, message=result.get("message")))

    return ToolBulkLoadResponse(results=results)


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
    
    # Build detailed response with computed fields
    execution_dict = enrich_tool_execution(execution)
    execution_data = ToolExecutionResponse(**execution_dict)
    logs_data = [ToolExecutionLogResponse.model_validate(log) for log in execution.logs]
    result_data = ToolExecutionResultResponse.model_validate(execution.result) if execution.result else None
    
    return ToolExecutionDetailResponse(
        execution=execution_data,
        logs=logs_data,
        result=result_data
    )




