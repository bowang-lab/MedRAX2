"""
Memory Management API Endpoints

Provides endpoints for managing chat memory and cleanup.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..dependencies import get_current_doctor
from ..models.doctor import Doctor
from ..models.patient import Patient
from ..models.chat import Chat
from ..schemas.memory import (
    MemoryStatsResponse,
    ClearMemoryResponse,
    SystemCleanupStatsResponse,
    SystemCleanupStatsData,
)
from ..utils.logging_config import logger


router = APIRouter()


@router.post("/chats/{chat_id}/memory/clear", response_model=ClearMemoryResponse)
async def clear_chat_memory(
    chat_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
) -> ClearMemoryResponse:
    """
    Clear conversation memory for a chat.

    This removes the LangGraph checkpointer state for the chat,
    effectively resetting the conversation context.

    Args:
        chat_id: Chat ID to clear memory for

    Returns:
        Success message with operation status
    """
    # Verify chat belongs to doctor
    chat = db.query(Chat).join(Patient).filter(Chat.id == chat_id, Patient.doctor_id == current_doctor.id).first()

    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    # Clear LangGraph checkpointer state for this thread_id (chat_id)
    from ..services.tool_manager import tool_manager

    success = tool_manager.clear_chat_memory(chat_id)

    logger.info(f"chat_memory_cleared chat_id={chat_id[:8]} success={success}")

    return ClearMemoryResponse(
        success=success,
        message=f"Memory cleared for chat {chat_id}" if success else "Failed to clear memory",
        chat_id=chat_id,
    )


@router.get("/chats/{chat_id}/memory/stats", response_model=MemoryStatsResponse)
async def get_chat_memory_stats(
    chat_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
) -> MemoryStatsResponse:
    """
    Get memory statistics for a chat.

    Returns information about the conversation context size,
    message count, and memory usage.

    Args:
        chat_id: Chat ID

    Returns:
        Memory statistics including message, scan, and tool execution counts
    """
    # Verify chat belongs to doctor
    chat = db.query(Chat).join(Patient).filter(Chat.id == chat_id, Patient.doctor_id == current_doctor.id).first()

    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    # Count messages
    from ..models.message import Message

    message_count = db.query(Message).filter(Message.chat_id == chat_id).count()

    # Count scans
    from ..models.scan import Scan

    scan_count = db.query(Scan).filter(Scan.chat_id == chat_id).count()

    # Count tool executions
    from ..models.tool_execution import ToolExecution

    tool_execution_count = db.query(ToolExecution).join(Message).filter(Message.chat_id == chat_id).count()

    logger.info(
        f"chat_memory_stats_fetched chat_id={chat_id[:8]} messages={message_count} scans={scan_count} executions={tool_execution_count}"
    )

    return MemoryStatsResponse(
        chat_id=chat_id,
        message_count=message_count,
        scan_count=scan_count,
        tool_execution_count=tool_execution_count,
        has_context=message_count > 0,
    )


@router.post("/system/memory/cleanup", response_model=SystemCleanupStatsResponse)
async def cleanup_system_memory(current_doctor: Doctor = Depends(get_current_doctor)) -> SystemCleanupStatsResponse:
    """
    Trigger system-wide memory cleanup.

    This is an admin operation that clears old checkpointer states
    and performs garbage collection.

    Returns:
        Cleanup statistics including memory freed and checkpoints cleared
    """
    import gc
    import os
    import psutil
    from datetime import datetime, timedelta

    logger.info(f"system_memory_cleanup_triggered by doctor {current_doctor.id[:8]}")

    checkpoints_cleared = 0
    memory_before = 0
    memory_after = 0

    try:
        # Get memory before cleanup
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB

        # Clear old checkpointer states
        from ..services.tool_manager import tool_manager

        if tool_manager.checkpointer and hasattr(tool_manager.checkpointer, "storage"):
            with tool_manager._checkpointer_lock:
                # MemorySaver stores data by thread_id without timestamps
                # Clear excess entries when storage exceeds threshold
                if len(tool_manager.checkpointer.storage) > 100:
                    # Retain 50 most recent entries
                    excess = len(tool_manager.checkpointer.storage) - 50
                    if excess > 0:
                        for tid in list(tool_manager.checkpointer.storage.keys())[:excess]:
                            try:
                                del tool_manager.checkpointer.storage[tid]
                                checkpoints_cleared += 1
                            except:
                                pass

        # Force garbage collection
        gc.collect()

        # Get memory after cleanup
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_freed = max(0, memory_before - memory_after)

        logger.info(
            f"system_memory_cleanup_completed checkpoints_cleared={checkpoints_cleared} memory_freed_mb={memory_freed:.2f}"
        )

        return SystemCleanupStatsResponse(
            success=True,
            message=f"System memory cleanup completed. Cleared {checkpoints_cleared} checkpoints.",
            stats=SystemCleanupStatsData(checkpoints_cleared=checkpoints_cleared, memory_freed_mb=round(memory_freed, 2)),
        )

    except Exception as e:
        logger.error(f"system_memory_cleanup_error error={str(e)}")
        return SystemCleanupStatsResponse(
            success=False,
            message=f"Cleanup completed with errors: {str(e)}",
            stats=SystemCleanupStatsData(checkpoints_cleared=checkpoints_cleared, memory_freed_mb=0),
        )
"""
Memory Management API Endpoints

Provides endpoints for managing chat memory and cleanup.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..dependencies import get_current_doctor
from ..models.doctor import Doctor
from ..models.patient import Patient
from ..models.chat import Chat
from ..schemas.memory import (
    MemoryStatsResponse,
    ClearMemoryResponse,
    SystemCleanupStatsResponse,
    SystemCleanupStatsData,
)
from ..utils.logging_config import logger


router = APIRouter()


@router.post("/chats/{chat_id}/memory/clear", response_model=ClearMemoryResponse)
async def clear_chat_memory(
    chat_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
) -> ClearMemoryResponse:
    """
    Clear conversation memory for a chat.
    
    This removes the LangGraph checkpointer state for the chat,
    effectively resetting the conversation context.
    
    Args:
        chat_id: Chat ID to clear memory for
        
    Returns:
        Success message with operation status
    """
    # Verify chat belongs to doctor
    chat = db.query(Chat).join(Patient).filter(
        Chat.id == chat_id,
        Patient.doctor_id == current_doctor.id
    ).first()
    
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    # Clear LangGraph checkpointer state for this thread_id (chat_id)
    from ..services.tool_manager import tool_manager
    success = tool_manager.clear_chat_memory(chat_id)
    
    logger.info(f"chat_memory_cleared chat_id={chat_id[:8]} success={success}")
    
    return ClearMemoryResponse(
        success=success,
        message=f"Memory cleared for chat {chat_id}" if success else "Failed to clear memory",
        chat_id=chat_id
    )


@router.get("/chats/{chat_id}/memory/stats", response_model=MemoryStatsResponse)
async def get_chat_memory_stats(
    chat_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
) -> MemoryStatsResponse:
    """
    Get memory statistics for a chat.
    
    Returns information about the conversation context size,
    message count, and memory usage.
    
    Args:
        chat_id: Chat ID
        
    Returns:
        Memory statistics including message, scan, and tool execution counts
    """
    # Verify chat belongs to doctor
    chat = db.query(Chat).join(Patient).filter(
        Chat.id == chat_id,
        Patient.doctor_id == current_doctor.id
    ).first()
    
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    # Count messages
    from ..models.message import Message
    message_count = db.query(Message).filter(Message.chat_id == chat_id).count()
    
    # Count scans
    from ..models.scan import Scan
    scan_count = db.query(Scan).filter(Scan.chat_id == chat_id).count()
    
    # Count tool executions
    from ..models.tool_execution import ToolExecution
    tool_execution_count = db.query(ToolExecution).join(Message).filter(
        Message.chat_id == chat_id
    ).count()
    
    logger.info(f"chat_memory_stats_fetched chat_id={chat_id[:8]} messages={message_count} scans={scan_count} executions={tool_execution_count}")
    
    return MemoryStatsResponse(
        chat_id=chat_id,
        message_count=message_count,
        scan_count=scan_count,
        tool_execution_count=tool_execution_count,
        has_context=message_count > 0
    )


@router.post("/system/memory/cleanup", response_model=SystemCleanupStatsResponse)
async def cleanup_system_memory(
    current_doctor: Doctor = Depends(get_current_doctor)
) -> SystemCleanupStatsResponse:
    """
    Trigger system-wide memory cleanup.
    
    This is an admin operation that clears old checkpointer states
    and performs garbage collection.
    
    Returns:
        Cleanup statistics including memory freed and checkpoints cleared
    """
    import gc
    import os
    import psutil
    from datetime import datetime, timedelta
    
    logger.info(f"system_memory_cleanup_triggered by doctor {current_doctor.id[:8]}")
    
    checkpoints_cleared = 0
    memory_before = 0
    memory_after = 0
    
    try:
        # Get memory before cleanup
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Clear old checkpointer states
        from ..services.tool_manager import tool_manager
        # Cleanup per-chat checkpointers to avoid unbounded growth
        try:
            tool_manager.cleanup_old_chats()
        except Exception as e:
            logger.warning(f"cleanup_old_chats_failed error={e}")
        
        if tool_manager.checkpointer and hasattr(tool_manager.checkpointer, 'storage'):
            with tool_manager._checkpointer_lock:
                # MemorySaver stores data by thread_id without timestamps
                # Clear excess entries when storage exceeds threshold
                if len(tool_manager.checkpointer.storage) > 100:
                    # Retain 50 most recent entries
                    excess = len(tool_manager.checkpointer.storage) - 50
                    if excess > 0:
                        for tid in list(tool_manager.checkpointer.storage.keys())[:excess]:
                            try:
                                del tool_manager.checkpointer.storage[tid]
                                checkpoints_cleared += 1
                            except:
                                pass
        
        # Force garbage collection
        gc.collect()
        
        # Get memory after cleanup
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_freed = max(0, memory_before - memory_after)
        
        logger.info(f"system_memory_cleanup_completed checkpoints_cleared={checkpoints_cleared} memory_freed_mb={memory_freed:.2f}")
        
        return SystemCleanupStatsResponse(
            success=True,
            message=f"System memory cleanup completed. Cleared {checkpoints_cleared} checkpoints.",
            stats=SystemCleanupStatsData(
                checkpoints_cleared=checkpoints_cleared,
                memory_freed_mb=round(memory_freed, 2)
            )
        )
        
    except Exception as e:
        logger.error(f"system_memory_cleanup_error error={str(e)}")
        return SystemCleanupStatsResponse(
            success=False,
            message=f"Cleanup completed with errors: {str(e)}",
            stats=SystemCleanupStatsData(
                checkpoints_cleared=checkpoints_cleared,
                memory_freed_mb=0
            )
        )

