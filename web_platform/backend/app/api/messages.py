"""
Message API Routes

Endpoints for messages and SSE streaming.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, selectinload
from typing import List
from datetime import datetime
import asyncio
import uuid

from ..database import get_db
from ..models import Doctor, Patient, Chat, Message, Scan, MessageScan, ToolExecution, ToolExecutionLog
from ..schemas.message import MessageCreate, MessageResponse, MessageWithDetails, StreamRequest
from ..schemas.scan import ScanResponse
from ..schemas.tool import ToolExecutionResponse
from ..dependencies import get_current_doctor
from ..utils.sse import create_sse_event
from ..utils.logging_config import logger
from ..services.tool_manager import tool_manager
from ..services.chat_processor import ChatProcessor
# from ..services.image_registry import image_registry  # TODO: Re-enable when wrapper is fixed

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
        tool_info = tool_manager.get_tool(execution.tool_name)
        if tool_info:
            tool_display_name = tool_info.name  # ToolInfo has 'name', not 'display_name'
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


@router.get("/chats/{chat_id}/messages", response_model=List[MessageWithDetails])
def list_messages(
    chat_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """List all messages in a chat."""

    # Verify chat belongs to doctor
    chat = db.query(Chat).join(Patient).filter(Chat.id == chat_id, Patient.doctor_id == current_doctor.id).first()

    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    # Use eager loading to avoid N+1 query problem
    messages = (
        db.query(Message)
        .options(selectinload(Message.attached_scans), selectinload(Message.tool_executions))
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at)
        .all()
    )

    # Build full message responses with scans and tool executions
    messages_with_details = []
    for msg in messages:
        msg_dict = MessageResponse.model_validate(msg).model_dump()
        msg_dict["attached_scans"] = [ScanResponse.model_validate(scan) for scan in msg.attached_scans]
        msg_dict["tool_executions"] = [ToolExecutionResponse(**enrich_tool_execution(ex)) for ex in msg.tool_executions]
        messages_with_details.append(MessageWithDetails(**msg_dict))

    return messages_with_details


@router.post("/chats/{chat_id}/messages", response_model=MessageWithDetails, status_code=status.HTTP_201_CREATED)
def create_message(
    chat_id: str,
    message_data: MessageCreate,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Create a new message in a chat."""

    # Verify chat belongs to doctor
    chat = db.query(Chat).join(Patient).filter(Chat.id == chat_id, Patient.doctor_id == current_doctor.id).first()

    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    # Create message
    message = Message(chat_id=chat_id, role="user", content=message_data.content)
    db.add(message)
    db.flush()

    # Attach scans if provided
    if message_data.scan_ids:
        for scan_id in message_data.scan_ids:
            scan = db.query(Scan).filter(Scan.id == scan_id, Scan.chat_id == chat_id).first()
            if scan:
                message.attached_scans.append(scan)

    # Update chat and patient timestamps
    chat.updated_at = datetime.utcnow()
    chat.patient.last_activity_at = datetime.utcnow()

    db.commit()
    db.refresh(message)

    # Build response
    msg_dict = MessageResponse.model_validate(message).model_dump()
    msg_dict["attached_scans"] = [ScanResponse.model_validate(scan) for scan in message.attached_scans]
    msg_dict["tool_executions"] = []

    return MessageWithDetails(**msg_dict)


@router.post("/chats/{chat_id}/stream")
async def stream_chat_response(
    chat_id: str,
    stream_data: StreamRequest,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Stream AI response for a user message using Server-Sent Events."""

    # Verify chat belongs to doctor
    chat = db.query(Chat).join(Patient).filter(Chat.id == chat_id, Patient.doctor_id == current_doctor.id).first()

    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    async def event_generator():
        """Generate SSE events for the streaming response using MedRAX Agent."""
        try:
            # 1. Create user message
            user_message = Message(chat_id=chat_id, role="user", content=stream_data.content)
            db.add(user_message)
            db.flush()

            # Attach scans with comprehensive logging
            logger.info(f"stream_request chat_id={chat_id[:8]} scan_ids={stream_data.scan_ids}")
            if stream_data.scan_ids:
                attached_count = 0
                for scan_id in stream_data.scan_ids:
                    scan = db.query(Scan).filter(Scan.id == scan_id, Scan.chat_id == chat_id).first()
                    if scan:
                        user_message.attached_scans.append(scan)
                        attached_count += 1
                        logger.info(f"attached_scan scan_id={scan.id[:8]} file_path={scan.file_path}")
                    else:
                        logger.warning(f"scan_not_found scan_id={scan_id} chat_id={chat_id[:8]}")
                logger.info(f"attached_scans_total count={attached_count}/{len(stream_data.scan_ids)}")
            else:
                logger.info("no_scan_ids_provided")

            db.commit()
            db.refresh(user_message)

            # 2. Send message_start event
            yield create_sse_event("message_start", messageId=user_message.id)

            # 3. Create assistant message
            assistant_message = Message(chat_id=chat_id, role="assistant", content="")
            db.add(assistant_message)
            db.commit()
            db.refresh(assistant_message)

            # 4. Generate unique request ID for this analysis
            request_id = str(uuid.uuid4())
            logger.info(f"Generated request_id={request_id[:8]} for chat_id={chat_id[:8]}")

            # 5. Create MedRAX agent with request-specific wrapped tools and chat isolation
            agent = tool_manager.create_agent(request_id=request_id, chat_id=chat_id)

            if agent is None:
                # No tools loaded - send error
                error_msg = (
                    "No MedRAX tools are currently loaded. Please load at least one tool in Settings → Tools Management."
                )
                yield create_sse_event("error", error=error_msg)
                assistant_message.content = error_msg
                db.commit()
                yield create_sse_event("message_done", messageId=assistant_message.id)
                return

            # 6. Create chat processor and process message
            # Attach tool executions to the assistant message for more intuitive UI grouping
            processor = ChatProcessor(agent, db, chat_id, tool_target_message_id=assistant_message.id)

            async for event in processor.process_message(user_message, scan_ids=stream_data.scan_ids):
                # Forward events from processor
                if event["type"] == "content_chunk":
                    assistant_message.content += event["data"].get("content", "")
                    yield create_sse_event("content_chunk", content=event["data"].get("content", ""))
                elif event["type"] == "tool_start":
                    yield create_sse_event("tool_start", **event["data"])
                elif event["type"] == "tool_output":
                    yield create_sse_event("tool_output", **event["data"])
                elif event["type"] == "tool_done":
                    yield create_sse_event("tool_done", **event["data"])
                elif event["type"] == "tool_error":
                    yield create_sse_event("tool_error", **event["data"])

            # Update final content and timestamps
            # Get fresh chat reference from DB to ensure session binding
            fresh_chat = db.query(Chat).filter(Chat.id == chat.id).first()
            if fresh_chat:
                fresh_chat.updated_at = datetime.utcnow()
                if fresh_chat.patient:
                    fresh_chat.patient.last_activity_at = datetime.utcnow()
            db.commit()

            # 6. Send message_done event
            yield create_sse_event("message_done", messageId=assistant_message.id)

        except Exception as e:
            logger.error(f"Error in stream_chat_response: {e}")
            # Send error event
            yield create_sse_event("error", error=str(e))
            db.rollback()
        finally:
            # Clean up request-specific resources
            try:
                # TODO: Clean up image registry when wrapper is re-enabled
                # if 'request_id' in locals():
                #     image_registry.cleanup_request(request_id)
                #     logger.debug(f"Cleaned up image registry for request {request_id[:8]}")

                # Close database session
                db.close()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/messages/{message_id}/executions", response_model=List[ToolExecutionResponse])
def get_message_executions(
    message_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get all tool executions for a message."""

    # Verify message belongs to doctor
    message = (
        db.query(Message).join(Chat).join(Patient).filter(Message.id == message_id, Patient.doctor_id == current_doctor.id).first()
    )

    if not message:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

    return [ToolExecutionResponse.model_validate(ex) for ex in message.tool_executions]



