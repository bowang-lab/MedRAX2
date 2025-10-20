"""
Message API Routes

Endpoints for messages and SSE streaming.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import asyncio

from ..database import get_db
from ..models import Doctor, Patient, Chat, Message, Scan, MessageScan, ToolExecution, ToolExecutionLog
from ..schemas.message import MessageCreate, MessageResponse, MessageWithDetails, StreamRequest
from ..schemas.scan import ScanResponse
from ..schemas.tool import ToolExecutionResponse
from ..dependencies import get_current_doctor
from ..utils.sse import create_sse_event

router = APIRouter()


@router.get("/chats/{chat_id}/messages", response_model=List[MessageWithDetails])
def list_messages(
    chat_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """List all messages in a chat."""
    
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
    
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at).all()
    
    # Build full message responses with scans and tool executions
    messages_with_details = []
    for msg in messages:
        msg_dict = MessageResponse.model_validate(msg).model_dump()
        msg_dict['attached_scans'] = [ScanResponse.model_validate(scan) for scan in msg.attached_scans]
        msg_dict['tool_executions'] = [ToolExecutionResponse.model_validate(ex) for ex in msg.tool_executions]
        messages_with_details.append(MessageWithDetails(**msg_dict))
    
    return messages_with_details


@router.post("/chats/{chat_id}/messages", response_model=MessageWithDetails, status_code=status.HTTP_201_CREATED)
def create_message(
    chat_id: str,
    message_data: MessageCreate,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Create a new message in a chat."""
    
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
    
    # Create message
    message = Message(
        chat_id=chat_id,
        role="user",  # User message
        content=message_data.content
    )
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
    msg_dict['attached_scans'] = [ScanResponse.model_validate(scan) for scan in message.attached_scans]
    msg_dict['tool_executions'] = []
    
    return MessageWithDetails(**msg_dict)


@router.post("/chats/{chat_id}/stream")
async def stream_chat_response(
    chat_id: str,
    stream_data: StreamRequest,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Stream AI response for a user message using Server-Sent Events."""
    
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
    
    async def event_generator():
        """Generate SSE events for the streaming response."""
        try:
            # 1. Create user message
            user_message = Message(
                chat_id=chat_id,
                role="user",
                content=stream_data.content
            )
            db.add(user_message)
            db.flush()
            
            # Attach scans
            if stream_data.scan_ids:
                for scan_id in stream_data.scan_ids:
                    scan = db.query(Scan).filter(Scan.id == scan_id, Scan.chat_id == chat_id).first()
                    if scan:
                        user_message.attached_scans.append(scan)
            
            db.commit()
            db.refresh(user_message)
            
            # 2. Send message_start event
            yield create_sse_event("message_start", messageId=user_message.id)
            
            # 3. Create assistant message
            assistant_message = Message(
                chat_id=chat_id,
                role="assistant",
                content=""  # Will be built incrementally
            )
            db.add(assistant_message)
            db.commit()
            db.refresh(assistant_message)
            
            # 4. Simulate tool execution (replace with actual MedRAX integration)
            # For now, just send a simple response
            response_text = f"Received your message: '{stream_data.content}'. "
            response_text += "MedRAX analysis would happen here with real tool execution."
            
            # Simulate streaming content chunks
            for i, char in enumerate(response_text):
                assistant_message.content += char
                if i % 10 == 0:  # Send chunk every 10 characters
                    yield create_sse_event("content_chunk", content=char * 10 if i + 10 < len(response_text) else response_text[i:])
                    await asyncio.sleep(0.05)  # Simulate processing delay
            
            # Update final content
            assistant_message.content = response_text
            chat.updated_at = datetime.utcnow()
            chat.patient.last_activity_at = datetime.utcnow()
            db.commit()
            
            # 5. Send message_done event
            yield create_sse_event("message_done", messageId=assistant_message.id)
            
        except Exception as e:
            # Send error event
            yield create_sse_event("error", error=str(e))
            db.rollback()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/messages/{message_id}/executions", response_model=List[ToolExecutionResponse])
def get_message_executions(
    message_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Get all tool executions for a message."""
    
    # Verify message belongs to doctor
    message = db.query(Message).join(Chat).join(Patient).filter(
        Message.id == message_id,
        Patient.doctor_id == current_doctor.id
    ).first()
    
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    return [ToolExecutionResponse.model_validate(ex) for ex in message.tool_executions]




