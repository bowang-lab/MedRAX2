"""
Chat API Routes

Endpoints for chat CRUD operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from sqlalchemy import func

from ..database import get_db
from ..models import Doctor, Patient, Chat, Message, Scan
from ..schemas.chat import ChatCreate, ChatUpdate, ChatResponse
from ..dependencies import get_current_doctor
from ..utils.formatting import generate_chat_name
from ..utils.logging_config import logger

router = APIRouter()


def generate_chat_name() -> str:
    """Generate a chat name from current datetime."""
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %I:%M %p")


@router.get("/patients/{patient_id}/chats", response_model=List[ChatResponse])
def list_patient_chats(
    patient_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """List all chats for a patient."""
    logger.debug(f"Listing chats for patient {patient_id} by doctor {current_doctor.id}")

    # Verify patient belongs to doctor
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == current_doctor.id).first()

    if not patient:
        logger.warning(f"Patient {patient_id} not found for doctor {current_doctor.id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    chats = db.query(Chat).filter(Chat.patient_id == patient_id).order_by(Chat.created_at.desc()).all()
    logger.info(f"Found {len(chats)} chats for patient {patient_id}")

    # Enrich with computed fields
    result = []
    for chat in chats:
        # Get last message timestamp
        last_message = db.query(Message).filter(Message.chat_id == chat.id).order_by(Message.created_at.desc()).first()

        # Count messages and scans
        message_count = db.query(func.count(Message.id)).filter(Message.chat_id == chat.id).scalar() or 0
        scan_count = db.query(func.count(Scan.id)).filter(Scan.chat_id == chat.id).scalar() or 0

        chat_dict = {
            "id": chat.id,
            "patient_id": chat.patient_id,
            "name": chat.name,
            "created_at": chat.created_at,
            "updated_at": chat.updated_at,
            "last_message_at": last_message.created_at if last_message else None,
            "message_count": message_count,
            "scan_count": scan_count,
        }
        result.append(ChatResponse(**chat_dict))

    return result


@router.post("/patients/{patient_id}/chats", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
def create_chat(
    patient_id: str,
    chat_data: ChatCreate,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Create a new chat for a patient."""

    # Verify patient belongs to doctor
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == current_doctor.id).first()

    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    # Generate name if not provided
    chat_name = chat_data.name if chat_data.name else generate_chat_name()

    chat = Chat(patient_id=patient_id, name=chat_name)
    db.add(chat)

    # Update patient last activity
    patient.last_activity_at = datetime.utcnow()

    db.commit()
    db.refresh(chat)

    # Return with default computed fields (new chat has no messages/scans yet)
    chat_dict = {
        "id": chat.id,
        "patient_id": chat.patient_id,
        "name": chat.name,
        "created_at": chat.created_at,
        "updated_at": chat.updated_at,
        "last_message_at": None,
        "message_count": 0,
        "scan_count": 0,
    }
    return ChatResponse(**chat_dict)


@router.get("/chats/{chat_id}", response_model=ChatResponse)
def get_chat(
    chat_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get a specific chat."""

    chat = (
        db.query(Chat).join(Patient).filter(Chat.id == chat_id, Patient.doctor_id == current_doctor.id).first()
    )

    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    # Enrich with computed fields
    last_message = db.query(Message).filter(Message.chat_id == chat.id).order_by(Message.created_at.desc()).first()
    message_count = db.query(func.count(Message.id)).filter(Message.chat_id == chat.id).scalar() or 0
    scan_count = db.query(func.count(Scan.id)).filter(Scan.chat_id == chat.id).scalar() or 0

    chat_dict = {
        "id": chat.id,
        "patient_id": chat.patient_id,
        "name": chat.name,
        "created_at": chat.created_at,
        "updated_at": chat.updated_at,
        "last_message_at": last_message.created_at if last_message else None,
        "message_count": message_count,
        "scan_count": scan_count,
    }
    return ChatResponse(**chat_dict)


@router.patch("/chats/{chat_id}", response_model=ChatResponse)
def update_chat(
    chat_id: str,
    chat_data: ChatUpdate,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Update a chat (rename)."""

    chat = (
        db.query(Chat).join(Patient).filter(Chat.id == chat_id, Patient.doctor_id == current_doctor.id).first()
    )

    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    # Update name
    if chat_data.name is not None:
        chat.name = chat_data.name

    chat.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(chat)

    # Enrich with computed fields
    last_message = db.query(Message).filter(Message.chat_id == chat.id).order_by(Message.created_at.desc()).first()
    message_count = db.query(func.count(Message.id)).filter(Message.chat_id == chat.id).scalar() or 0
    scan_count = db.query(func.count(Scan.id)).filter(Scan.chat_id == chat.id).scalar() or 0

    chat_dict = {
        "id": chat.id,
        "patient_id": chat.patient_id,
        "name": chat.name,
        "created_at": chat.created_at,
        "updated_at": chat.updated_at,
        "last_message_at": last_message.created_at if last_message else None,
        "message_count": message_count,
        "scan_count": scan_count,
    }
    return ChatResponse(**chat_dict)


@router.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(
    chat_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Delete a chat and all associated messages, scans, and tool execution files."""
    from ..models import Scan, ToolExecution
    from ..utils.file_utils import delete_file, is_generated_tool_image_path

    chat = (
        db.query(Chat).join(Patient).filter(Chat.id == chat_id, Patient.doctor_id == current_doctor.id).first()
    )

    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    # Clean up scan files from disk before deleting DB records
    scans = db.query(Scan).filter(Scan.chat_id == chat_id).all()
    for scan in scans:
        delete_file(scan.file_path)

    # Clean up tool execution generated images from disk
    # Get all tool executions via messages in this chat
    tool_executions = db.query(ToolExecution).join(Message).filter(Message.chat_id == chat_id).all()

    for execution in tool_executions:
        if execution.image_paths:
            for image_path in execution.image_paths:
                # Only delete generated images (in temp or output dirs), not input scans
                if isinstance(image_path, str) and is_generated_tool_image_path(image_path):
                    delete_file(image_path)

    # Delete database record (cascades to messages, scans, tool executions)
    db.delete(chat)
    db.commit()

    return None
