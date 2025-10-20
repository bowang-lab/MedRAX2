"""
Chat API Routes

Endpoints for chat CRUD operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from ..database import get_db
from ..models import Doctor, Patient, Chat
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
    db: Session = Depends(get_db)
):
    """List all chats for a patient."""
    logger.debug(f"Listing chats for patient {patient_id} by doctor {current_doctor.id}")
    
    # Verify patient belongs to doctor
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_doctor.id
    ).first()
    
    if not patient:
        logger.warning(f"Patient {patient_id} not found for doctor {current_doctor.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    chats = db.query(Chat).filter(Chat.patient_id == patient_id).order_by(Chat.created_at.desc()).all()
    logger.info(f"Found {len(chats)} chats for patient {patient_id}")
    return [ChatResponse.model_validate(chat) for chat in chats]


@router.post("/patients/{patient_id}/chats", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
def create_chat(
    patient_id: str,
    chat_data: ChatCreate,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Create a new chat for a patient."""
    
    # Verify patient belongs to doctor
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_doctor.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    # Generate name if not provided
    chat_name = chat_data.name if chat_data.name else generate_chat_name()
    
    chat = Chat(
        patient_id=patient_id,
        name=chat_name
    )
    db.add(chat)
    
    # Update patient last activity
    patient.last_activity_at = datetime.utcnow()
    
    db.commit()
    db.refresh(chat)
    
    return ChatResponse.model_validate(chat)


@router.get("/chats/{chat_id}", response_model=ChatResponse)
def get_chat(
    chat_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Get a specific chat."""
    
    chat = db.query(Chat).join(Patient).filter(
        Chat.id == chat_id,
        Patient.doctor_id == current_doctor.id
    ).first()
    
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    return ChatResponse.model_validate(chat)


@router.patch("/chats/{chat_id}", response_model=ChatResponse)
def update_chat(
    chat_id: str,
    chat_data: ChatUpdate,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Update a chat (rename)."""
    
    chat = db.query(Chat).join(Patient).filter(
        Chat.id == chat_id,
        Patient.doctor_id == current_doctor.id
    ).first()
    
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    # Update name
    if chat_data.name is not None:
        chat.name = chat_data.name
    
    chat.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(chat)
    
    return ChatResponse.model_validate(chat)


@router.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(
    chat_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Delete a chat and all associated messages."""
    
    chat = db.query(Chat).join(Patient).filter(
        Chat.id == chat_id,
        Patient.doctor_id == current_doctor.id
    ).first()
    
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    db.delete(chat)
    db.commit()
    
    return None

