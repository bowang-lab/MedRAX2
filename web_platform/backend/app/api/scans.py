"""
Scan API Routes

Endpoints for medical image/scan management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List

from ..database import get_db
from ..models import Doctor, Patient, Chat, Scan
from ..schemas.scan import ScanResponse
from ..dependencies import get_current_doctor
from ..utils.file_utils import save_upload_file, delete_file, is_allowed_file, get_file_extension

router = APIRouter()


@router.get("/patients/{patient_id}/scans", response_model=List[ScanResponse])
def get_patient_scans(
    patient_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Get all scans for a patient across all chats."""
    
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
    
    # Get all scans from all chats for this patient
    scans = db.query(Scan).join(Chat).filter(Chat.patient_id == patient_id).order_by(Scan.uploaded_at.desc()).all()
    
    return [ScanResponse.model_validate(scan) for scan in scans]


@router.get("/chats/{chat_id}/scans", response_model=List[ScanResponse])
def get_chat_scans(
    chat_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Get all scans for a specific chat."""
    
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
    
    scans = db.query(Scan).filter(Scan.chat_id == chat_id).order_by(Scan.uploaded_at.desc()).all()
    
    return [ScanResponse.model_validate(scan) for scan in scans]


@router.post("/chats/{chat_id}/scans", response_model=List[ScanResponse], status_code=status.HTTP_201_CREATED)
async def upload_scans(
    chat_id: str,
    files: List[UploadFile] = File(...),
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Upload one or more scans to a chat."""
    
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
    
    uploaded_scans = []
    
    for file in files:
        # Validate file type
        if not is_allowed_file(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed: {file.filename}"
            )
        
        # Save file
        file_path, display_path = await save_upload_file(file, f"chats/{chat_id}")
        
        # Create scan record
        scan = Scan(
            chat_id=chat_id,
            file_path=file_path,
            display_path=display_path,
            file_type=get_file_extension(file.filename),
            file_size=file.size or 0
        )
        db.add(scan)
        uploaded_scans.append(scan)
    
    db.commit()
    
    # Refresh all scans
    for scan in uploaded_scans:
        db.refresh(scan)
    
    return [ScanResponse.model_validate(scan) for scan in uploaded_scans]


@router.delete("/{scan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scan(
    scan_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Delete a scan."""
    
    # Verify scan belongs to doctor
    scan = db.query(Scan).join(Chat).join(Patient).filter(
        Scan.id == scan_id,
        Patient.doctor_id == current_doctor.id
    ).first()
    
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found"
        )
    
    # Delete file from disk
    delete_file(scan.file_path)
    
    # Delete database record
    db.delete(scan)
    db.commit()
    
    return None




