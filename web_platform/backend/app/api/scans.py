"""
Scan API Routes

Endpoints for medical image/scan management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import logging

from ..database import get_db
from ..models import Doctor, Patient, Chat, Scan
from ..schemas.scan import ScanResponse
from ..dependencies import get_current_doctor
from ..utils.file_utils import save_upload_file, delete_file, is_allowed_file, get_file_extension
from ..config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/patients/{patient_id}/scans", response_model=List[ScanResponse])
def get_patient_scans(
    patient_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get all scans for a patient across all chats."""

    # Verify patient belongs to doctor
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == current_doctor.id).first()

    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    # Get all scans from all chats for this patient
    scans = db.query(Scan).join(Chat).filter(Chat.patient_id == patient_id).order_by(Scan.uploaded_at.desc()).all()

    return [ScanResponse.model_validate(scan) for scan in scans]


@router.get("/chats/{chat_id}/scans", response_model=List[ScanResponse])
def get_chat_scans(
    chat_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get all scans for a specific chat."""

    # Verify chat belongs to doctor
    chat = db.query(Chat).join(Patient).filter(Chat.id == chat_id, Patient.doctor_id == current_doctor.id).first()

    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    scans = db.query(Scan).filter(Scan.chat_id == chat_id).order_by(Scan.uploaded_at.desc()).all()

    return [ScanResponse.model_validate(scan) for scan in scans]


@router.post("/chats/{chat_id}/scans", response_model=List[ScanResponse], status_code=status.HTTP_201_CREATED)
async def upload_scans(
    chat_id: str,
    files: List[UploadFile] = File(...),
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Upload one or more scans to a chat."""

    # Verify chat belongs to doctor
    chat = db.query(Chat).join(Patient).filter(Chat.id == chat_id, Patient.doctor_id == current_doctor.id).first()

    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    uploaded_scans = []
    saved_files = []  # Track saved files for cleanup on error

    try:
        for file in files:
            logger.info(f"Processing upload: filename={file.filename}, size={file.size}, content_type={file.content_type}")

            # Validate filename exists
            if not file.filename:
                logger.error("Upload failed: File has no filename")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must have a valid filename")

            # Validate file type
            if not is_allowed_file(file.filename):
                logger.warning(f"Upload rejected: Invalid file type for {file.filename}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"File type not allowed: {file.filename}"
                )

            # Validate file size (CRITICAL: prevent large file DoS attacks)
            if file.size and file.size > settings.MAX_UPLOAD_SIZE:
                max_size_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
                logger.warning(f"Upload rejected: File too large - {file.size / (1024 * 1024):.1f}MB")
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"File size ({file.size / (1024 * 1024):.1f}MB) exceeds maximum allowed size "
                        f"({max_size_mb:.0f}MB)"
                    ),
                )

            # Save file
            file_path, display_path = await save_upload_file(file, f"chats/{chat_id}")
            saved_files.append(file_path)  # Track for cleanup

            logger.info(f"File saved: path={file_path}, display_path={display_path}")

            # Validate that display_path was generated
            if not display_path:
                logger.error(f"Failed to generate display path for file: {file.filename}")
                raise ValueError(f"Failed to generate display path for file: {file.filename}")

            # Create scan record
            scan = Scan(
                chat_id=chat_id,
                file_path=file_path,
                display_path=display_path,
                file_type=get_file_extension(file.filename),
                file_size=file.size or 0,
            )
            db.add(scan)
            uploaded_scans.append(scan)
            logger.debug(f"Scan record created: id={scan.id}, display_path={scan.display_path}")

        db.commit()

        # Refresh all scans
        for scan in uploaded_scans:
            db.refresh(scan)
            logger.debug(f"Refreshed scan: id={scan.id}, display_path={scan.display_path}")

        # Validate and return scans
        responses = [ScanResponse.model_validate(scan) for scan in uploaded_scans]
        logger.info(f"Successfully uploaded {len(responses)} scan(s) to chat {chat_id}")

        # Debug: Log what we're actually returning
        for resp in responses:
            logger.debug(f"Returning scan: id={resp.id}, displayPath={resp.display_path}")

        return responses

    except Exception as e:
        # Rollback database changes
        db.rollback()

        # Clean up any files that were saved
        for file_path in saved_files:
            delete_file(file_path)

        # Re-raise the exception
        raise


@router.delete("/{scan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scan(
    scan_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Delete a scan."""

    # Verify scan belongs to doctor
    scan = (
        db.query(Scan).join(Chat).join(Patient).filter(Scan.id == scan_id, Patient.doctor_id == current_doctor.id).first()
    )

    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")

    # Delete file from disk
    delete_file(scan.file_path)

    # Delete database record
    db.delete(scan)
    db.commit()

    return None
"""
Scan API Routes

Endpoints for medical image/scan management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import logging

from ..database import get_db
from ..models import Doctor, Patient, Chat, Scan
from ..schemas.scan import ScanResponse
from ..dependencies import get_current_doctor
from ..utils.file_utils import save_upload_file, delete_file, is_allowed_file, get_file_extension
from ..config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


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
    saved_files = []  # Track saved files for cleanup on error
    
    try:
        for file in files:
            logger.info(f"Processing upload: filename={file.filename}, size={file.size}, content_type={file.content_type}")
            
            # Validate filename exists
            if not file.filename:
                logger.error("Upload failed: File has no filename")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File must have a valid filename"
                )
            
            # Validate file type
            if not is_allowed_file(file.filename):
                logger.warning(f"Upload rejected: Invalid file type for {file.filename}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File type not allowed: {file.filename}"
                )

            # Validate MIME type if provided
            allowed_mime_types = {
                "image/jpeg",
                "image/png",
                "image/gif",
                "application/dicom",
                "application/octet-stream",  # Some DICOM uploads come as octet-stream
            }
            if file.content_type and file.content_type not in allowed_mime_types:
                logger.warning(f"Upload rejected: Invalid content type {file.content_type} for {file.filename}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported content type: {file.content_type}"
                )
            
            # Validate file size (CRITICAL: prevent large file DoS attacks)
            if file.size and file.size > settings.MAX_UPLOAD_SIZE:
                max_size_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
                logger.warning(f"Upload rejected: File too large - {file.size / (1024 * 1024):.1f}MB")
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File size ({file.size / (1024 * 1024):.1f}MB) exceeds maximum allowed size ({max_size_mb:.0f}MB)"
                )
            
            # Save file (also returns actual byte size)
            try:
                file_path, display_path, file_size = await save_upload_file(file, f"chats/{chat_id}")
            except ValueError as e:
                logger.warning(f"Upload rejected during save: {e}")
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File exceeds maximum allowed size"
                )
            saved_files.append(file_path)  # Track for cleanup
            
            logger.info(f"File saved: path={file_path}, display_path={display_path}")
            
            # Validate that display_path was generated
            if not display_path:
                logger.error(f"Failed to generate display path for file: {file.filename}")
                raise ValueError(f"Failed to generate display path for file: {file.filename}")
            
            # Create scan record
            scan = Scan(
                chat_id=chat_id,
                file_path=file_path,
                display_path=display_path,
                file_type=get_file_extension(file.filename),
                file_size=file.size or file_size
            )
            db.add(scan)
            uploaded_scans.append(scan)
            logger.debug(f"Scan record created: id={scan.id}, display_path={scan.display_path}")
        
        db.commit()
        
        # Refresh all scans
        for scan in uploaded_scans:
            db.refresh(scan)
            logger.debug(f"Refreshed scan: id={scan.id}, display_path={scan.display_path}")
        
        # Validate and return scans
        responses = [ScanResponse.model_validate(scan) for scan in uploaded_scans]
        logger.info(f"Successfully uploaded {len(responses)} scan(s) to chat {chat_id}")
        
        # Debug: Log what we're actually returning
        for resp in responses:
            logger.debug(f"Returning scan: id={resp.id}, displayPath={resp.display_path}")
        
        return responses
    
    except Exception as e:
        # Rollback database changes
        db.rollback()
        
        # Clean up any files that were saved
        for file_path in saved_files:
            delete_file(file_path)
        
        # Re-raise the exception
        raise


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
    # Delete display file if it is a derived artifact (e.g., DICOM -> PNG)
    if scan.display_path and scan.display_path != scan.file_path:
        display_path = scan.display_path.lstrip("/")
        delete_file(display_path)
    
    # Delete database record
    db.delete(scan)
    db.commit()
    
    return None




