"""
Patient API Routes

Endpoints for patient CRUD operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from ..database import get_db
from ..models import Doctor, Patient, Chat, Scan
from ..schemas.patient import PatientCreate, PatientUpdate, PatientResponse, PatientWithStats
from ..dependencies import get_current_doctor
from datetime import datetime

router = APIRouter()


@router.get("", response_model=List[PatientWithStats])
def list_patients(current_doctor: Doctor = Depends(get_current_doctor), db: Session = Depends(get_db)):
    """List all patients for the current doctor with statistics."""

    patients = db.query(Patient).filter(Patient.doctor_id == current_doctor.id).all()

    # Add statistics to each patient
    patients_with_stats = []
    for patient in patients:
        total_chats = db.query(func.count(Chat.id)).filter(Chat.patient_id == patient.id).scalar() or 0
        total_scans = db.query(func.count(Scan.id)).join(Chat).filter(Chat.patient_id == patient.id).scalar() or 0

        patient_dict = PatientResponse.model_validate(patient).model_dump()
        patient_dict["total_chats"] = total_chats
        patient_dict["total_scans"] = total_scans
        patients_with_stats.append(PatientWithStats(**patient_dict))

    return patients_with_stats


@router.post("", response_model=PatientWithStats, status_code=status.HTTP_201_CREATED)
def create_patient(
    patient_data: PatientCreate,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Create a new patient."""

    patient = Patient(doctor_id=current_doctor.id, name=patient_data.name)
    db.add(patient)
    db.commit()
    db.refresh(patient)

    # Create first chat automatically (named "Initial Consultation") if not already present
    existing_initial = (
        db.query(Chat).filter(Chat.patient_id == patient.id, Chat.name == "Initial Consultation").first()
    )
    if not existing_initial:
        first_chat = Chat(patient_id=patient.id, name="Initial Consultation")
        db.add(first_chat)
        db.commit()

    # Return patient with stats
    total_chats = db.query(func.count(Chat.id)).filter(Chat.patient_id == patient.id).scalar() or 0
    total_scans = db.query(func.count(Scan.id)).join(Chat).filter(Chat.patient_id == patient.id).scalar() or 0
    patient_dict = PatientResponse.model_validate(patient).model_dump()
    patient_dict["total_chats"] = total_chats
    patient_dict["total_scans"] = total_scans
    return PatientWithStats(**patient_dict)


@router.get("/{patient_id}", response_model=PatientWithStats)
def get_patient(
    patient_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get a specific patient."""

    patient = (
        db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == current_doctor.id).first()
    )

    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    # Add statistics
    total_chats = db.query(func.count(Chat.id)).filter(Chat.patient_id == patient.id).scalar() or 0
    total_scans = db.query(func.count(Scan.id)).join(Chat).filter(Chat.patient_id == patient.id).scalar() or 0

    patient_dict = PatientResponse.model_validate(patient).model_dump()
    patient_dict["total_chats"] = total_chats
    patient_dict["total_scans"] = total_scans
    return PatientWithStats(**patient_dict)


@router.patch("/{patient_id}", response_model=PatientWithStats)
def update_patient(
    patient_id: str,
    patient_data: PatientUpdate,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Update a patient (rename)."""

    patient = (
        db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == current_doctor.id).first()
    )

    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    # Update name
    if patient_data.name is not None:
        patient.name = patient_data.name

    patient.last_activity_at = datetime.utcnow()
    db.commit()
    db.refresh(patient)

    # Add statistics
    total_chats = db.query(func.count(Chat.id)).filter(Chat.patient_id == patient.id).scalar() or 0
    total_scans = db.query(func.count(Scan.id)).join(Chat).filter(Chat.patient_id == patient.id).scalar() or 0

    patient_dict = PatientResponse.model_validate(patient).model_dump()
    patient_dict["total_chats"] = total_chats
    patient_dict["total_scans"] = total_scans
    return PatientWithStats(**patient_dict)


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(
    patient_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Delete a patient and all associated data including files on disk."""
    from ..models import Chat, Message, Scan, ToolExecution
    from ..utils.file_utils import delete_file

    patient = (
        db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == current_doctor.id).first()
    )

    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    # Clean up all scan files for this patient from disk
    scans = db.query(Scan).join(Chat).filter(Chat.patient_id == patient_id).all()
    for scan in scans:
        delete_file(scan.file_path)

    # Clean up all tool execution generated images for this patient
    tool_executions = (
        db.query(ToolExecution).join(Message).join(Chat).filter(Chat.patient_id == patient_id).all()
    )

    for execution in tool_executions:
        if execution.image_paths:
            for image_path in execution.image_paths:
                # Only delete generated images (in temp or output dirs), not input scans
                if isinstance(image_path, str) and ("temp" in image_path.lower() or "output" in image_path.lower()):
                    delete_file(image_path)

    # Delete database record (cascades to chats, messages, scans, tool executions)
    db.delete(patient)
    db.commit()

    return None
