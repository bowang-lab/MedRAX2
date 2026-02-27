"""
Authentication API Routes

Endpoints for doctor registration, login, logout, and profile.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Doctor
from ..schemas.doctor import (
    DoctorCreate,
    DoctorLogin,
    DoctorResponse,
    DoctorUpdate,
    TokenResponse,
)
from ..utils.security import verify_password, get_password_hash, create_access_token
from ..dependencies import get_current_doctor

router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(doctor_data: DoctorCreate, db: Session = Depends(get_db)):
    """Register a new doctor account."""

    # Check if doctor with this name already exists
    existing = db.query(Doctor).filter(Doctor.name == doctor_data.name).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Doctor with this name already exists")

    # Create new doctor
    doctor = Doctor(name=doctor_data.name, password_hash=get_password_hash(doctor_data.password))
    db.add(doctor)
    db.commit()
    db.refresh(doctor)

    # Create access token
    access_token = create_access_token(data={"sub": doctor.id})

    return TokenResponse(access_token=access_token, token_type="bearer", doctor=DoctorResponse.model_validate(doctor))


@router.post("/login", response_model=TokenResponse)
def login(credentials: DoctorLogin, db: Session = Depends(get_db)):
    """Login with doctor credentials."""

    # Find doctor by name
    doctor = db.query(Doctor).filter(Doctor.name == credentials.name).first()
    if not doctor or not verify_password(credentials.password, doctor.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect name or password")

    # Create access token
    access_token = create_access_token(data={"sub": doctor.id})

    return TokenResponse(access_token=access_token, token_type="bearer", doctor=DoctorResponse.model_validate(doctor))


@router.post("/logout")
def logout():
    """Logout (token invalidation handled client-side)."""
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=DoctorResponse)
def get_me(current_doctor: Doctor = Depends(get_current_doctor)):
    """Get current doctor profile."""
    return DoctorResponse.model_validate(current_doctor)


@router.patch("/me", response_model=DoctorResponse)
def update_profile(
    update_data: DoctorUpdate,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Update current doctor profile."""

    # Update name if provided
    if update_data.name is not None:
        # Check if new name is already taken
        existing = db.query(Doctor).filter(Doctor.name == update_data.name, Doctor.id != current_doctor.id).first()
        if existing:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Name already taken")
        current_doctor.name = update_data.name

    # Update password if provided
    if update_data.password is not None:
        current_doctor.password_hash = get_password_hash(update_data.password)

    db.commit()
    db.refresh(current_doctor)

    return DoctorResponse.model_validate(current_doctor)
