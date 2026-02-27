"""
Patient Schemas

Pydantic schemas for patient-related operations.
"""

from pydantic import BaseModel, Field
from datetime import datetime


class PatientBase(BaseModel):
    """Base patient schema."""

    name: str | None = Field(None, max_length=255)


class PatientCreate(PatientBase):
    """Schema for creating a patient."""

    pass


class PatientUpdate(PatientBase):
    """Schema for updating a patient."""

    pass


class PatientResponse(PatientBase):
    """Schema for patient response."""

    id: str
    doctor_id: str
    created_at: datetime
    last_activity_at: datetime

    class Config:
        from_attributes = True


class PatientWithStats(PatientResponse):
    """Schema for patient with statistics."""

    total_chats: int = 0
    total_scans: int = 0
