"""
Doctor Schemas

Pydantic schemas for doctor-related operations.
"""

from pydantic import BaseModel, Field
from datetime import datetime


class DoctorBase(BaseModel):
    """Base doctor schema."""

    name: str = Field(..., min_length=1, max_length=255)


class DoctorCreate(DoctorBase):
    """Schema for doctor registration."""

    password: str = Field(..., min_length=6)


class DoctorLogin(BaseModel):
    """Schema for doctor login."""

    name: str
    password: str


class DoctorUpdate(BaseModel):
    """Schema for updating doctor profile."""

    name: str | None = None
    password: str | None = Field(None, min_length=6)


class DoctorResponse(DoctorBase):
    """Schema for doctor response."""

    id: str
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Schema for authentication token response."""

    access_token: str
    token_type: str = "bearer"
    doctor: DoctorResponse
