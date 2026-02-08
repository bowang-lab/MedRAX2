"""
Question Schemas

Pydantic schemas for suggested questions.
"""

from pydantic import BaseModel, Field
from datetime import datetime


class QuestionBase(BaseModel):
    """Base question schema."""

    question: str = Field(..., min_length=1, max_length=512)
    display_order: int = 0


class QuestionCreate(QuestionBase):
    """Schema for creating a question."""

    pass


class QuestionResponse(QuestionBase):
    """Schema for question response."""

    id: str
    doctor_id: str | None
    is_default: bool
    created_at: datetime

    class Config:
        from_attributes = True
