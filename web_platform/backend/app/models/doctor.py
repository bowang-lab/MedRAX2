"""
Doctor Model

Database model for doctor accounts.
"""

from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from ..database.base import Base


class Doctor(Base):
    """Doctor account model."""

    __tablename__ = "doctors"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    patients = relationship("Patient", back_populates="doctor", cascade="all, delete-orphan")
    questions = relationship("SuggestedQuestion", back_populates="doctor", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Doctor(id={self.id}, name={self.name})>"
