"""
Suggested Question Model

Database model for suggested questions.
"""

from sqlalchemy import Column, String, Boolean, Integer, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from ..database.base import Base


class SuggestedQuestion(Base):
    """Suggested question model."""

    __tablename__ = "suggested_questions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id = Column(String(36), ForeignKey("doctors.id", ondelete="CASCADE"), nullable=True)
    question = Column(String(512), nullable=False)
    is_default = Column(Boolean, default=False, nullable=False)
    display_order = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    doctor = relationship("Doctor", back_populates="questions")

    def __repr__(self):
        return f"<SuggestedQuestion(id={self.id}, question={self.question[:30]}...)>"
