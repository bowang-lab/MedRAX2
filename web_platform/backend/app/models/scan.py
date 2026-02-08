"""
Scan Model

Database model for medical images/scans.
"""

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from ..database.base import Base
from .message import MessageScan


class Scan(Base):
    """Medical scan/image model."""

    __tablename__ = "scans"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    file_path = Column(String(512), nullable=False)  # Actual file path on disk
    display_path = Column(String(512), nullable=False)  # URL path for frontend
    file_type = Column(String(50), nullable=False)  # 'dicom', 'jpg', 'png'
    file_size = Column(Integer, nullable=False)  # Size in bytes
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    chat = relationship("Chat", back_populates="scans")
    messages = relationship("Message", secondary=MessageScan, back_populates="attached_scans")

    def __repr__(self):
        return f"<Scan(id={self.id}, type={self.file_type})>"
