"""
Message Model

Database model for chat messages.
"""

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from ..database.base import Base


# Junction table for message-scan many-to-many relationship
MessageScan = Table(
    "message_scans",
    Base.metadata,
    Column("message_id", String(36), ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True),
    Column("scan_id", String(36), ForeignKey("scans.id", ondelete="CASCADE"), primary_key=True),
)


class Message(Base):
    """Chat message model."""

    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    request_id = Column(String(36), nullable=True)  # Groups tool executions for this message

    # Relationships
    chat = relationship("Chat", back_populates="messages")
    attached_scans = relationship("Scan", secondary=MessageScan, back_populates="messages")
    tool_executions = relationship("ToolExecution", back_populates="message", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role})>"
