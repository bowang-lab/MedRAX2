"""
Tool Execution Models

Database models for AI tool execution tracking.
"""

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from ..database.base import Base


class ToolExecution(Base):
    """Tool execution tracking model."""

    __tablename__ = "tool_executions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id = Column(String(36), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    request_id = Column(String(36), nullable=True)  # Groups executions from same analysis request
    tool_name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)  # 'pending', 'running', 'completed', 'failed'
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    image_paths = Column(JSON, nullable=True)  # Track which images were used

    # Relationships
    message = relationship("Message", back_populates="tool_executions")
    logs = relationship(
        "ToolExecutionLog",
        back_populates="execution",
        cascade="all, delete-orphan",
        order_by="ToolExecutionLog.timestamp",
    )
    result = relationship("ToolExecutionResult", back_populates="execution", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ToolExecution(id={self.id}, tool={self.tool_name}, status={self.status})>"


class ToolExecutionLog(Base):
    """Tool execution log entry model."""

    __tablename__ = "tool_execution_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    execution_id = Column(String(36), ForeignKey("tool_executions.id", ondelete="CASCADE"), nullable=False)
    log_level = Column(String(50), nullable=False)  # 'info', 'warning', 'error'
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    execution = relationship("ToolExecution", back_populates="logs")

    def __repr__(self):
        return f"<ToolExecutionLog(id={self.id}, level={self.log_level})>"


class ToolExecutionResult(Base):
    """Tool execution result model."""

    __tablename__ = "tool_execution_results"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    execution_id = Column(String(36), ForeignKey("tool_executions.id", ondelete="CASCADE"), unique=True, nullable=False)
    result_data = Column(JSON, nullable=False)  # Flexible JSON structure for different tool types
    result_metadata = Column(JSON, nullable=True)  # Additional metadata about the result
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    execution = relationship("ToolExecution", back_populates="result")

    def __repr__(self):
        return f"<ToolExecutionResult(id={self.id})>"
