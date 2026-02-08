"""
Pydantic Schemas

Request/response validation schemas for all entities.
"""

from .doctor import (
    DoctorBase,
    DoctorCreate,
    DoctorLogin,
    DoctorResponse,
    DoctorUpdate,
    TokenResponse,
)
from .patient import (
    PatientBase,
    PatientCreate,
    PatientUpdate,
    PatientResponse,
    PatientWithStats,
)
from .chat import (
    ChatBase,
    ChatCreate,
    ChatUpdate,
    ChatResponse,
)
from .message import (
    MessageBase,
    MessageCreate,
    MessageResponse,
    MessageWithDetails,
    StreamRequest,
)
from .scan import (
    ScanBase,
    ScanResponse,
)
from .tool import (
    ToolExecutionResponse,
    ToolExecutionLogResponse,
    ToolExecutionResultResponse,
    ToolExecutionDetailResponse,
    ToolInfo,
    ToolLoadRequest,
)
from .question import (
    QuestionBase,
    QuestionCreate,
    QuestionResponse,
)

__all__ = [
    # Doctor
    "DoctorBase",
    "DoctorCreate",
    "DoctorLogin",
    "DoctorResponse",
    "DoctorUpdate",
    "TokenResponse",
    # Patient
    "PatientBase",
    "PatientCreate",
    "PatientUpdate",
    "PatientResponse",
    "PatientWithStats",
    # Chat
    "ChatBase",
    "ChatCreate",
    "ChatUpdate",
    "ChatResponse",
    # Message
    "MessageBase",
    "MessageCreate",
    "MessageResponse",
    "MessageWithDetails",
    "StreamRequest",
    # Scan
    "ScanBase",
    "ScanResponse",
    # Tool
    "ToolExecutionResponse",
    "ToolExecutionLogResponse",
    "ToolExecutionResultResponse",
    "ToolExecutionDetailResponse",
    "ToolInfo",
    "ToolLoadRequest",
    # Question
    "QuestionBase",
    "QuestionCreate",
    "QuestionResponse",
]
