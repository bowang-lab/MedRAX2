"""
FastAPI Dependencies

Dependency injection for authentication and database access.
"""

from fastapi import Depends, HTTPException, status, Header, Query
from sqlalchemy.orm import Session
from typing import Optional

from .database import get_db
from .models import Doctor
from .utils.security import decode_access_token


async def get_current_doctor(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> Doctor:
    """
    Get the current authenticated doctor from the Authorization header.

    Args:
        authorization: Authorization header value
        db: Database session

    Returns:
        Current doctor

    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not authorization:
        raise credentials_exception

    # Extract token from "Bearer <token>" format
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise credentials_exception
    except ValueError:
        raise credentials_exception

    # Decode token
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    doctor_id: str = payload.get("sub")
    if doctor_id is None:
        raise credentials_exception

    # Get doctor from database
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if doctor is None:
        raise credentials_exception

    return doctor


async def get_current_doctor_sse(
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> Doctor:
    """
    Get the current authenticated doctor for SSE endpoints.

    EventSource doesn't support custom headers, so we accept token as query param.
    Falls back to Authorization header if query param not provided.

    Args:
        token: JWT token from query parameter
        authorization: Authorization header value (fallback)
        db: Database session

    Returns:
        Current doctor

    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Try query param first (for EventSource compatibility)
    jwt_token = None
    if token:
        jwt_token = token
    elif authorization:
        # Fall back to Authorization header
        try:
            scheme, jwt_token = authorization.split()
            if scheme.lower() != "bearer":
                raise credentials_exception
        except ValueError:
            raise credentials_exception
    else:
        raise credentials_exception

    # Decode token
    payload = decode_access_token(jwt_token)
    if payload is None:
        raise credentials_exception

    doctor_id: str = payload.get("sub")
    if doctor_id is None:
        raise credentials_exception

    # Get doctor from database
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if doctor is None:
        raise credentials_exception

    return doctor
