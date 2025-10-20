"""
FastAPI Dependencies

Dependency injection for authentication and database access.
"""

from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from typing import Optional

from .database import get_db
from .models import Doctor
from .utils.security import decode_access_token


async def get_current_doctor(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
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




