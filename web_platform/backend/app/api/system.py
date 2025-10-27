"""
System API Endpoints

Handles system-level operations like API secret validation.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..config import settings
from ..utils.logging_config import logger

router = APIRouter()


class ValidateSecretRequest(BaseModel):
    """Request to validate API secret."""
    secret: str


class ValidateSecretResponse(BaseModel):
    """Response from API secret validation."""
    valid: bool
    message: str


@router.post("/api/system/validate-secret", response_model=ValidateSecretResponse)
async def validate_secret(request: ValidateSecretRequest):
    """
    Validate an API secret key.
    
    This endpoint is public (no API secret required) to allow
    users to validate their secret before making other requests.
    """
    if not settings.REQUIRE_API_SECRET:
        return ValidateSecretResponse(
            valid=True,
            message="API secret validation is disabled"
        )
    
    is_valid = request.secret == settings.API_SECRET_KEY
    
    if is_valid:
        logger.info("✓ API secret validated successfully")
        return ValidateSecretResponse(
            valid=True,
            message="API secret is valid"
        )
    else:
        logger.warning("✗ Invalid API secret attempt")
        return ValidateSecretResponse(
            valid=False,
            message="Invalid API secret"
        )

