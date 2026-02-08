"""
System API Routes

Routes for system-level operations like API secret validation.
"""

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from ..config import settings
from ..utils.logging_config import logger

router = APIRouter()


class ValidateSecretResponse(BaseModel):
    """Response for API secret validation."""

    valid: bool
    message: str


@router.post("/system/validate-secret", response_model=ValidateSecretResponse)
async def validate_api_secret(x_api_secret: str = Header(None, alias="X-API-Secret")):
    """
    Validate API secret key.

    This endpoint allows the frontend to validate the API secret
    before storing it locally.
    """
    if not settings.REQUIRE_API_SECRET:
        return ValidateSecretResponse(valid=True, message="API secret validation is disabled")

    if not x_api_secret:
        logger.warning("API secret validation failed - no secret provided")
        return ValidateSecretResponse(valid=False, message="API secret is required")

    if x_api_secret != settings.API_SECRET_KEY:
        logger.warning("API secret validation failed - invalid secret provided")
        return ValidateSecretResponse(valid=False, message="Invalid API secret")

    logger.info("API secret validated successfully")
    return ValidateSecretResponse(valid=True, message="API secret is valid")
