"""
MedRAX Backend Main Application

FastAPI application with all routes, middleware, and configuration.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
import time

from .config import settings
from .api import api_router
from .services.tool_manager import tool_manager
from .database import engine, Base
from .utils.logging_config import logger

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Secret validation middleware (SECURITY LAYER)
@app.middleware("http")
async def validate_api_secret(request: Request, call_next):
    """
    Validate API secret key for all requests (except whitelisted public endpoints).
    This prevents unauthorized access even if someone gets network access.
    """
    # Allow CORS preflight requests (OPTIONS)
    if request.method == "OPTIONS":
        return await call_next(request)
    
    # Whitelist public endpoints that don't require API secret
    public_paths = [
        "/health",                      # Health check
        "/docs",                        # API documentation
        "/redoc",                       # ReDoc documentation
        "/openapi.json",                # OpenAPI schema
        "/api/system/validate-secret",  # API secret validation endpoint
        "/"                             # Root endpoint
    ]
    
    # Allow /uploads/ paths for serving medical images
    # Note: In production, consider using signed URLs or token-based auth
    if request.url.path.startswith("/uploads/"):
        return await call_next(request)
    
    # Allow SSE endpoints - EventSource doesn't support custom headers
    # These endpoints use JWT token in query string for authentication instead
    # SECURITY: Use exact path matching to prevent bypass attacks
    if request.url.path.startswith("/api/tools/") and request.url.path.endswith("/load-stream"):
        return await call_next(request)
    
    # Allow public endpoints without secret
    if request.url.path in public_paths:
        return await call_next(request)
    
    # If API secret requirement is disabled, allow all requests
    if not settings.REQUIRE_API_SECRET:
        return await call_next(request)
    
    # Validate API secret header
    api_secret = request.headers.get("X-API-Secret")
    
    if not api_secret:
        logger.warning(f"🚫 Request blocked - Missing API secret: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
        return JSONResponse(
            status_code=403,
            content={
                "detail": "API secret required. Include X-API-Secret header.",
                "error": "forbidden"
            }
        )
    
    if api_secret != settings.API_SECRET_KEY:
        logger.warning(f"🚫 Request blocked - Invalid API secret: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
        return JSONResponse(
            status_code=403,
            content={
                "detail": "Invalid API secret.",
                "error": "forbidden"
            }
        )
    
    # API secret is valid, proceed with request
    return await call_next(request)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their responses."""
    start_time = time.time()
    
    # Filter out common scanner/attack patterns to reduce log noise
    suspicious_patterns = [
        '.cgi', '.php', '.jsp', '.asp', '.aspx', '.exe',
        'htaccess', 'config', 'admin', 'login/', 'web/',
        'platform-ui', 'management', 'cgi-bin', 'webct'
    ]
    path = request.url.path.lower()
    is_suspicious = any(pattern in path for pattern in suspicious_patterns) and response_would_be_404(path)
    
    # Only log legitimate API requests, not scanner noise
    if not is_suspicious:
        logger.info(f"→ {request.method} {request.url.path}")
        logger.debug(f"  Headers: {dict(request.headers)}")
    
    # Process request
    response = await call_next(request)
    
    # Log response (skip 404s from scanners)
    process_time = time.time() - start_time
    if not is_suspicious:
        logger.info(f"← {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
    elif response.status_code != 404:
        # Log if suspicious path somehow got a non-404 (security concern!)
        logger.warning(f"⚠️ Suspicious path got {response.status_code}: {request.url.path}")
    
    return response

def response_would_be_404(path: str) -> bool:
    """Check if a path would likely result in 404."""
    # API routes and valid endpoints
    valid_prefixes = ['/api/', '/docs', '/redoc', '/health', '/uploads/']
    if path == '/' or any(path.startswith(prefix) for prefix in valid_prefixes):
        return False
    return True

# Include API routes
app.include_router(api_router)

# Mount static files for uploads
uploads_path = Path(settings.UPLOAD_DIR)
uploads_path.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")


@app.on_event("startup")
async def startup_event():
    """Initialize database and perform startup tasks."""
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    logger.info(f"🚀 {settings.APP_NAME} started successfully!")
    logger.info(f"📚 API documentation: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"🗄️  Database: {settings.DATABASE_URL}")
    logger.info(f"📂 Upload directory: {settings.UPLOAD_DIR}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup tasks on shutdown."""
    logger.info(f"👋 {settings.APP_NAME} shutting down...")
    try:
        tool_manager.shutdown()
    except Exception as e:
        logger.debug(f"Error during tool manager shutdown: {e}")


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

