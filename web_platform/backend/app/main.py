"""
MedRAX Backend Main Application

FastAPI application with all routes, middleware, and configuration.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import time

from .config import settings
from .api import api_router
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

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their responses."""
    start_time = time.time()
    
    # Log request
    logger.info(f"→ {request.method} {request.url.path}")
    logger.debug(f"  Headers: {dict(request.headers)}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"← {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
    
    return response

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

