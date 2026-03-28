"""
MedRAX Backend Main Application

FastAPI application with all routes, middleware, and configuration.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pathlib import Path
import time

from .config import (
    settings,
    resolve_medrax_data_root,
    resolve_upload_dir,
    resolve_generated_dir,
)
import os
from .api import api_router
from .services.tool_manager import tool_manager
from .database import engine, Base
from .utils.logging_config import logger


# Custom JSONResponse that uses aliases (camelCase) for Pydantic models
class CamelCaseJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        # Use by_alias=True to convert snake_case to camelCase
        return super().render(jsonable_encoder(content, by_alias=True))


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=CamelCaseJSONResponse,  # Use camelCase for all responses
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
        "/health",  # Health check
        "/docs",  # API documentation
        "/redoc",  # ReDoc documentation
        "/openapi.json",  # OpenAPI schema
        "/api/system/validate-secret",  # API secret validation endpoint
        "/",  # Root endpoint
    ]

    # MedRAX static tree: /medrax/uploads/, /medrax/generated/
    if request.url.path.startswith("/medrax/"):
        return await call_next(request)

    # Legacy aliases (same directories as under medrax_data/)
    if request.url.path.startswith("/uploads/"):
        return await call_next(request)

    if request.url.path.startswith("/temp/"):
        return await call_next(request)

    # Allow /api/test/ endpoints for local tool testing (development only)
    if settings.DEBUG and request.url.path.startswith("/api/test/"):
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
        logger.warning(
            f"🚫 Request blocked - Missing API secret: {request.method} {request.url.path} from "
            f"{request.client.host if request.client else 'unknown'}"
        )
        return JSONResponse(
            status_code=403,
            content={
                "detail": "API secret required. Include X-API-Secret header.",
                "error": "forbidden",
            },
        )

    if api_secret != settings.API_SECRET_KEY:
        logger.warning(
            f"🚫 Request blocked - Invalid API secret: {request.method} {request.url.path} from "
            f"{request.client.host if request.client else 'unknown'}"
        )
        return JSONResponse(
            status_code=403,
            content={
                "detail": "Invalid API secret.",
                "error": "forbidden",
            },
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
        ".cgi",
        ".php",
        ".jsp",
        ".asp",
        ".aspx",
        ".exe",
        "htaccess",
        "config",
        "admin",
        "login/",
        "web/",
        "platform-ui",
        "management",
        "cgi-bin",
        "webct",
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
        logger.info(
            f"← {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s"
        )
    elif response.status_code != 404:
        # Log if suspicious path somehow got a non-404 (security concern!)
        logger.warning(f"⚠️ Suspicious path got {response.status_code}: {request.url.path}")

    return response


def response_would_be_404(path: str) -> bool:
    """Check if a path would likely result in 404."""
    # API routes and valid endpoints
    valid_prefixes = [
        "/api/",
        "/docs",
        "/redoc",
        "/health",
        "/medrax/",
        "/uploads/",
        "/temp/",
    ]
    if path == "/" or any(path.startswith(prefix) for prefix in valid_prefixes):
        return False
    return True


# Include API routes
app.include_router(api_router)

# Single on-disk root: medrax_data/uploads + medrax_data/generated
medrax_data_root = resolve_medrax_data_root()
medrax_data_root.mkdir(parents=True, exist_ok=True)
uploads_path = resolve_upload_dir()
uploads_path.mkdir(parents=True, exist_ok=True)
generated_path = resolve_generated_dir()
generated_path.mkdir(parents=True, exist_ok=True)

# Canonical URL prefix: /medrax/uploads/..., /medrax/generated/...
app.mount("/medrax", StaticFiles(directory=str(medrax_data_root)), name="medrax")

# Legacy URL prefixes (same folders on disk)
app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")
app.mount("/temp", StaticFiles(directory=str(generated_path)), name="temp")


@app.on_event("startup")
async def startup_event():
    """Initialize database and perform startup tasks."""
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    logger.info(f"🚀 {settings.APP_NAME} started successfully!")
    logger.info(f"📚 API documentation: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"🗄️  Database: {settings.DATABASE_URL}")
    logger.info(f"📂 MedRAX data root: {medrax_data_root}")
    logger.info(f"📂 Uploads: {uploads_path}")
    logger.info(f"📂 Generated (tools): {generated_path}")

    # Optional eager loading gated by env var to avoid long cold starts
    try:
        eager_env = os.getenv("EAGER_LOAD_TOOLS", "0").lower() in ("1", "true", "yes")
        if eager_env:
            exclude_ids = {"chest_xray_generator"}
            all_tools = tool_manager.get_all_tools()
            target_ids = [
                t["id"]
                for t in all_tools
                if t.get("status") in ("available", "unloaded") and t["id"] not in exclude_ids
            ]

            if target_ids:
                logger.info(
                    f"🔧 Eager-loading tools at startup (excluding: {', '.join(exclude_ids)}): {', '.join(target_ids)}"
                )
            else:
                logger.info("🔧 No tools eligible for eager loading at startup")

            # Sequential loading strategy to prevent GPU memory contention
            # when multiple large models are loaded simultaneously
            import threading

            def load_tools_sequentially():
                """Load tools sequentially to prevent GPU memory conflicts."""
                for i, tool_id in enumerate(target_ids):
                    tool = tool_manager.tools.get(tool_id)
                    if not tool or tool.status != "available":
                        logger.warning(f"⚠️ Tool {tool_id} not available for loading")
                        continue

                    logger.info(f"🔧 Loading tool {i+1}/{len(target_ids)}: {tool_id}")
                    try:
                        # Use the proper background loading method
                        # This will handle the loading state and threading properly
                        started = tool_manager.start_background_load(tool_id)

                        if not started:
                            logger.error(f"❌ Failed to start loading {tool_id}")
                            continue

                        # Wait for the tool to finish loading before moving to next
                        # This ensures sequential loading
                        import time

                        max_wait = 300  # 5 minutes max per tool
                        waited = 0
                        while waited < max_wait:
                            tool = tool_manager.tools.get(tool_id)
                            if tool.status == "loaded":
                                logger.info(f"✅ Successfully loaded: {tool_id}")
                                break
                            if tool.status == "error":
                                logger.error(f"❌ Failed to load {tool_id}: {tool.error_message}")
                                break
                            time.sleep(2)
                            waited += 2

                        if waited >= max_wait:
                            logger.warning(f"⏱️ Timeout waiting for {tool_id} to load")

                    except Exception as e:
                        logger.error(f"❌ Exception loading {tool_id}: {e}")

            # Start loading in background thread to avoid blocking server startup
            threading.Thread(target=load_tools_sequentially, daemon=True).start()
            logger.info(f"🔧 Started sequential loading of {len(target_ids)} tools in background")
        else:
            logger.info("🔧 Eager tool loading disabled (EAGER_LOAD_TOOLS=0). Server will start faster.")
    except Exception as e:
        logger.warning(f"Failed during eager-load phase: {e}")


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
    return {"name": settings.APP_NAME, "version": settings.APP_VERSION, "status": "running", "docs": "/docs"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
