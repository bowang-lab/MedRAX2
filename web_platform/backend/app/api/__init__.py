"""
API Routes Package

All API endpoints organized by resource.
"""

from fastapi import APIRouter

from . import auth, patients, chats, messages, scans, tools, tools_sse, questions, tool_history, memory, system

# Import tool testing routes (development only)
try:
    from . import tool_testing
    TOOL_TESTING_AVAILABLE = True
except ImportError:
    TOOL_TESTING_AVAILABLE = False

# Main API router
api_router = APIRouter(prefix="/api")

# Include all routers
# System endpoints (no prefix - routes include full paths like /api/system/validate-secret)
api_router.include_router(system.router, tags=["system"])
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(patients.router, prefix="/patients", tags=["patients"])
api_router.include_router(chats.router, tags=["chats"])  # No prefix - routes include full paths
api_router.include_router(messages.router, tags=["messages"])  # No prefix - routes include full paths
api_router.include_router(scans.router, tags=["scans"])  # No prefix - routes include full paths
api_router.include_router(tools.router, prefix="/tools", tags=["tools"])
api_router.include_router(tools_sse.router, prefix="/tools", tags=["tools-sse"])  # SSE for tool loading
api_router.include_router(questions.router, prefix="/questions", tags=["questions"])
api_router.include_router(tool_history.router, tags=["tool-history"])  # No prefix - routes include full paths
api_router.include_router(memory.router, tags=["memory"])

# Include tool testing endpoints (development only)
if TOOL_TESTING_AVAILABLE:
    api_router.include_router(tool_testing.router, prefix="/test", tags=["tool-testing"])
