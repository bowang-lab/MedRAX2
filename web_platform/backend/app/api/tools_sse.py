"""
Tool Loading with Server-Sent Events (SSE)

Provides real-time progress updates during tool loading.
No polling needed - server pushes updates to client.
"""

import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from ..models import Doctor
from ..dependencies import get_current_doctor_sse
from ..services.tool_manager import tool_manager, ToolStatus
from ..utils.logging_config import logger

router = APIRouter()


async def tool_load_event_generator(tool_id: str, doctor_id: str) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events for tool loading progress.

    Yields events in SSE format:
    - data: {"status": "loading", "progress": 10, "message": "Downloading model..."}
    - data: {"status": "loaded", "message": "Tool loaded successfully"}
    - data: {"status": "error", "message": "Error message"}
    """
    try:
        # Get tool info
        tool = tool_manager.get_tool(tool_id)
        if not tool:
            yield f"data: {json.dumps({'status': 'error', 'message': 'Tool not found'})}\n\n"
            return

        # Check if already loaded
        if tool.status == ToolStatus.LOADED:
            yield f"data: {json.dumps({'status': 'loaded', 'message': 'Tool already loaded', 'tool': tool_manager._tool_to_dict(tool)})}\n\n"
            return

        # Check if currently loading
        if tool.status == ToolStatus.LOADING:
            yield f"data: {json.dumps({'status': 'loading', 'progress': 0, 'message': 'Tool is already loading...'})}\n\n"
            # Fall through to monitor progress
        else:
            # Initiate and start managed background loading
            started = tool_manager.start_background_load(tool_id)
            if not started:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Failed to start background load'})}\n\n"
                return
            yield f"data: {json.dumps({'status': 'loading', 'progress': 0, 'message': 'Starting tool loading...'})}\n\n"

        # Monitor loading progress
        last_status = None
        progress_messages = [
            "Initializing...",
            "Downloading model files...",
            "This may take several minutes for first-time download...",
            "Loading model into memory...",
            "Almost ready...",
        ]
        message_index = 0
        check_count = 0

        while True:
            await asyncio.sleep(2)  # Check every 2 seconds
            check_count += 1

            # Refresh tool status
            tool = tool_manager.get_tool(tool_id)
            if not tool:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Tool not found'})}\n\n"
                break

            current_status = tool.status

            # Status changed - send update
            if current_status != last_status:
                if current_status == ToolStatus.LOADED:
                    yield f"data: {json.dumps({'status': 'loaded', 'progress': 100, 'message': 'Tool loaded successfully!', 'tool': tool_manager._tool_to_dict(tool)})}\n\n"
                    break
                elif current_status == ToolStatus.ERROR:
                    error_msg = tool.error_message or "Failed to load tool"
                    yield f"data: {json.dumps({'status': 'error', 'message': error_msg})}\n\n"
                    break

                last_status = current_status

            # Send periodic progress updates (fake progress since we can't track real progress)
            if current_status == ToolStatus.LOADING:
                # Cycle through messages every ~10 seconds
                if check_count % 5 == 0 and message_index < len(progress_messages):
                    progress = min(10 + (message_index * 15), 90)  # Progress from 10% to 90%
                    yield f"data: {json.dumps({'status': 'loading', 'progress': progress, 'message': progress_messages[message_index]})}\n\n"
                    message_index += 1
                else:
                    # Send keepalive
                    yield f": keepalive\n\n"

            # Timeout after 30 minutes (1800 seconds / 2 seconds per check = 900 checks)
            if check_count > 900:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Tool loading timeout (30 minutes exceeded)'})}\n\n"
                break

    except asyncio.CancelledError:
        logger.info(f"Tool loading SSE cancelled for tool {tool_id} by doctor {doctor_id}")
        raise
    except Exception as e:
        logger.error(f"Error in tool loading SSE for {tool_id}: {e}")
        yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"


@router.get("/{tool_id}/load-stream")
async def stream_tool_loading(tool_id: str, current_doctor: Doctor = Depends(get_current_doctor_sse)):
    """
    Stream tool loading progress using Server-Sent Events.

    Note: Token can be passed as query parameter since EventSource doesn't support headers.

    Returns a stream of events with loading progress:
    - Client connects and receives real-time updates
    - No polling needed
    - Connection stays open until loading completes or fails

    Event format:
    ```
    data: {"status": "loading", "progress": 50, "message": "Loading model..."}
    data: {"status": "loaded", "message": "Done!", "tool": {...}}
    data: {"status": "error", "message": "Error details"}
    ```
    """
    logger.info(f"Doctor {current_doctor.id} starting SSE stream for tool loading: {tool_id}")

    return StreamingResponse(
        tool_load_event_generator(tool_id, current_doctor.id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
