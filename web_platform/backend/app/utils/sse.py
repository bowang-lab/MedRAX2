"""
Server-Sent Events Utilities

Helper functions for SSE streaming.
"""

import json
from typing import Any, Dict


def format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """
    Format data as an SSE event.

    Args:
        event_type: The event type (e.g., 'message_start', 'tool_start')
        data: The event data

    Returns:
        Formatted SSE event string
    """
    json_data = json.dumps(data)
    return f"event: {event_type}\ndata: {json_data}\n\n"


def create_sse_event(event_type: str, **kwargs) -> str:
    """
    Create an SSE event with the given type and data.

    Args:
        event_type: The event type
        **kwargs: Key-value pairs for the event data

    Returns:
        Formatted SSE event string
    """
    return format_sse_event(event_type, kwargs)
