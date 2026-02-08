"""
Formatting Utilities

Helper functions for formatting data.
"""

from datetime import datetime


def generate_chat_name() -> str:
    """
    Generate a chat name from current datetime.
    Format: "MM/DD/YYYY, H:MM AM/PM"

    Returns:
        Formatted chat name
    """
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %I:%M %p")
