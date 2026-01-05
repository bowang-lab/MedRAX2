"""
Tests for Memory Management API

Tests for chat memory and context management endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.message import Message
from app.models.scan import Scan
from app.models.tool_execution import ToolExecution


def test_get_memory_stats_empty(client: TestClient, auth_headers, test_chat):
    """Test getting memory stats for a chat with no data."""
    response = client.get(
        f"/api/chats/{test_chat.id}/memory/stats",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    stats = response.json()
    assert stats['chat_id'] == test_chat.id
    assert stats['message_count'] == 0
    assert stats['scan_count'] == 0
    assert stats['tool_execution_count'] == 0
    assert stats['has_context'] == False


def test_get_memory_stats_with_data(
    client: TestClient,
    auth_headers,
    test_chat,
    db_session: Session
):
    """Test getting memory stats for a chat with messages and scans."""
    # Add messages
    message1 = Message(
        chat_id=test_chat.id,
        role="user",
        content="First message"
    )
    message2 = Message(
        chat_id=test_chat.id,
        role="assistant",
        content="First response"
    )
    db_session.add_all([message1, message2])
    db_session.flush()
    
    # Add scan
    scan = Scan(
        chat_id=test_chat.id,
        file_path="/test/scan.jpg",
        display_path="/test/scan.jpg",
        file_type="jpg",
        file_size=1024
    )
    db_session.add(scan)
    db_session.flush()
    
    # Add tool execution
    execution = ToolExecution(
        message_id=message1.id,
        tool_name="classifier",
        status="completed"
    )
    db_session.add(execution)
    db_session.commit()
    
    # Get memory stats
    response = client.get(
        f"/api/chats/{test_chat.id}/memory/stats",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    stats = response.json()
    assert stats['message_count'] == 2
    assert stats['scan_count'] == 1
    assert stats['tool_execution_count'] == 1
    assert stats['has_context'] == True


def test_clear_chat_memory(client: TestClient, auth_headers, test_chat):
    """Test clearing chat memory."""
    response = client.post(
        f"/api/chats/{test_chat.id}/memory/clear",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data['success'] == True
    assert data['chat_id'] == test_chat.id
    assert 'Memory cleared' in data['message']


def test_system_memory_cleanup(client: TestClient, auth_headers):
    """Test system-wide memory cleanup."""
    response = client.post(
        "/api/system/memory/cleanup",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data['success'] == True
    assert 'stats' in data


def test_memory_stats_nonexistent_chat(client: TestClient, auth_headers):
    """Test getting memory stats for nonexistent chat."""
    response = client.get(
        "/api/chats/nonexistent-chat-id/memory/stats",
        headers=auth_headers
    )
    assert response.status_code == 404


def test_clear_memory_nonexistent_chat(client: TestClient, auth_headers):
    """Test clearing memory for nonexistent chat."""
    response = client.post(
        "/api/chats/nonexistent-chat-id/memory/clear",
        headers=auth_headers
    )
    assert response.status_code == 404


def test_memory_stats_unauthorized(client: TestClient, test_chat):
    """Test that memory stats endpoint requires authentication."""
    response = client.get(f"/api/chats/{test_chat.id}/memory/stats")
    assert response.status_code == 401


def test_clear_memory_unauthorized(client: TestClient, test_chat):
    """Test that clear memory endpoint requires authentication."""
    response = client.post(f"/api/chats/{test_chat.id}/memory/clear")
    assert response.status_code == 401


def test_system_cleanup_unauthorized(client: TestClient):
    """Test that system cleanup requires authentication."""
    response = client.post("/api/system/memory/cleanup")
    assert response.status_code == 401


def test_memory_stats_tracks_tool_executions_correctly(
    client: TestClient,
    auth_headers,
    test_chat,
    db_session: Session
):
    """Test that memory stats correctly counts tool executions across messages."""
    # Create multiple messages with tool executions
    message1 = Message(chat_id=test_chat.id, role="user", content="Test 1")
    message2 = Message(chat_id=test_chat.id, role="user", content="Test 2")
    db_session.add_all([message1, message2])
    db_session.flush()
    
    # Add tool executions
    exec1 = ToolExecution(message_id=message1.id, tool_name="tool1", status="completed")
    exec2 = ToolExecution(message_id=message1.id, tool_name="tool2", status="completed")
    exec3 = ToolExecution(message_id=message2.id, tool_name="tool1", status="completed")
    db_session.add_all([exec1, exec2, exec3])
    db_session.commit()
    
    # Get stats
    response = client.get(
        f"/api/chats/{test_chat.id}/memory/stats",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    stats = response.json()
    assert stats['tool_execution_count'] == 3
    assert stats['message_count'] == 2


def test_memory_stats_multiple_chats_isolated(
    client: TestClient,
    auth_headers,
    test_patient,
    db_session: Session
):
    """Test that memory stats are properly isolated between chats."""
    from app.models.chat import Chat
    
    # Create two chats
    chat1 = Chat(patient_id=test_patient.id, name="Chat 1")
    chat2 = Chat(patient_id=test_patient.id, name="Chat 2")
    db_session.add_all([chat1, chat2])
    db_session.flush()
    
    # Add messages to chat1
    msg1 = Message(chat_id=chat1.id, role="user", content="Chat 1 message")
    db_session.add(msg1)
    
    # Add messages to chat2
    msg2a = Message(chat_id=chat2.id, role="user", content="Chat 2 message 1")
    msg2b = Message(chat_id=chat2.id, role="user", content="Chat 2 message 2")
    db_session.add_all([msg2a, msg2b])
    db_session.commit()
    
    # Check chat1 stats
    response1 = client.get(f"/api/chats/{chat1.id}/memory/stats", headers=auth_headers)
    assert response1.status_code == 200
    assert response1.json()['message_count'] == 1
    
    # Check chat2 stats
    response2 = client.get(f"/api/chats/{chat2.id}/memory/stats", headers=auth_headers)
    assert response2.status_code == 200
    assert response2.json()['message_count'] == 2

