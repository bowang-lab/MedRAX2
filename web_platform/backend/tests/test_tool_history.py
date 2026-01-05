"""
Tests for Tool History API

Tests for tool execution history tracking and retrieval.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.message import Message
from app.models.tool_execution import ToolExecution, ToolExecutionResult


def test_get_chat_tool_history_empty(client: TestClient, auth_headers, test_chat):
    """Test getting tool history for a chat with no executions."""
    response = client.get(
        f"/api/chats/{test_chat.id}/tool-history",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    history = response.json()
    assert isinstance(history, list)
    assert len(history) == 0


def test_get_chat_tool_history_with_executions(
    client: TestClient,
    auth_headers,
    test_chat,
    db_session: Session
):
    """Test getting tool history with actual executions."""
    # Create a message
    message = Message(
        chat_id=test_chat.id,
        role="user",
        content="Test message",
        request_id="test-request-123"
    )
    db_session.add(message)
    db_session.flush()
    
    # Create tool executions
    execution1 = ToolExecution(
        message_id=message.id,
        request_id="test-request-123",
        tool_name="classifier",
        status="completed",
        image_paths=["image1.jpg", "image2.jpg"]
    )
    db_session.add(execution1)
    
    execution2 = ToolExecution(
        message_id=message.id,
        request_id="test-request-123",
        tool_name="segmentation",
        status="completed",
        image_paths=["image1.jpg"]
    )
    db_session.add(execution2)
    db_session.commit()
    
    # Get tool history
    response = client.get(
        f"/api/chats/{test_chat.id}/tool-history",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    history = response.json()
    assert len(history) == 2
    
    # Verify first execution
    assert history[0]['tool_name'] in ['classifier', 'segmentation']
    assert history[0]['request_id'] == "test-request-123"
    assert history[0]['status'] == "completed"
    assert 'image_paths' in history[0]


def test_get_message_tool_history(
    client: TestClient,
    auth_headers,
    test_chat,
    db_session: Session
):
    """Test getting tool history for a specific message."""
    # Create a message
    message = Message(
        chat_id=test_chat.id,
        role="user",
        content="Analyze this image",
        request_id="message-request-456"
    )
    db_session.add(message)
    db_session.flush()
    
    # Create tool executions for this message
    execution = ToolExecution(
        message_id=message.id,
        request_id="message-request-456",
        tool_name="vqa",
        status="completed",
        image_paths=["scan.jpg"]
    )
    db_session.add(execution)
    db_session.commit()
    
    # Get message tool history
    response = client.get(
        f"/api/messages/{message.id}/tool-history",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    history = response.json()
    assert len(history) == 1
    assert history[0]['tool_name'] == "vqa"
    assert history[0]['message_id'] == message.id


def test_filter_tool_history_by_request(
    client: TestClient,
    auth_headers,
    test_chat,
    db_session: Session
):
    """Test filtering tool history by request ID."""
    # Create two messages with different request IDs
    message1 = Message(
        chat_id=test_chat.id,
        role="user",
        content="First message",
        request_id="request-1"
    )
    message2 = Message(
        chat_id=test_chat.id,
        role="user",
        content="Second message",
        request_id="request-2"
    )
    db_session.add_all([message1, message2])
    db_session.flush()
    
    # Create executions for different requests
    exec1 = ToolExecution(
        message_id=message1.id,
        request_id="request-1",
        tool_name="classifier",
        status="completed"
    )
    exec2 = ToolExecution(
        message_id=message2.id,
        request_id="request-2",
        tool_name="classifier",
        status="completed"
    )
    db_session.add_all([exec1, exec2])
    db_session.commit()
    
    # Filter by request-1
    response = client.get(
        f"/api/chats/{test_chat.id}/tool-history?filter_by_request=request-1",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    history = response.json()
    assert len(history) == 1
    assert history[0]['request_id'] == "request-1"


def test_filter_tool_history_by_tool_name(
    client: TestClient,
    auth_headers,
    test_chat,
    db_session: Session
):
    """Test filtering tool history by tool name."""
    message = Message(
        chat_id=test_chat.id,
        role="user",
        content="Test message",
        request_id="request-multi"
    )
    db_session.add(message)
    db_session.flush()
    
    # Create multiple tool executions
    exec1 = ToolExecution(
        message_id=message.id,
        request_id="request-multi",
        tool_name="classifier",
        status="completed"
    )
    exec2 = ToolExecution(
        message_id=message.id,
        request_id="request-multi",
        tool_name="segmentation",
        status="completed"
    )
    exec3 = ToolExecution(
        message_id=message.id,
        request_id="request-multi",
        tool_name="classifier",
        status="completed"
    )
    db_session.add_all([exec1, exec2, exec3])
    db_session.commit()
    
    # Filter by tool name
    response = client.get(
        f"/api/chats/{test_chat.id}/tool-history?filter_by_tool=classifier",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    history = response.json()
    assert len(history) == 2
    assert all(h['tool_name'] == 'classifier' for h in history)


def test_latest_only_tool_history(
    client: TestClient,
    auth_headers,
    test_chat,
    db_session: Session
):
    """Test getting only latest execution per tool."""
    message = Message(
        chat_id=test_chat.id,
        role="user",
        content="Test",
        request_id="latest-test"
    )
    db_session.add(message)
    db_session.flush()
    
    # Create multiple executions of same tool
    exec1 = ToolExecution(
        message_id=message.id,
        request_id="latest-test",
        tool_name="classifier",
        status="completed"
    )
    db_session.add(exec1)
    db_session.flush()
    
    # Slightly later execution
    exec2 = ToolExecution(
        message_id=message.id,
        request_id="latest-test",
        tool_name="classifier",
        status="completed"
    )
    db_session.add(exec2)
    db_session.commit()
    
    # Get latest only
    response = client.get(
        f"/api/chats/{test_chat.id}/tool-history?latest_only=true",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    history = response.json()
    # Should only get one execution (the latest)
    assert len(history) == 1


def test_get_tool_execution_details(
    client: TestClient,
    auth_headers,
    test_chat,
    db_session: Session
):
    """Test getting detailed information about a specific execution."""
    message = Message(
        chat_id=test_chat.id,
        role="user",
        content="Test",
        request_id="detail-test"
    )
    db_session.add(message)
    db_session.flush()
    
    execution = ToolExecution(
        message_id=message.id,
        request_id="detail-test",
        tool_name="classifier",
        status="completed",
        image_paths=["test.jpg"]
    )
    db_session.add(execution)
    db_session.flush()
    
    # Add result
    result = ToolExecutionResult(
        execution_id=execution.id,
        result_data={"prediction": "pneumonia", "confidence": 0.95},
        result_metadata={"model": "densenet121"}
    )
    db_session.add(result)
    db_session.commit()
    
    # Get execution details
    response = client.get(
        f"/api/tool-executions/{execution.id}",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    details = response.json()
    assert details['id'] == execution.id
    assert details['tool_name'] == "classifier"
    assert details['status'] == "completed"
    assert details['image_paths'] == ["test.jpg"]


def test_tool_history_unauthorized(client: TestClient, test_chat):
    """Test that tool history endpoints require authentication."""
    response = client.get(f"/api/chats/{test_chat.id}/tool-history")
    assert response.status_code == 401


def test_tool_history_nonexistent_chat(client: TestClient, auth_headers):
    """Test getting tool history for nonexistent chat."""
    response = client.get(
        "/api/chats/nonexistent-chat-id/tool-history",
        headers=auth_headers
    )
    assert response.status_code == 404


def test_message_tool_history_nonexistent_message(client: TestClient, auth_headers):
    """Test getting tool history for nonexistent message."""
    response = client.get(
        "/api/messages/nonexistent-message-id/tool-history",
        headers=auth_headers
    )
    assert response.status_code == 404


def test_image_paths_tracking(
    client: TestClient,
    auth_headers,
    test_chat,
    db_session: Session
):
    """Test that image paths are properly tracked in executions."""
    message = Message(
        chat_id=test_chat.id,
        role="user",
        content="Analyze multiple images",
        request_id="multi-image"
    )
    db_session.add(message)
    db_session.flush()
    
    # Create execution with multiple image paths
    execution = ToolExecution(
        message_id=message.id,
        request_id="multi-image",
        tool_name="classifier",
        status="completed",
        image_paths=["image1.jpg", "image2.jpg", "image3.jpg"]
    )
    db_session.add(execution)
    db_session.commit()
    
    # Get tool history
    response = client.get(
        f"/api/chats/{test_chat.id}/tool-history",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    history = response.json()
    assert len(history) == 1
    assert len(history[0]['image_paths']) == 3
    assert "image2.jpg" in history[0]['image_paths']

