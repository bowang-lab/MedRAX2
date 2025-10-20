"""
Message API Tests
"""

import pytest


def test_list_messages_empty(client, auth_headers, test_chat):
    """Test listing messages when none exist."""
    response = client.get(
        f"/api/chats/{test_chat.id}/messages",
        headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json() == []


def test_list_messages(client, auth_headers, test_chat, test_message):
    """Test listing messages in a chat."""
    response = client.get(
        f"/api/chats/{test_chat.id}/messages",
        headers=auth_headers
    )
    assert response.status_code == 200
    messages = response.json()
    assert len(messages) == 1
    assert messages[0]["id"] == test_message.id
    assert messages[0]["content"] == "Test message"


def test_create_message(client, auth_headers, test_chat):
    """Test creating a message."""
    response = client.post(
        f"/api/chats/{test_chat.id}/messages",
        json={"content": "Hello, doctor!"},
        headers=auth_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["content"] == "Hello, doctor!"
    assert data["role"] == "user"


def test_list_messages_nonexistent_chat(client, auth_headers):
    """Test listing messages for nonexistent chat."""
    response = client.get(
        "/api/chats/nonexistent-id/messages",
        headers=auth_headers
    )
    assert response.status_code == 404


def test_list_messages_unauthorized(client, test_chat):
    """Test listing messages without auth."""
    response = client.get(f"/api/chats/{test_chat.id}/messages")
    assert response.status_code == 401


def test_get_message_executions_empty(client, auth_headers, test_message):
    """Test getting tool executions when none exist."""
    response = client.get(
        f"/api/messages/{test_message.id}/executions",
        headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json() == []


def test_get_message_executions_unauthorized(client, test_message):
    """Test getting executions without auth."""
    response = client.get(f"/api/messages/{test_message.id}/executions")
    assert response.status_code == 401

