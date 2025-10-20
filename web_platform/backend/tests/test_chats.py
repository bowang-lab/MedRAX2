"""
Chat API Tests
"""

import pytest


def test_list_patient_chats_empty(client, auth_headers, test_patient):
    """Test listing chats when none exist."""
    response = client.get(
        f"/api/patients/{test_patient.id}/chats",
        headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json() == []


def test_list_patient_chats(client, auth_headers, test_patient, test_chat):
    """Test listing chats for a patient."""
    response = client.get(
        f"/api/patients/{test_patient.id}/chats",
        headers=auth_headers
    )
    assert response.status_code == 200
    chats = response.json()
    assert len(chats) == 1
    assert chats[0]["id"] == test_chat.id
    assert chats[0]["name"] == "Test Chat"


def test_create_chat(client, auth_headers, test_patient):
    """Test creating a chat."""
    response = client.post(
        f"/api/patients/{test_patient.id}/chats",
        json={},
        headers=auth_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert "name" in data
    assert data["patient_id"] == test_patient.id


def test_create_chat_with_name(client, auth_headers, test_patient):
    """Test creating a chat with custom name."""
    response = client.post(
        f"/api/patients/{test_patient.id}/chats",
        json={"name": "Custom Chat Name"},
        headers=auth_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Custom Chat Name"


def test_update_chat_name(client, auth_headers, test_chat):
    """Test updating chat name."""
    response = client.patch(
        f"/api/chats/{test_chat.id}",
        json={"name": "Updated Chat Name"},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Chat Name"


def test_delete_chat(client, auth_headers, test_chat):
    """Test deleting a chat."""
    response = client.delete(
        f"/api/chats/{test_chat.id}",
        headers=auth_headers
    )
    assert response.status_code == 204


def test_list_chats_nonexistent_patient(client, auth_headers):
    """Test listing chats for nonexistent patient."""
    response = client.get(
        "/api/patients/nonexistent-id/chats",
        headers=auth_headers
    )
    assert response.status_code == 404


def test_list_chats_unauthorized(client, test_patient):
    """Test listing chats without auth."""
    response = client.get(f"/api/patients/{test_patient.id}/chats")
    assert response.status_code == 401




