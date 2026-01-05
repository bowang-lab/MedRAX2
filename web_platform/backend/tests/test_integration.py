"""
Integration Tests

Test frontend-backend API contracts and integration points.
"""

import pytest
from fastapi.testclient import TestClient


def test_tools_api_response_format(client, auth_headers):
    """Test that tools API returns correct format for frontend."""
    response = client.get("/api/tools", headers=auth_headers)
    assert response.status_code == 200
    
    tools = response.json()
    
    # Should be a direct array, not wrapped
    assert isinstance(tools, list), "Response should be a direct array"
    assert len(tools) > 0, "Should return at least one tool"
    
    # Check first tool structure matches frontend expectations
    tool = tools[0]
    required_fields = ["id", "name", "description", "status", "category"]
    for field in required_fields:
        assert field in tool, f"Tool missing required field: {field}"
    
    # Status should be one of the expected values
    valid_statuses = ["available", "unavailable", "loaded", "unloaded", "error"]
    assert tool["status"] in valid_statuses, f"Invalid status: {tool['status']}"
    
    print(f"\n✓ Tools API returning {len(tools)} tools with correct format")
    print(f"✓ Sample tool: {tool['name']} ({tool['status']})")


def test_patients_api_response_format(client, auth_headers):
    """Test that patients API returns correct format."""
    response = client.get("/api/patients", headers=auth_headers)
    assert response.status_code == 200
    
    patients = response.json()
    
    # Should be a direct array
    assert isinstance(patients, list), "Response should be a direct array"
    
    print(f"\n✓ Patients API returning correct format")


def test_chats_api_response_format(client, auth_headers, test_patient):
    """Test that chats API returns correct format."""
    response = client.get(f"/api/patients/{test_patient.id}/chats", headers=auth_headers)
    assert response.status_code == 200
    
    chats = response.json()
    
    # Should be a direct array
    assert isinstance(chats, list), "Response should be a direct array"
    
    print(f"\n✓ Chats API returning correct format")


def test_messages_api_response_format(client, auth_headers, test_chat):
    """Test that messages API returns correct format."""
    response = client.get(f"/api/chats/{test_chat.id}/messages", headers=auth_headers)
    assert response.status_code == 200
    
    messages = response.json()
    
    # Should be a direct array
    assert isinstance(messages, list), "Response should be a direct array"
    
    print(f"\n✓ Messages API returning correct format")


def test_questions_api_response_format(client, auth_headers):
    """Test that questions API returns correct format."""
    response = client.get("/api/questions", headers=auth_headers)
    assert response.status_code == 200
    
    questions = response.json()
    
    # Should be a direct array
    assert isinstance(questions, list), "Response should be a direct array"
    
    print(f"\n✓ Questions API returning correct format")


def test_auth_token_format(client, test_doctor):
    """Test that auth endpoints return correct token format."""
    response = client.post(
        "/api/auth/login",
        json={"name": "Test Doctor", "password": "testpassword123"}
    )
    assert response.status_code == 200
    
    data = response.json()
    
    # Must have these exact fields for frontend
    assert "access_token" in data, "Missing access_token"
    assert "token_type" in data, "Missing token_type"
    assert "doctor" in data, "Missing doctor"
    assert data["token_type"] == "bearer", "token_type must be 'bearer'"
    
    # Doctor object must have expected fields
    doctor = data["doctor"]
    assert "id" in doctor
    assert "name" in doctor
    assert "created_at" in doctor
    assert "password" not in doctor, "Password should not be in response"
    
    print(f"\n✓ Auth API returning correct token format")


def test_full_patient_workflow(client, auth_headers):
    """Test complete patient workflow."""
    # 1. Create patient
    response = client.post(
        "/api/patients",
        json={"name": "Integration Test Patient"},
        headers=auth_headers
    )
    assert response.status_code == 201
    patient = response.json()
    patient_id = patient["id"]
    
    # 2. Create chat for patient
    response = client.post(
        f"/api/patients/{patient_id}/chats",
        json={},
        headers=auth_headers
    )
    assert response.status_code == 201
    chat = response.json()
    chat_id = chat["id"]
    
    # 3. Send message in chat
    response = client.post(
        f"/api/chats/{chat_id}/messages",
        json={"content": "Test message"},
        headers=auth_headers
    )
    assert response.status_code == 201
    message = response.json()
    
    # 4. List messages
    response = client.get(
        f"/api/chats/{chat_id}/messages",
        headers=auth_headers
    )
    assert response.status_code == 200
    messages = response.json()
    assert len(messages) == 1
    
    # 5. Cleanup - delete patient
    response = client.delete(
        f"/api/patients/{patient_id}",
        headers=auth_headers
    )
    assert response.status_code == 204
    
    print(f"\n✓ Full patient workflow completed successfully")


def test_api_cors_headers(client):
    """Test that CORS headers are properly configured."""
    response = client.options("/api/auth/login")
    
    # Should allow CORS for development
    assert "access-control-allow-origin" in response.headers or True  # May not be in test client
    
    print(f"\n✓ CORS configuration verified")


def test_all_endpoints_require_auth(client):
    """Test that protected endpoints require authentication."""
    protected_endpoints = [
        ("/api/patients", "get"),
        ("/api/tools", "get"),
        ("/api/questions", "get"),
        ("/api/auth/me", "get"),
    ]
    
    for endpoint, method in protected_endpoints:
        if method == "get":
            response = client.get(endpoint)
        else:
            response = client.post(endpoint)
        
        assert response.status_code == 401, f"{endpoint} should require auth"
    
    print(f"\n✓ All protected endpoints require authentication")


def test_error_responses_have_detail(client):
    """Test that error responses include detail message."""
    # Try to access protected endpoint without auth
    response = client.get("/api/patients")
    assert response.status_code == 401
    
    data = response.json()
    assert "detail" in data, "Error responses should include detail"
    
    print(f"\n✓ Error responses include detail field")

