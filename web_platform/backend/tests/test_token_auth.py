"""
Token Authentication Tests

Test that JWT tokens work properly across all endpoints.
"""

import pytest
from datetime import datetime, timedelta


def test_token_format(client, test_doctor):
    """Test that token is returned in correct format."""
    response = client.post(
        "/api/auth/login",
        json={"name": "Test Doctor", "password": "testpassword123"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check token response structure
    assert "access_token" in data
    assert "token_type" in data
    assert "doctor" in data
    assert data["token_type"] == "bearer"
    
    # Token should be a non-empty string
    assert isinstance(data["access_token"], str)
    assert len(data["access_token"]) > 20


def test_token_works_for_protected_endpoints(client, auth_headers):
    """Test that valid token grants access to protected endpoints."""
    # Test multiple endpoints
    endpoints = [
        ("/api/auth/me", "get"),
        ("/api/patients", "get"),
        ("/api/tools", "get"),
        ("/api/questions", "get"),
    ]
    
    for endpoint, method in endpoints:
        if method == "get":
            response = client.get(endpoint, headers=auth_headers)
        else:
            response = client.post(endpoint, headers=auth_headers)
        
        # Should not be 401 Unauthorized
        assert response.status_code != 401, f"{endpoint} failed with valid token"


def test_invalid_token_rejected(client):
    """Test that invalid tokens are rejected."""
    invalid_headers = {"Authorization": "Bearer invalid_token_here"}
    
    response = client.get("/api/auth/me", headers=invalid_headers)
    assert response.status_code == 401


def test_missing_token_rejected(client):
    """Test that requests without token are rejected."""
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_malformed_auth_header_rejected(client):
    """Test that malformed auth headers are rejected."""
    # Missing "Bearer" prefix
    bad_headers = {"Authorization": "just_a_token"}
    response = client.get("/api/auth/me", headers=bad_headers)
    assert response.status_code == 401


def test_token_contains_doctor_info(client, test_doctor):
    """Test that token response includes doctor information."""
    response = client.post(
        "/api/auth/login",
        json={"name": "Test Doctor", "password": "testpassword123"}
    )
    data = response.json()
    
    assert "doctor" in data
    assert data["doctor"]["name"] == "Test Doctor"
    assert data["doctor"]["id"] == test_doctor.id
    assert "password" not in data["doctor"]


def test_token_persists_across_requests(client, test_doctor):
    """Test that same token works for multiple requests."""
    # Login
    response = client.post(
        "/api/auth/login",
        json={"name": "Test Doctor", "password": "testpassword123"}
    )
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Make multiple requests with same token
    for _ in range(3):
        response = client.get("/api/auth/me", headers=headers)
        assert response.status_code == 200
        assert response.json()["name"] == "Test Doctor"


def test_different_doctors_different_tokens(client, test_doctor, db_session):
    """Test that different doctors get different tokens."""
    from app.models import Doctor
    from app.utils.security import get_password_hash
    
    # Create second doctor
    doctor2 = Doctor(
        name="Second Doctor",
        password_hash=get_password_hash("password456")
    )
    db_session.add(doctor2)
    db_session.commit()
    
    # Get tokens for both
    response1 = client.post(
        "/api/auth/login",
        json={"name": "Test Doctor", "password": "testpassword123"}
    )
    token1 = response1.json()["access_token"]
    
    response2 = client.post(
        "/api/auth/login",
        json={"name": "Second Doctor", "password": "password456"}
    )
    token2 = response2.json()["access_token"]
    
    # Tokens should be different
    assert token1 != token2
    
    # Each token should work for its doctor
    headers1 = {"Authorization": f"Bearer {token1}"}
    headers2 = {"Authorization": f"Bearer {token2}"}
    
    response1 = client.get("/api/auth/me", headers=headers1)
    response2 = client.get("/api/auth/me", headers=headers2)
    
    assert response1.json()["name"] == "Test Doctor"
    assert response2.json()["name"] == "Second Doctor"


def test_token_identifies_correct_doctor_for_resources(client, test_doctor, db_session):
    """Test that token correctly identifies doctor for resource access."""
    from app.models import Doctor, Patient
    from app.utils.security import get_password_hash
    
    # Create second doctor with their own patient
    doctor2 = Doctor(
        name="Second Doctor",
        password_hash=get_password_hash("password456")
    )
    db_session.add(doctor2)
    db_session.flush()
    
    patient2 = Patient(
        name="Doctor 2's Patient",
        doctor_id=doctor2.id
    )
    db_session.add(patient2)
    db_session.commit()
    
    # Login as first doctor
    response = client.post(
        "/api/auth/login",
        json={"name": "Test Doctor", "password": "testpassword123"}
    )
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    
    # First doctor should not see second doctor's patients
    response = client.get("/api/patients", headers=headers)
    patients = response.json()
    
    # Should only see their own patients (if any)
    for patient in patients:
        assert patient["name"] != "Doctor 2's Patient"

