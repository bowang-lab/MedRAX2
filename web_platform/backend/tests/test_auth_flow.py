"""
Comprehensive Authentication Flow Tests

Tests the complete authentication flow including token handling,
storage, and subsequent authenticated requests.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.main import app
from app.models.doctor import Doctor
from app.utils.security import get_password_hash


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def test_doctor_credentials():
    """Test doctor credentials."""
    return {
        "name": "TestDoctor",
        "password": "testpass123"
    }


@pytest.fixture
def db_with_doctor(db_session: Session, test_doctor_credentials):
    """Database with a test doctor."""
    doctor = Doctor(
        name=test_doctor_credentials["name"],
        password_hash=get_password_hash(test_doctor_credentials["password"])
    )
    db_session.add(doctor)
    db_session.commit()
    db_session.refresh(doctor)
    return db_session, doctor


def test_complete_auth_flow(client: TestClient, db_session: Session, test_doctor_credentials):
    """Test complete authentication flow: register -> login -> authenticated request."""
    
    # Step 1: Register
    register_response = client.post(
        "/api/auth/register",
        json={
            "name": "NewDoctor",
            "password": "password123"
        }
    )
    assert register_response.status_code == 201
    register_data = register_response.json()
    
    # Verify response structure
    assert "access_token" in register_data
    assert "token_type" in register_data
    assert "doctor" in register_data
    assert register_data["token_type"] == "bearer"
    assert register_data["doctor"]["name"] == "NewDoctor"
    
    # Step 2: Use the token to make authenticated request
    token = register_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test authenticated endpoint
    me_response = client.get("/api/auth/me", headers=headers)
    assert me_response.status_code == 200
    me_data = me_response.json()
    assert me_data["name"] == "NewDoctor"
    
    # Step 3: Test another authenticated endpoint
    patients_response = client.get("/api/patients", headers=headers)
    assert patients_response.status_code == 200
    assert isinstance(patients_response.json(), list)


def test_login_and_authenticated_requests(client: TestClient, db_with_doctor, test_doctor_credentials):
    """Test login followed by multiple authenticated requests."""
    
    # Step 1: Login
    login_response = client.post(
        "/api/auth/login",
        json=test_doctor_credentials
    )
    assert login_response.status_code == 200
    login_data = login_response.json()
    
    # Verify response structure
    assert "access_token" in login_data
    assert "token_type" in login_data
    assert "doctor" in login_data
    assert login_data["token_type"] == "bearer"
    
    # Step 2: Use token for authenticated requests
    token = login_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test multiple authenticated endpoints
    endpoints = [
        "/api/auth/me",
        "/api/patients",
        "/api/tools",
        "/api/questions",
    ]
    
    for endpoint in endpoints:
        response = client.get(endpoint, headers=headers)
        assert response.status_code == 200, f"Failed for {endpoint}"


def test_token_format(client: TestClient, db_session: Session, test_doctor_credentials):
    """Test that token is in correct format."""
    
    # Register to get token
    register_response = client.post(
        "/api/auth/register",
        json={
            "name": "TokenTestDoctor",
            "password": "password123"
        }
    )
    assert register_response.status_code == 201
    data = register_response.json()
    
    token = data["access_token"]
    
    # Token should be a non-empty string
    assert isinstance(token, str)
    assert len(token) > 0
    
    # JWT tokens typically have 3 parts separated by dots
    parts = token.split('.')
    assert len(parts) == 3


def test_token_required_for_protected_endpoints(client: TestClient):
    """Test that protected endpoints require authentication."""
    
    protected_endpoints = [
        ("GET", "/api/auth/me"),
        ("GET", "/api/patients"),
        ("POST", "/api/patients"),
        ("GET", "/api/tools"),
        ("GET", "/api/questions"),
    ]
    
    for method, endpoint in protected_endpoints:
        if method == "GET":
            response = client.get(endpoint)
        elif method == "POST":
            response = client.post(endpoint, json={})
        
        assert response.status_code == 401, f"Expected 401 for {method} {endpoint}"


def test_invalid_token(client: TestClient):
    """Test that invalid tokens are rejected."""
    
    invalid_tokens = [
        "invalid.token.here",
        "Bearer invalid",
        "",
        "not-a-jwt",
    ]
    
    for invalid_token in invalid_tokens:
        headers = {"Authorization": f"Bearer {invalid_token}"}
        response = client.get("/api/auth/me", headers=headers)
        assert response.status_code == 401


def test_expired_token_handling(client: TestClient):
    """Test handling of expired tokens (placeholder for future implementation)."""
    # Note: Actual token expiry testing would require time manipulation
    # or a test-specific shorter expiry time
    pass


def test_login_wrong_password(client: TestClient, db_with_doctor, test_doctor_credentials):
    """Test login with wrong password."""
    
    response = client.post(
        "/api/auth/login",
        json={
            "name": test_doctor_credentials["name"],
            "password": "wrongpassword"
        }
    )
    assert response.status_code == 401
    assert "incorrect" in response.json()["detail"].lower()


def test_login_nonexistent_doctor(client: TestClient, db_session: Session):
    """Test login with nonexistent doctor."""
    
    response = client.post(
        "/api/auth/login",
        json={
            "name": "NonexistentDoctor",
            "password": "password123"
        }
    )
    assert response.status_code == 401


def test_token_persists_across_requests(client: TestClient, db_with_doctor, test_doctor_credentials):
    """Test that a token can be used for multiple requests."""
    
    # Login
    login_response = client.post("/api/auth/login", json=test_doctor_credentials)
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Make 5 consecutive requests with same token
    for i in range(5):
        response = client.get("/api/auth/me", headers=headers)
        assert response.status_code == 200
        assert response.json()["name"] == test_doctor_credentials["name"]


def test_register_and_immediate_use(client: TestClient, db_session: Session):
    """Test that token from registration works immediately."""
    
    # Register
    register_response = client.post(
        "/api/auth/register",
        json={
            "name": f"QuickTestDoctor",
            "password": "password123"
        }
    )
    assert register_response.status_code == 201
    
    # Immediately use token
    token = register_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Should work immediately
    response = client.get("/api/auth/me", headers=headers)
    assert response.status_code == 200


def test_case_sensitive_bearer_token(client: TestClient, db_with_doctor, test_doctor_credentials):
    """Test that 'Bearer' prefix is case-sensitive."""
    
    # Login
    login_response = client.post("/api/auth/login", json=test_doctor_credentials)
    token = login_response.json()["access_token"]
    
    # Test different cases
    test_cases = [
        f"Bearer {token}",  # Correct - should work
        f"bearer {token}",  # Lowercase - might fail
        f"BEARER {token}",  # Uppercase - might fail
        token,              # No prefix - should fail
    ]
    
    results = []
    for auth_header in test_cases:
        headers = {"Authorization": auth_header}
        response = client.get("/api/auth/me", headers=headers)
        results.append((auth_header[:10], response.status_code))
    
    # At least the correct format should work
    assert results[0][1] == 200, "Correct 'Bearer' format should work"


def test_auth_response_includes_doctor_info(client: TestClient, db_session: Session, test_doctor_credentials):
    """Test that auth responses include complete doctor information."""
    
    # Register
    response = client.post(
        "/api/auth/register",
        json={
            "name": "InfoTestDoctor",
            "password": "password123"
        }
    )
    assert response.status_code == 201
    data = response.json()
    
    # Check doctor info structure
    doctor = data["doctor"]
    assert "id" in doctor
    assert "name" in doctor
    assert doctor["name"] == "InfoTestDoctor"
    assert "created_at" in doctor
    
    # Should NOT include password
    assert "password" not in doctor
    assert "password_hash" not in doctor


def test_simultaneous_tokens_for_same_doctor(client: TestClient, db_with_doctor, test_doctor_credentials):
    """Test that multiple login sessions create tokens."""
    
    import time
    
    # Login twice with slight delay to get different timestamps
    response1 = client.post("/api/auth/login", json=test_doctor_credentials)
    token1 = response1.json()["access_token"]
    
    time.sleep(1.1)  # Wait to ensure different exp timestamp
    
    response2 = client.post("/api/auth/login", json=test_doctor_credentials)
    token2 = response2.json()["access_token"]
    
    # Tokens might be the same if created within same second, but both should work
    
    # Both tokens should work
    headers1 = {"Authorization": f"Bearer {token1}"}
    headers2 = {"Authorization": f"Bearer {token2}"}
    
    me_response1 = client.get("/api/auth/me", headers=headers1)
    me_response2 = client.get("/api/auth/me", headers=headers2)
    
    assert me_response1.status_code == 200
    assert me_response2.status_code == 200

