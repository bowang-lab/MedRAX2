"""
Authentication API Tests
"""

import pytest


def test_register_doctor(client):
    """Test doctor registration."""
    response = client.post(
        "/api/auth/register",
        json={"name": "New Doctor", "password": "password123"}
    )
    assert response.status_code == 201
    data = response.json()
    assert "doctor" in data
    assert "access_token" in data
    assert data["doctor"]["name"] == "New Doctor"
    assert "password" not in data["doctor"]


def test_register_duplicate_doctor(client, test_doctor):
    """Test registering a doctor with duplicate name."""
    response = client.post(
        "/api/auth/register",
        json={"name": "Test Doctor", "password": "password123"}
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"].lower()


def test_login_success(client, test_doctor):
    """Test successful login."""
    response = client.post(
        "/api/auth/login",
        json={"name": "Test Doctor", "password": "testpassword123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "doctor" in data
    assert data["doctor"]["name"] == "Test Doctor"


def test_login_invalid_password(client, test_doctor):
    """Test login with invalid password."""
    response = client.post(
        "/api/auth/login",
        json={"name": "Test Doctor", "password": "wrongpassword"}
    )
    assert response.status_code == 401


def test_login_nonexistent_doctor(client):
    """Test login with nonexistent doctor."""
    response = client.post(
        "/api/auth/login",
        json={"name": "Nonexistent Doctor", "password": "password123"}
    )
    assert response.status_code == 401


def test_get_current_doctor(client, auth_headers):
    """Test getting current doctor profile."""
    response = client.get("/api/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Doctor"
    assert "password" not in data


def test_get_current_doctor_unauthorized(client):
    """Test getting current doctor without auth."""
    response = client.get("/api/auth/me")
    assert response.status_code == 401

