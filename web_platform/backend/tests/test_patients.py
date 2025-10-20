"""
Patient API Tests
"""

import pytest


def test_list_patients_empty(client, auth_headers):
    """Test listing patients when none exist."""
    response = client.get("/api/patients", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []


def test_list_patients(client, auth_headers, test_patient):
    """Test listing patients."""
    response = client.get("/api/patients", headers=auth_headers)
    assert response.status_code == 200
    patients = response.json()
    assert len(patients) == 1
    assert patients[0]["id"] == test_patient.id
    assert patients[0]["name"] == "Test Patient"


def test_create_patient_with_name(client, auth_headers):
    """Test creating a patient with a name."""
    response = client.post(
        "/api/patients",
        json={"name": "John Doe"},
        headers=auth_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["name"] == "John Doe"


def test_create_patient_anonymous(client, auth_headers):
    """Test creating an anonymous patient."""
    response = client.post(
        "/api/patients",
        json={"name": None},
        headers=auth_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["name"] is None


def test_update_patient_name(client, auth_headers, test_patient):
    """Test updating patient name."""
    response = client.patch(
        f"/api/patients/{test_patient.id}",
        json={"name": "Updated Name"},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Name"


def test_delete_patient(client, auth_headers, test_patient):
    """Test deleting a patient."""
    response = client.delete(
        f"/api/patients/{test_patient.id}",
        headers=auth_headers
    )
    assert response.status_code == 204
    
    # Verify patient is deleted
    response = client.get("/api/patients", headers=auth_headers)
    assert len(response.json()) == 0


def test_delete_nonexistent_patient(client, auth_headers):
    """Test deleting a nonexistent patient."""
    response = client.delete(
        "/api/patients/nonexistent-id",
        headers=auth_headers
    )
    assert response.status_code == 404


def test_list_patients_unauthorized(client):
    """Test listing patients without auth."""
    response = client.get("/api/patients")
    assert response.status_code == 401

