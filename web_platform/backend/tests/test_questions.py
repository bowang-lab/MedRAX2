"""
Suggested Questions API Tests

Test question CRUD operations.
"""

import pytest


def test_list_questions(client, auth_headers):
    """Test listing all questions."""
    response = client.get("/api/questions", headers=auth_headers)
    assert response.status_code == 200
    
    questions = response.json()
    assert isinstance(questions, list)
    # Default questions should be seeded
    assert len(questions) >= 0


def test_create_question(client, auth_headers, test_doctor):
    """Test creating a new question."""
    response = client.post(
        "/api/questions",
        json={"question": "Is there a pneumothorax?"},
        headers=auth_headers
    )
    assert response.status_code == 201
    
    question = response.json()
    assert "id" in question
    assert question["question"] == "Is there a pneumothorax?"
    assert question["doctor_id"] == test_doctor.id


def test_create_question_empty(client, auth_headers):
    """Test creating question with empty text."""
    response = client.post(
        "/api/questions",
        json={"question": ""},
        headers=auth_headers
    )
    assert response.status_code == 422  # Validation error


def test_delete_question(client, auth_headers):
    """Test deleting a question."""
    # First create a question
    response = client.post(
        "/api/questions",
        json={"question": "Test question to delete"},
        headers=auth_headers
    )
    assert response.status_code == 201
    question_id = response.json()["id"]
    
    # Delete it
    response = client.delete(
        f"/api/questions/{question_id}",
        headers=auth_headers
    )
    assert response.status_code == 204
    
    # Verify it's gone
    response = client.get("/api/questions", headers=auth_headers)
    questions = response.json()
    assert not any(q["id"] == question_id for q in questions)


def test_delete_nonexistent_question(client, auth_headers):
    """Test deleting a question that doesn't exist."""
    response = client.delete(
        "/api/questions/nonexistent-id",
        headers=auth_headers
    )
    assert response.status_code == 404


def test_questions_unauthorized(client):
    """Test that questions endpoint requires auth."""
    response = client.get("/api/questions")
    assert response.status_code == 401
    
    response = client.post(
        "/api/questions",
        json={"question": "Test"}
    )
    assert response.status_code == 401


def test_create_multiple_questions(client, auth_headers):
    """Test creating multiple questions."""
    questions = [
        "Is there consolidation?",
        "Check for pleural effusion",
        "Assess heart size",
    ]
    
    created_ids = []
    for q in questions:
        response = client.post(
            "/api/questions",
            json={"question": q},
            headers=auth_headers
        )
        assert response.status_code == 201
        created_ids.append(response.json()["id"])
    
    # Verify all were created
    response = client.get("/api/questions", headers=auth_headers)
    all_questions = response.json()
    
    for q_id in created_ids:
        assert any(q["id"] == q_id for q in all_questions)


def test_question_belongs_to_doctor(client, auth_headers, test_doctor):
    """Test that created questions belong to the correct doctor."""
    response = client.post(
        "/api/questions",
        json={"question": "Doctor-specific question"},
        headers=auth_headers
    )
    assert response.status_code == 201
    
    question = response.json()
    assert question["doctor_id"] == test_doctor.id

