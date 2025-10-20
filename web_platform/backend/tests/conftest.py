"""
Pytest Configuration and Fixtures

Shared test fixtures for all backend tests.
"""

import pytest
import logging
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Reduce SQLAlchemy logging noise during tests
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

from app.main import app
from app.database import Base, get_db
from app.models import Doctor, Patient, Chat, Message, Scan, SuggestedQuestion
from app.utils.security import get_password_hash

# Create in-memory SQLite database for testing
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency with test database."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Override dependency
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    db = TestingSessionLocal()
    
    yield db
    
    # Cleanup
    db.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create a test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_doctor(db_session):
    """Create a test doctor."""
    doctor = Doctor(
        name="Test Doctor",
        password_hash=get_password_hash("testpassword123")
    )
    db_session.add(doctor)
    db_session.commit()
    db_session.refresh(doctor)
    return doctor


@pytest.fixture
def auth_headers(client, test_doctor):
    """Get authentication headers for test doctor."""
    response = client.post(
        "/api/auth/login",
        json={"name": "Test Doctor", "password": "testpassword123"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_patient(db_session, test_doctor):
    """Create a test patient."""
    patient = Patient(
        name="Test Patient",
        doctor_id=test_doctor.id
    )
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)
    return patient


@pytest.fixture
def test_chat(db_session, test_patient):
    """Create a test chat."""
    chat = Chat(
        name="Test Chat",
        patient_id=test_patient.id
    )
    db_session.add(chat)
    db_session.commit()
    db_session.refresh(chat)
    return chat


@pytest.fixture
def test_message(db_session, test_chat):
    """Create a test message."""
    message = Message(
        chat_id=test_chat.id,
        role="user",
        content="Test message"
    )
    db_session.add(message)
    db_session.commit()
    db_session.refresh(message)
    return message




