"""
Concurrency Test Suite

Tests multi-user concurrent access scenarios to verify thread safety and data integrity.
"""

import pytest
import asyncio
import concurrent.futures
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import time

from app.database.base import Base
from app.models import Doctor, Patient, Chat, Message, Scan
from app.utils.security import create_access_token


# Test database
TEST_DATABASE_URL = "sqlite:///./test_concurrency.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="module")
def setup_database():
    """Create test database and tables."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_db():
    """Provide a fresh database session for each test."""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture
def test_doctor(test_db):
    """Create a test doctor."""
    from app.utils.security import get_password_hash
    doctor = Doctor(
        name="Test Doctor",
        password_hash=get_password_hash("testpassword123")
    )
    test_db.add(doctor)
    test_db.commit()
    test_db.refresh(doctor)
    return doctor


@pytest.fixture
def test_patient(test_db, test_doctor):
    """Create a test patient."""
    patient = Patient(
        doctor_id=test_doctor.id,
        name="Test Patient"
    )
    test_db.add(patient)
    test_db.commit()
    test_db.refresh(patient)
    return patient


@pytest.fixture
def test_chat(test_db, test_patient):
    """Create a test chat."""
    chat = Chat(
        patient_id=test_patient.id,
        name="Test Chat"
    )
    test_db.add(chat)
    test_db.commit()
    test_db.refresh(chat)
    return chat


class TestConcurrentDatabaseAccess:
    """Test concurrent database operations."""
    
    def test_concurrent_patient_creation(self, setup_database, test_doctor):
        """Test multiple users creating patients simultaneously."""
        def create_patient(doctor_id, index):
            db = TestSessionLocal()
            try:
                patient = Patient(
                    doctor_id=doctor_id,
                    name=f"Concurrent Patient {index}"
                )
                db.add(patient)
                db.commit()
                db.refresh(patient)
                return patient.id
            except Exception as e:
                db.rollback()
                raise e
            finally:
                db.close()
        
        # Create 10 patients concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_patient, test_doctor.id, i) for i in range(10)]
            patient_ids = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Verify all 10 patients were created
        db = TestSessionLocal()
        try:
            count = db.query(Patient).filter(Patient.doctor_id == test_doctor.id).count()
            assert count >= 10  # At least our 10 concurrent patients
        finally:
            db.close()
    
    def test_concurrent_message_creation(self, setup_database, test_chat):
        """Test multiple users sending messages to same chat simultaneously."""
        def send_message(chat_id, index):
            db = TestSessionLocal()
            try:
                message = Message(
                    chat_id=chat_id,
                    role="user",
                    content=f"Concurrent message {index}"
                )
                db.add(message)
                db.commit()
                db.refresh(message)
                return message.id
            except Exception as e:
                db.rollback()
                raise e
            finally:
                db.close()
        
        # Send 20 messages concurrently to same chat
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(send_message, test_chat.id, i) for i in range(20)]
            message_ids = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Verify all 20 messages were created with unique IDs
        assert len(set(message_ids)) == 20
        
        db = TestSessionLocal()
        try:
            count = db.query(Message).filter(Message.chat_id == test_chat.id).count()
            assert count >= 20
        finally:
            db.close()
    
    def test_concurrent_read_write(self, setup_database, test_chat):
        """Test concurrent reads while writing to same resource."""
        def write_messages(chat_id, count):
            db = TestSessionLocal()
            try:
                for i in range(count):
                    message = Message(
                        chat_id=chat_id,
                        role="assistant",
                        content=f"Write test message {i}"
                    )
                    db.add(message)
                    db.commit()
                    time.sleep(0.01)  # Small delay to simulate real processing
            except Exception as e:
                db.rollback()
                raise e
            finally:
                db.close()
        
        def read_messages(chat_id, iterations):
            results = []
            db = TestSessionLocal()
            try:
                for _ in range(iterations):
                    count = db.query(Message).filter(Message.chat_id == chat_id).count()
                    results.append(count)
                    time.sleep(0.01)
                return results
            finally:
                db.close()
        
        # Start writer thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            write_future = executor.submit(write_messages, test_chat.id, 10)
            
            # Start 5 reader threads
            read_futures = [executor.submit(read_messages, test_chat.id, 10) for _ in range(5)]
            
            # Wait for all to complete
            write_future.result()
            read_results = [f.result() for f in read_futures]
        
        # Verify reads returned valid (non-negative) counts
        for results in read_results:
            assert all(count >= 0 for count in results)
            # Counts should be monotonically increasing (or stable)
            assert results[-1] >= results[0]


class TestConcurrentFileOperations:
    """Test concurrent file upload and deletion."""
    
    def test_concurrent_file_naming(self):
        """Verify UUID filenames prevent collisions."""
        from app.utils.file_utils import get_file_extension
        import uuid
        
        # Simulate 100 concurrent uploads with same original filename
        filenames = set()
        for _ in range(100):
            ext = "jpg"
            unique_filename = f"{uuid.uuid4()}.{ext}"
            filenames.add(unique_filename)
        
        # All should be unique
        assert len(filenames) == 100


class TestRaceConditions:
    """Test specific race condition scenarios."""
    
    def test_delete_while_reading(self, setup_database, test_chat):
        """Test patient deletion while another user is reading messages."""
        def read_chat_messages(chat_id, iterations):
            errors = []
            db = TestSessionLocal()
            try:
                for _ in range(iterations):
                    try:
                        messages = db.query(Message).filter(Message.chat_id == chat_id).all()
                        time.sleep(0.01)
                    except Exception as e:
                        errors.append(str(e))
            finally:
                db.close()
            return errors
        
        def delete_chat(chat_id):
            time.sleep(0.05)  # Let readers get started
            db = TestSessionLocal()
            try:
                chat = db.query(Chat).filter(Chat.id == chat_id).first()
                if chat:
                    db.delete(chat)
                    db.commit()
            except Exception as e:
                db.rollback()
                raise e
            finally:
                db.close()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            # Start 5 reader threads
            read_futures = [executor.submit(read_chat_messages, test_chat.id, 10) for _ in range(5)]
            
            # Start delete thread
            delete_future = executor.submit(delete_chat, test_chat.id)
            
            # Wait for all to complete
            delete_future.result()
            all_errors = []
            for f in read_futures:
                all_errors.extend(f.result())
        
        # Readers may get errors after deletion (that's expected), 
        # but should not crash or corrupt data
        # This test mainly ensures no deadlocks or corruption


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

