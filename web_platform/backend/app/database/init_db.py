"""
Database Initialization

Creates all tables and optionally seeds default data.
"""

from sqlalchemy.orm import Session
from datetime import datetime

from .base import Base
from .session import engine, SessionLocal
from ..models import SuggestedQuestion


def init_db() -> None:
    """
    Initialize database by creating all tables.
    """
    # Import all models to register them with Base
    from ..models import (
        Doctor,
        Patient,
        Chat,
        Message,
        Scan,
        ToolExecution,
        ToolExecutionLog,
        ToolExecutionResult,
        SuggestedQuestion,
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")

    # Seed default data
    seed_default_questions()


def seed_default_questions() -> None:
    """
    Seed default suggested questions.
    """
    db: Session = SessionLocal()

    try:
        # Check if default questions already exist
        existing = db.query(SuggestedQuestion).filter(SuggestedQuestion.is_default == True).first()

        if existing:
            print("Default questions already exist, skipping seed")
            return

        # Default questions
        default_questions = [
            {"question": "Is there pneumonia?", "display_order": 1},
            {"question": "Measure heart size", "display_order": 2},
            {"question": "What abnormalities do you see?", "display_order": 3},
            {"question": "Generate a report", "display_order": 4},
            {"question": "Classify this image", "display_order": 5},
            {"question": "Segment the lungs", "display_order": 6},
        ]

        for q_data in default_questions:
            question = SuggestedQuestion(
                doctor_id=None,
                question=q_data["question"],
                is_default=True,
                display_order=q_data["display_order"],
                created_at=datetime.utcnow(),
            )
            db.add(question)

        db.commit()
        print(f"Seeded {len(default_questions)} default questions")

    except Exception as e:
        print(f"Error seeding default questions: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialization complete!")
