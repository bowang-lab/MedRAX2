"""
Question API Routes

Endpoints for suggested questions management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..database import get_db
from ..models import Doctor, SuggestedQuestion
from ..schemas.question import QuestionCreate, QuestionResponse
from ..dependencies import get_current_doctor

router = APIRouter()


@router.get("", response_model=List[QuestionResponse])
def list_questions(current_doctor: Doctor = Depends(get_current_doctor), db: Session = Depends(get_db)):
    """List all questions (default + doctor's custom questions)."""

    # Get default questions + doctor's custom questions
    questions = (
        db.query(SuggestedQuestion)
        .filter((SuggestedQuestion.is_default == True) | (SuggestedQuestion.doctor_id == current_doctor.id))
        .order_by(SuggestedQuestion.display_order)
        .all()
    )

    return [QuestionResponse.model_validate(q) for q in questions]


@router.post("", response_model=QuestionResponse, status_code=status.HTTP_201_CREATED)
def create_question(
    question_data: QuestionCreate,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Create a custom question for the current doctor."""

    question = SuggestedQuestion(
        doctor_id=current_doctor.id,
        question=question_data.question,
        is_default=False,
        display_order=question_data.display_order,
    )
    db.add(question)
    db.commit()
    db.refresh(question)

    return QuestionResponse.model_validate(question)


@router.delete("/{question_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_question(
    question_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Delete a custom question."""

    question = (
        db.query(SuggestedQuestion)
        .filter(
            SuggestedQuestion.id == question_id,
            SuggestedQuestion.doctor_id == current_doctor.id,
            SuggestedQuestion.is_default == False,  # Can't delete default questions
        )
        .first()
    )

    if not question:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found or cannot be deleted")

    db.delete(question)
    db.commit()

    return None
