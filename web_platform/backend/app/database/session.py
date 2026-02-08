"""
Database Session

SQLAlchemy engine and session factory.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from typing import Generator

from ..config import settings

# Configure SQLite for better concurrency
if settings.DATABASE_URL.startswith("sqlite"):

    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        """
        Configure SQLite for multi-user concurrency.

        - WAL mode: Allows concurrent reads with writes
        - Busy timeout: Wait up to 5 seconds for lock instead of failing immediately
        - Journal mode: Write-Ahead Logging for better concurrency
        """
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
        cursor.execute("PRAGMA synchronous=NORMAL")  # Good balance of safety and speed
        cursor.close()

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
    echo=False,  # Disable SQL query logging for cleaner logs
    pool_size=10,  # Connection pool size (only for non-SQLite)
    max_overflow=20,  # Additional connections when pool full
    pool_pre_ping=True,  # Verify connections before using
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for getting database session.

    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
