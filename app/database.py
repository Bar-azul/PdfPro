"""
database.py — SQLAlchemy setup
מחבר ל-PostgreSQL (או SQLite לפיתוח מקומי).
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from .config import settings

# ── Engine ────────────────────────────────────────────────────────────────────
# PostgreSQL בפרודקשן, SQLite לפיתוח מקומי
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,          # בדוק חיבור לפני כל שאילתה
    pool_size=10,                # מקסימום חיבורים במאגר
    max_overflow=20,
    echo=settings.DEBUG,         # הדפס SQL בזמן debug
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Dependency ────────────────────────────────────────────────────────────────
def get_db() -> Session:
    """FastAPI dependency — מספק session ומשחרר אחרי הבקשה."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session():
    """Context manager לשימוש מחוץ ל-FastAPI."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """יוצר את כל הטבלאות אם לא קיימות."""
    from .models import db_models  # noqa — import so tables are registered
    Base.metadata.create_all(bind=engine)
