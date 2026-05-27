"""
AuthService — with PostgreSQL database support.
מחליף את מבנה ה-dict בזיכרון ב-DB אמיתי.
"""

import logging
from datetime import datetime, timedelta
from typing import Literal

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from ..config import settings
from ..database import db_session
from ..models.db_models import User, UsageRecord

logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DAILY_LIMITS = {"free": 5, "pro": 1000, "enterprise": 999999}


class AuthService:

    # ── Registration ──────────────────────────────────────────────────────────

    @staticmethod
    def register(email: str, password: str, full_name: str) -> User:
        email = email.lower().strip()
        with db_session() as db:
            if db.query(User).filter(User.email == email).first():
                raise ValueError("כתובת האימייל כבר קיימת במערכת")
            user = User(
                email=email,
                password_hash=pwd_context.hash(password),
                full_name=full_name,
                plan="free",
            )
            db.add(user)
            db.flush()   # get the ID before commit
            db.refresh(user)
            logger.info(f"New user: {email}")
            # Return a plain dict so session doesn't expire
            return _user_to_dict(user)

    # ── Login ─────────────────────────────────────────────────────────────────

    @staticmethod
    def login(email: str, password: str) -> dict:
        email = email.lower().strip()
        with db_session() as db:
            user = db.query(User).filter(User.email == email).first()
            if not user or not pwd_context.verify(password, user.password_hash):
                raise ValueError("אימייל או סיסמה שגויים")
            if not user.is_active:
                raise ValueError("חשבון זה אינו פעיל")
            return _user_to_dict(user)

    # ── JWT ───────────────────────────────────────────────────────────────────

    @staticmethod
    def create_token(user_id: str) -> str:
        expire = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
        return jwt.encode(
            {"sub": user_id, "exp": expire, "iat": datetime.utcnow()},
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

    @staticmethod
    def verify_token(token: str) -> str:
        try:
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
            )
            user_id = payload.get("sub")
            if not user_id:
                raise JWTError("Missing subject")
            return user_id
        except JWTError as e:
            raise ValueError(f"טוקן לא תקין: {e}")

    # ── User lookup ───────────────────────────────────────────────────────────

    @staticmethod
    def get_user(user_id: str) -> dict | None:
        with db_session() as db:
            user = db.query(User).filter(User.id == user_id).first()
            return _user_to_dict(user) if user else None

    @staticmethod
    def get_user_by_email(email: str) -> dict | None:
        with db_session() as db:
            user = db.query(User).filter(
                User.email == email.lower().strip()
            ).first()
            return _user_to_dict(user) if user else None

    # ── Usage tracking ────────────────────────────────────────────────────────

    @staticmethod
    def check_and_increment_usage(user_id: str) -> tuple[bool, int]:
        """Returns (allowed, count_today)."""
        user = AuthService.get_user(user_id)
        plan = user["plan"] if user else "free"
        limit = DAILY_LIMITS.get(plan, 5)
        today = datetime.utcnow().date().isoformat()

        with db_session() as db:
            record = db.query(UsageRecord).filter(
                UsageRecord.user_id == user_id,
                UsageRecord.date == today,
            ).first()

            if not record:
                record = UsageRecord(user_id=user_id, date=today, count=0)
                db.add(record)

            if record.count >= limit:
                return False, record.count

            record.count += 1
            return True, record.count

    @staticmethod
    def get_usage_today(user_id: str) -> int:
        today = datetime.utcnow().date().isoformat()
        with db_session() as db:
            record = db.query(UsageRecord).filter(
                UsageRecord.user_id == user_id,
                UsageRecord.date == today,
            ).first()
            return record.count if record else 0

    # ── Plan upgrade ──────────────────────────────────────────────────────────

    @staticmethod
    def upgrade_plan(user_id: str, plan: Literal["free", "pro", "enterprise"]) -> dict:
        with db_session() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("משתמש לא נמצא")
            user.plan = plan
            logger.info(f"User {user_id} upgraded to {plan}")
            return _user_to_dict(user)


# ── Helper ────────────────────────────────────────────────────────────────────

def _user_to_dict(user: User) -> dict:
    """Convert SQLAlchemy User object to plain dict (avoids lazy loading issues)."""
    return {
        "id":         user.id,
        "email":      user.email,
        "full_name":  user.full_name,
        "plan":       user.plan,
        "is_active":  user.is_active,
        "created_at": user.created_at,
    }