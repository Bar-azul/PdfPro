"""
db_models.py — SQLAlchemy ORM models
טבלאות מסד הנתונים.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Integer, String, Text, ForeignKey
)
from sqlalchemy.orm import relationship

from ..database import Base


def _uuid():
    return str(uuid.uuid4())


# ── Users ─────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id            = Column(String(36),  primary_key=True, default=_uuid)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name     = Column(String(100), nullable=False)
    plan          = Column(String(20),  nullable=False, default="free")   # free/pro/enterprise
    is_active     = Column(Boolean,     nullable=False, default=True)
    created_at    = Column(DateTime,    nullable=False, default=datetime.utcnow)
    updated_at    = Column(DateTime,    nullable=False, default=datetime.utcnow,
                           onupdate=datetime.utcnow)

    # Relationships
    usage_records = relationship("UsageRecord", back_populates="user",
                                 cascade="all, delete-orphan")
    subscription  = relationship("Subscription", back_populates="user",
                                 uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email} plan={self.plan}>"


# ── Daily usage ───────────────────────────────────────────────────────────────
class UsageRecord(Base):
    __tablename__ = "usage_records"

    id         = Column(String(36), primary_key=True, default=_uuid)
    user_id    = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    date       = Column(String(10), nullable=False, index=True)  # "2026-05-27"
    count      = Column(Integer,    nullable=False, default=0)
    updated_at = Column(DateTime,   nullable=False, default=datetime.utcnow,
                        onupdate=datetime.utcnow)

    user = relationship("User", back_populates="usage_records")

    def __repr__(self):
        return f"<UsageRecord user={self.user_id} date={self.date} count={self.count}>"


# ── Subscriptions ─────────────────────────────────────────────────────────────
class Subscription(Base):
    __tablename__ = "subscriptions"

    id              = Column(String(36),  primary_key=True, default=_uuid)
    user_id         = Column(String(36),  ForeignKey("users.id"), unique=True, nullable=False)
    plan            = Column(String(20),  nullable=False)
    status          = Column(String(20),  nullable=False, default="active")  # active/cancelled/past_due
    card_token      = Column(String(255), nullable=True)   # טוקן כרטיס אשראי
    amount          = Column(Integer,     nullable=False)   # סכום בשקלים
    next_charge     = Column(DateTime,    nullable=True)
    cancelled_at    = Column(DateTime,    nullable=True)
    tranzila_ref    = Column(String(100), nullable=True)    # מספר אסמכתא
    created_at      = Column(DateTime,    nullable=False, default=datetime.utcnow)
    updated_at      = Column(DateTime,    nullable=False, default=datetime.utcnow,
                             onupdate=datetime.utcnow)

    user = relationship("User", back_populates="subscription")

    def __repr__(self):
        return f"<Subscription user={self.user_id} plan={self.plan} status={self.status}>"
