"""
Configuration — final version with all services.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"
    BASE_URL: str = "http://localhost:8000"
    FRONTEND_URL: str = "http://localhost:8000"

    # ── Database ──────────────────────────────────────────────────────────────
    # SQLite לפיתוח, PostgreSQL בפרודקשן
    DATABASE_URL: str = "sqlite:///./pdfpro.db"
    # Production: postgresql://user:password@host:5432/pdfpro

    # ── CORS ──────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:63342",
        "null",
    ]

    # ── Storage ───────────────────────────────────────────────────────────────
    UPLOAD_DIR: Path = Path("/tmp/pdfpro/uploads")
    OUTPUT_DIR: Path = Path("/tmp/pdfpro/outputs")
    MAX_FILE_SIZE_MB: int = 100
    MAX_FILE_SIZE_PRO_MB: int = 500
    FILE_TTL_SECONDS: int = 3600
    CLEANUP_INTERVAL_SECONDS: int = 300

    # ── Rate limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_FREE: str = "5/day"
    RATE_LIMIT_PRO: str = "1000/day"

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24 * 7

    # ── Tranzila ──────────────────────────────────────────────────────────────
    TRANZILA_SUPPLIER: str = "placeholder"
    TRANZILA_TERMINAL_PASSWORD: str = ""

    # ── External ──────────────────────────────────────────────────────────────
    LIBREOFFICE_PATH: str = "C:\Program Files\LibreOffice\program\soffice.exe"
    TESSERACT_PATH: str = "/usr/bin/tesseract"
    OCR_LANGUAGES: list[str] = ["heb", "eng", "ara"]

    @field_validator("UPLOAD_DIR", "OUTPUT_DIR", mode="before")
    @classmethod
    def ensure_path(cls, v):
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    model_config = {"env_file": "../.env", "env_file_encoding": "utf-8"}


settings = Settings()