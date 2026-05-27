"""
Pydantic schemas — request / response models for the entire API.
"""

from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, EmailStr


# ── Common ─────────────────────────────────────────────────────────────────────

class FileResult(BaseModel):
    """Returned after any successful file operation."""
    file_id: str
    filename: str
    download_url: str
    size_bytes: int
    expires_at: datetime
    processing_time_ms: int


class ErrorDetail(BaseModel):
    detail: str
    code: str | None = None


# ── Auth ───────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=1, max_length=100)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    plan: Literal["free", "pro", "enterprise"] = "free"


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    plan: Literal["free", "pro", "enterprise"]
    conversions_today: int
    created_at: datetime


# ── Convert ────────────────────────────────────────────────────────────────────

class ConvertPDFToImagesRequest(BaseModel):
    """Extra options for PDF → image conversions."""
    dpi: int = Field(default=150, ge=72, le=300, description="Image resolution in DPI")
    format: Literal["jpg", "png", "webp"] = "jpg"
    quality: int = Field(default=85, ge=10, le=100)
    pages: list[int] | None = Field(
        default=None,
        description="Specific page numbers (1-based). None = all pages.",
    )


class ConvertImagesToPDFRequest(BaseModel):
    page_size: Literal["A4", "Letter", "Legal", "A3", "fit"] = "fit"


# ── Organize ───────────────────────────────────────────────────────────────────

class SplitRequest(BaseModel):
    mode: Literal["pages", "ranges", "every_n"] = "ranges"
    ranges: list[str] | None = Field(
        default=None,
        description="Page ranges like ['1-3', '4-6', '7']. Required when mode='ranges'.",
        examples=[["1-3", "4-6"]],
    )
    every_n: int | None = Field(
        default=None,
        description="Split every N pages. Required when mode='every_n'.",
    )
    pages: list[int] | None = Field(
        default=None,
        description="Extract specific pages. Required when mode='pages'.",
    )


class MergeRequest(BaseModel):
    """Order matters — files are merged in the order provided."""
    pass  # file IDs come from the multipart upload


class CompressRequest(BaseModel):
    level: Literal["low", "medium", "high", "extreme"] = "medium"
    # low    → ~10% size reduction,  near-lossless
    # medium → ~40% size reduction,  minimal quality loss
    # high   → ~65% size reduction,  noticeable on images
    # extreme→ ~80% size reduction,  suitable for text-only docs


class RotateRequest(BaseModel):
    angle: Literal[90, 180, 270]
    pages: list[int] | None = Field(
        default=None,
        description="Pages to rotate (1-based). None = all pages.",
    )


class SplitResult(BaseModel):
    parts: list[FileResult]
    total_parts: int


# ── Edit ───────────────────────────────────────────────────────────────────────

class WatermarkRequest(BaseModel):
    type: Literal["text", "image"] = "text"
    text: str | None = Field(default=None, max_length=100)
    image_file_id: str | None = None

    # Appearance
    opacity: float = Field(default=0.3, ge=0.05, le=1.0)
    font_size: int = Field(default=48, ge=8, le=200)
    color: str = Field(default="#CCCCCC", description="Hex color for text watermark")
    rotation: int = Field(default=-45, ge=-180, le=180)

    # Position
    position: Literal["center", "tile", "top-left", "top-right", "bottom-left", "bottom-right"] = "center"

    # Pages
    pages: list[int] | None = None


class SignatureRequest(BaseModel):
    type: Literal["draw", "text", "image"] = "text"
    text: str | None = Field(default=None, max_length=100, description="Name to render as signature")
    image_file_id: str | None = None

    # Position on page (percentage-based, 0.0–1.0)
    x: float = Field(default=0.6, ge=0.0, le=1.0)
    y: float = Field(default=0.85, ge=0.0, le=1.0)
    width: float = Field(default=0.25, ge=0.05, le=0.8)
    page: int = Field(default=-1, description="Page number (1-based). -1 = last page.")


class PasswordRequest(BaseModel):
    action: Literal["protect", "unlock"]
    password: str = Field(min_length=1, max_length=128)
    owner_password: str | None = None
    # Permissions when protecting
    allow_print: bool = True
    allow_copy: bool = False
    allow_edit: bool = False


class RedactRequest(BaseModel):
    """Black-out sensitive text in a PDF."""
    texts: list[str] = Field(description="Exact text strings to redact")
    case_sensitive: bool = False
    pages: list[int] | None = None


# ── OCR ────────────────────────────────────────────────────────────────────────

class OCRRequest(BaseModel):
    language: str = Field(
        default="heb+eng",
        description="Tesseract language codes, e.g. 'heb', 'eng', 'heb+eng', 'ara'.",
    )
    output_format: Literal["txt", "pdf", "docx", "json"] = "txt"
    dpi: int = Field(default=300, ge=150, le=600)
    pages: list[int] | None = None


class OCRResult(BaseModel):
    text: str | None = None
    pages: list[dict] | None = None   # [{page: 1, text: "...", confidence: 0.95}]
    file: FileResult | None = None     # If output_format != "txt"


# ── Translate ──────────────────────────────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    "iw": "עברית",
    "en": "English",
    "ar": "العربية",
    "ru": "Русский",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "tr": "Türkçe",
    "pl": "Polski",
    "nl": "Nederlands",
    "uk": "Українська",
    "ro": "Română",
    "sv": "Svenska",
    "fi": "Suomi",
    "da": "Dansk",
    "cs": "Čeština",
    "hu": "Magyar",
    "el": "Ελληνικά",
    "th": "ภาษาไทย",
    "vi": "Tiếng Việt",
    "id": "Bahasa Indonesia",
    "ms": "Bahasa Melayu",
    "fa": "فارسی",
    "hi": "हिन्दी",
    "bn": "বাংলা",
}


class TranslateRequest(BaseModel):
    source_language: str = Field(default="auto", description="Source language code or 'auto'")
    target_language: str = Field(description="Target language code, e.g. 'he', 'en'")
    preserve_layout: bool = Field(
        default=True,
        description="Try to preserve original document layout in output PDF",
    )
    pages: list[int] | None = None
