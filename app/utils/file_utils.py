"""
File utilities — validation, format detection, size limits.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile
from ..config import settings

logger = logging.getLogger(__name__)

# ── Allowed MIME types ─────────────────────────────────────────────────────────
ALLOWED_TYPES = {
    # PDF
    "application/pdf": ".pdf",
    # Word
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    # Excel
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    # PowerPoint
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    # Images
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
    # Text
    "text/plain": ".txt",
    "text/html": ".html",
}

PDF_ONLY = {"application/pdf"}
OFFICE_TYPES = {
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp", "image/gif"}


async def validate_upload(
    file: UploadFile,
    allowed_mimes: set | None = None,
    max_mb: int | None = None,
    is_pro: bool = False,
) -> bytes:
    """
    Read and validate an uploaded file.
    Returns raw bytes on success, raises HTTPException on failure.
    """
    # Check content type
    mime = file.content_type or ""
    allowed = allowed_mimes or set(ALLOWED_TYPES.keys())
    if mime not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"סוג קובץ לא נתמך: {mime}. נתמך: {', '.join(sorted(allowed))}",
        )

    # Read file
    data = await file.read()
    size_mb = len(data) / (1024 * 1024)

    # Size check
    limit = max_mb or (
        settings.MAX_FILE_SIZE_PRO_MB if is_pro else settings.MAX_FILE_SIZE_MB
    )
    if size_mb > limit:
        raise HTTPException(
            status_code=413,
            detail=f"הקובץ גדול מדי ({size_mb:.1f}MB). מקסימום: {limit}MB",
        )

    return data


def make_file_response(
    file_id: str,
    filename: str,
    download_url: str,
    size_bytes: int,
    started_at: float,
) -> dict:
    """Build the standard FileResult response dict."""
    from datetime import timedelta

    return {
        "file_id": file_id,
        "filename": filename,
        "download_url": download_url,
        "size_bytes": size_bytes,
        "expires_at": (
            datetime.utcnow() + timedelta(seconds=settings.FILE_TTL_SECONDS)
        ).isoformat() + "Z",
        "processing_time_ms": int((time.time() - started_at) * 1000),
    }


def safe_filename(name: str) -> str:
    """Strip path components and dangerous characters from a filename."""
    import re
    name = Path(name).name  # Remove directory components
    name = re.sub(r"[^\w\-. ]", "_", name)
    return name[:200]
