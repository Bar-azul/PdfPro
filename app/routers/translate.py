"""
Translate router — /api/translate/*
Translate PDFs into 40+ languages using Google Translate.
"""

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Request, File, Form, UploadFile

from ..middleware.rate_limit import limiter
from ..models.schemas import FileResult, SUPPORTED_LANGUAGES
from .auth import get_optional_user
from ..services.storage_service import StorageService
from ..services.translate_service import TranslateService
from ..utils.file_utils import PDF_ONLY, make_file_response, validate_upload

router = APIRouter()


def _is_pro(user): return user is not None and user.get("plan") in ("pro", "enterprise")


@router.post("/pdf", response_model=FileResult, summary="Translate a PDF document")
@limiter.limit("5/hour")
async def translate_pdf(
    request: Request,
    file: UploadFile = File(...),
    target_language: str = Form(..., description="Target language code, e.g. 'he', 'en', 'ar'"),
    source_language: str = Form(default="auto"),
    preserve_layout: bool = Form(default=True),
    pages: str | None = Form(default=None),
    user: dict | None = Depends(get_optional_user),
):
    """
    Translate a PDF document into another language.
    Supports 40+ languages including Hebrew, Arabic, English, Russian, and more.
    """
    t0 = time.time()

    if target_language not in SUPPORTED_LANGUAGES and target_language != "auto":
        from fastapi import HTTPException
        raise HTTPException(400, detail=f"שפה לא נתמכת: {target_language}. השתמש ב-/api/translate/languages לרשימה מלאה.")

    page_list = [int(p) for p in pages.split(",") if p.strip().isdigit()] if pages else None

    data = await validate_upload(file, allowed_mimes=PDF_ONLY, is_pro=_is_pro(user))
    upload_meta = await StorageService.save_upload(data, file.filename or "upload.pdf")

    result_path = await asyncio.to_thread(
        TranslateService.translate_pdf,
        upload_meta["path"],
        target_language,
        source_language,
        preserve_layout,
        page_list,
    )

    lang_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
    out_name = f"{Path(file.filename or 'translated').stem}_{target_language}.pdf"
    output_meta = await StorageService.save_output(result_path, out_name)

    return {
        **make_file_response(
            output_meta["file_id"], out_name, output_meta["download_url"],
            output_meta["size"], t0,
        ),
        "target_language": target_language,
        "target_language_name": lang_name,
    }


@router.post("/text", summary="Translate plain text")
@limiter.limit("30/hour")
async def translate_text(
    request: Request,
    text: str = Form(..., max_length=10000),
    target_language: str = Form(...),
    source_language: str = Form(default="auto"),
    user: dict | None = Depends(get_optional_user),
):
    """Translate a plain text string."""
    t0 = time.time()
    result = await asyncio.to_thread(
        TranslateService.translate_text, text, target_language, source_language
    )
    return {
        "original": text,
        "translated": result,
        "target_language": target_language,
        "processing_time_ms": int((time.time() - t0) * 1000),
    }


@router.get("/languages", summary="List supported translation languages")
async def list_languages():
    """Return all supported language codes and their names."""
    return {"languages": SUPPORTED_LANGUAGES, "count": len(SUPPORTED_LANGUAGES)}
