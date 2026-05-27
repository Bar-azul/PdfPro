"""
OCR router — /api/ocr/*
Extract text from scanned PDFs and images using Tesseract.
"""

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from ..middleware.rate_limit import limiter
from ..models.schemas import FileResult
from .auth import get_optional_user
from ..services.ocr_service import OCRService
from ..services.storage_service import StorageService
from ..utils.file_utils import PDF_ONLY, IMAGE_TYPES, make_file_response, validate_upload

router = APIRouter()


def _is_pro(user): return user is not None and user.get("plan") in ("pro", "enterprise")


@router.post("/extract", summary="Extract text from scanned PDF or image")
@limiter.limit("10/hour")
async def ocr_extract(
    request: Request,
    file: UploadFile = File(..., description="PDF or image file"),
    language: str = Form(default="heb+eng", description="Tesseract language codes, e.g. 'heb+eng', 'eng', 'ara'"),
    output_format: str = Form(default="txt", description="Output format: 'txt', 'pdf', or 'docx'"),
    dpi: int = Form(default=300),
    pages: str | None = Form(default=None),
    user: dict | None = Depends(get_optional_user),
):
    """
    Run OCR on a scanned PDF or image and extract the text.
    Supports Hebrew, Arabic, English and 40+ languages.
    """
    t0 = time.time()
    allowed = PDF_ONLY | IMAGE_TYPES
    page_list = [int(p) for p in pages.split(",") if p.strip().isdigit()] if pages else None

    data = await validate_upload(file, allowed_mimes=allowed, is_pro=_is_pro(user))
    upload_meta = await StorageService.save_upload(data, file.filename or "upload")

    mime = file.content_type or ""
    is_image = mime in IMAGE_TYPES

    # ── TXT output ────────────────────────────────────────────────────────────
    if output_format == "txt":
        if is_image:
            result = await asyncio.to_thread(
                OCRService.ocr_image, upload_meta["path"], language
            )
            return {
                "text": result["text"],
                "confidence": result["confidence"],
                "processing_time_ms": int((time.time() - t0) * 1000),
            }
        else:
            results = await asyncio.to_thread(
                OCRService.extract_text, upload_meta["path"], language, dpi, page_list
            )
            full_text = "\n\n".join(r["text"] for r in results)
            avg_conf = sum(r["confidence"] for r in results) / len(results) if results else 0
            return {
                "text": full_text,
                "pages": results,
                "avg_confidence": round(avg_conf, 3),
                "processing_time_ms": int((time.time() - t0) * 1000),
            }

    # ── PDF output (searchable) ───────────────────────────────────────────────
    elif output_format == "pdf":
        # extract_to_searchable_pdf now handles both PDF and image input
        result_path = await asyncio.to_thread(
            OCRService.extract_to_searchable_pdf, upload_meta["path"], language, dpi
        )
        out_name = f"{Path(file.filename or 'ocr').stem}_searchable.pdf"
        output_meta = await StorageService.save_output(result_path, out_name)
        return make_file_response(
            output_meta["file_id"], out_name, output_meta["download_url"],
            output_meta["size"], t0,
        )

    # ── DOCX output ───────────────────────────────────────────────────────────
    elif output_format == "docx":
        result_path = await asyncio.to_thread(
            OCRService.extract_to_docx, upload_meta["path"], language, dpi
        )
        out_name = f"{Path(file.filename or 'ocr').stem}_ocr.docx"
        output_meta = await StorageService.save_output(result_path, out_name)
        return make_file_response(
            output_meta["file_id"], out_name, output_meta["download_url"],
            output_meta["size"], t0,
        )

    raise HTTPException(400, detail=f"פורמט פלט לא נתמך: {output_format}")


@router.get("/languages", summary="List available OCR languages")
async def list_languages():
    """Return all installed Tesseract language codes."""
    langs = await asyncio.to_thread(OCRService.get_available_languages)
    return {"languages": langs, "count": len(langs)}