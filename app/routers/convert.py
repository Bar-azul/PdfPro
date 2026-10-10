"""
Convert router — /api/convert/*
Handles all format conversion endpoints.
"""

import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse

from ..middleware.rate_limit import limiter
from ..models.schemas import ConvertPDFToImagesRequest, FileResult
from .auth import get_optional_user
from ..services.auth_service import AuthService
from ..services.convert_service import ConvertService
from ..services.storage_service import StorageService
from ..utils.file_utils import (
    IMAGE_TYPES, OFFICE_TYPES, PDF_ONLY, make_file_response, validate_upload
)

router = APIRouter()


def _is_pro(user: dict | None) -> bool:
    return user is not None and user.get("plan") in ("pro", "enterprise")


# ── Helper: save upload, run conversion, save output ─────────────────────────

async def _run_conversion(
    file: UploadFile,
    allowed_mimes: set,
    output_ext: str,
    output_name_suffix: str,
    converter_fn,
    user: dict | None,
    t0: float,
) -> dict:
    data = await validate_upload(file, allowed_mimes=allowed_mimes, is_pro=_is_pro(user))
    upload_meta = await StorageService.save_upload(data, file.filename or "upload")

    result_path: Path = await __import__("asyncio").to_thread(
        converter_fn, upload_meta["path"]
    )

    original_stem = Path(file.filename or "document").stem
    out_name = f"{original_stem}{output_name_suffix}{output_ext}"
    output_meta = await StorageService.save_output(result_path, out_name)

    return make_file_response(
        file_id=output_meta["file_id"],
        filename=out_name,
        download_url=output_meta["download_url"],
        size_bytes=output_meta["size"],
        started_at=t0,
    )


# ════════════════════════════════════════════════════════════════════════════
# PDF → Other formats
# ════════════════════════════════════════════════════════════════════════════

@router.post("/pdf-to-word", response_model=FileResult, summary="PDF → Word (.docx)")
@limiter.limit("20/hour")
async def pdf_to_word(
    request: Request,
    file: UploadFile = File(..., description="PDF file to convert"),
    user: dict | None = Depends(get_optional_user),
):
    """Convert a PDF to an editable Word document (.docx)."""
    t0 = time.time()
    return await _run_conversion(
        file=file,
        allowed_mimes=PDF_ONLY,
        output_ext=".docx",
        output_name_suffix="_converted",
        converter_fn=ConvertService.pdf_to_word,
        user=user,
        t0=t0,
    )


@router.post("/pdf-to-excel", response_model=FileResult, summary="PDF → Excel (.xlsx)")
@limiter.limit("20/hour")
async def pdf_to_excel(
    request: Request,
    file: UploadFile = File(...),
    user: dict | None = Depends(get_optional_user),
):
    """Extract tables from a PDF into an Excel spreadsheet (.xlsx)."""
    t0 = time.time()
    return await _run_conversion(
        file=file,
        allowed_mimes=PDF_ONLY,
        output_ext=".xlsx",
        output_name_suffix="_tables",
        converter_fn=ConvertService.pdf_to_excel,
        user=user,
        t0=t0,
    )


@router.post("/pdf-to-pptx", response_model=FileResult, summary="PDF → PowerPoint (.pptx)")
@limiter.limit("10/hour")
async def pdf_to_pptx(
    request: Request,
    file: UploadFile = File(...),
    dpi: int = Query(default=150, ge=72, le=300),
    user: dict | None = Depends(get_optional_user),
):
    """Convert PDF pages into PowerPoint slides."""
    t0 = time.time()
    import asyncio

    data = await validate_upload(file, allowed_mimes=PDF_ONLY, is_pro=_is_pro(user))
    upload_meta = await StorageService.save_upload(data, file.filename or "upload.pdf")
    result_path = await asyncio.to_thread(ConvertService.pdf_to_pptx, upload_meta["path"], dpi)
    out_name = f"{Path(file.filename or 'document').stem}_converted.pptx"
    output_meta = await StorageService.save_output(result_path, out_name)
    return make_file_response(
        output_meta["file_id"], out_name, output_meta["download_url"],
        output_meta["size"], t0,
    )


@router.post("/pdf-to-images", summary="PDF → Images (JPG / PNG)")
@limiter.limit("20/hour")
async def pdf_to_images(
    request: Request,
    file: UploadFile = File(...),
    format: str = Query(default="jpg", pattern="^(jpg|png|webp)$"),
    dpi: int = Query(default=150, ge=72, le=300),
    quality: int = Query(default=85, ge=10, le=100),
    pages: str | None = Query(default=None, description="Comma-separated page numbers, e.g. '1,2,3'"),
    user: dict | None = Depends(get_optional_user),
):
    """
    Convert PDF pages to images.
    Returns a list of download URLs (one per page).
    """
    import asyncio
    t0 = time.time()
    page_list = [int(p) for p in pages.split(",") if p.strip().isdigit()] if pages else None

    data = await validate_upload(file, allowed_mimes=PDF_ONLY, is_pro=_is_pro(user))
    upload_meta = await StorageService.save_upload(data, file.filename or "upload.pdf")

    image_paths: list[Path] = await asyncio.to_thread(
        ConvertService.pdf_to_images,
        upload_meta["path"], dpi, format, quality, page_list,
    )

    results = []
    for i, img_path in enumerate(image_paths):
        name = f"{Path(file.filename or 'page').stem}_page{i+1}.{format}"
        meta = await StorageService.save_output(img_path, name)
        results.append(make_file_response(
            meta["file_id"], name, meta["download_url"], meta["size"], t0
        ))

    return {"files": results, "total_pages": len(results)}


@router.post("/pdf-to-text", summary="PDF → Plain text")
@limiter.limit("30/hour")
async def pdf_to_text(
    request: Request,
    file: UploadFile = File(...),
    user: dict | None = Depends(get_optional_user),
):
    """Extract all plain text from a PDF."""
    import asyncio
    t0 = time.time()
    data = await validate_upload(file, allowed_mimes=PDF_ONLY, is_pro=_is_pro(user))
    upload_meta = await StorageService.save_upload(data, file.filename or "upload.pdf")
    text = await asyncio.to_thread(ConvertService.pdf_to_text, upload_meta["path"])
    return {"text": text, "char_count": len(text), "processing_time_ms": int((time.time() - t0) * 1000)}


# ════════════════════════════════════════════════════════════════════════════
# Other formats → PDF
# ════════════════════════════════════════════════════════════════════════════

@router.post("/office-to-pdf", response_model=FileResult, summary="Word / Excel / PPT → PDF")
@limiter.limit("20/hour")
async def office_to_pdf(
    request: Request,
    file: UploadFile = File(..., description="Word, Excel, or PowerPoint file"),
    user: dict | None = Depends(get_optional_user),
):
    """Convert Office documents (Word, Excel, PowerPoint) to PDF using LibreOffice."""
    t0 = time.time()
    return await _run_conversion(
        file=file,
        allowed_mimes=OFFICE_TYPES,
        output_ext=".pdf",
        output_name_suffix="_converted",
        converter_fn=ConvertService.office_to_pdf,
        user=user,
        t0=t0,
    )


@router.post("/images-to-pdf", response_model=FileResult, summary="Images → PDF")
@limiter.limit("20/hour")
async def images_to_pdf(
    request: Request,
    files: list[UploadFile] = File(..., description="One or more image files"),
    page_size: str = Form(default="fit"),
    user: dict | None = Depends(get_optional_user),
):
    """Combine multiple images into a single PDF document."""
    import asyncio
    t0 = time.time()

    if len(files) > 50:
        return JSONResponse(status_code=400, content={"detail": "מקסימום 50 תמונות בפעם אחת"})

    upload_paths = []
    for f in files:
        data = await validate_upload(f, allowed_mimes=IMAGE_TYPES, is_pro=_is_pro(user))
        meta = await StorageService.save_upload(data, f.filename or "image.jpg")
        upload_paths.append(meta["path"])

    result_path = await asyncio.to_thread(ConvertService.images_to_pdf, upload_paths, page_size)
    out_name = "combined.pdf"
    output_meta = await StorageService.save_output(result_path, out_name)
    return make_file_response(
        output_meta["file_id"], out_name, output_meta["download_url"],
        output_meta["size"], t0,
    )


# ── PDF info ──────────────────────────────────────────────────────────────────

@router.post("/info", summary="Get PDF metadata and page info")
@limiter.limit("60/hour")
async def pdf_info(
    request: Request,
    file: UploadFile = File(...),
    user: dict | None = Depends(get_optional_user),
):
    """Return metadata and page dimensions for a PDF."""
    import asyncio
    from ..services.pdf_service import PDFService
    data = await validate_upload(file, allowed_mimes=PDF_ONLY)
    meta = await StorageService.save_upload(data, file.filename or "upload.pdf")
    info = await asyncio.to_thread(PDFService.get_info, meta["path"])
    return info
