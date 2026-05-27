"""
Organize router — /api/organize/*
Merge, split, compress, rotate PDF files.
"""

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Request, File, Form, Query, UploadFile

from ..middleware.rate_limit import limiter
from ..models.schemas import CompressRequest, FileResult, MergeRequest, RotateRequest, SplitRequest, SplitResult
from .auth import get_optional_user
from ..services.pdf_service import PDFService
from ..services.storage_service import StorageService
from ..utils.file_utils import PDF_ONLY, make_file_response, validate_upload

router = APIRouter()


def _is_pro(user): return user is not None and user.get("plan") in ("pro", "enterprise")


# ── Merge ──────────────────────────────────────────────────────────────────────

@router.post("/merge", response_model=FileResult, summary="Merge multiple PDFs into one")
@limiter.limit("15/hour")
async def merge_pdfs(
    request: Request,
    files: list[UploadFile] = File(..., description="2–20 PDF files to merge, in order"),
    user: dict | None = Depends(get_optional_user),
):
    """Merge 2 to 20 PDF files into a single PDF, preserving all pages."""
    t0 = time.time()

    if len(files) < 2:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="נדרשים לפחות 2 קבצי PDF למיזוג")
    if len(files) > 20:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="מקסימום 20 קבצים לפעולה אחת")

    upload_paths: list[Path] = []
    for f in files:
        data = await validate_upload(f, allowed_mimes=PDF_ONLY, is_pro=_is_pro(user))
        meta = await StorageService.save_upload(data, f.filename or "upload.pdf")
        upload_paths.append(meta["path"])

    result_path = await asyncio.to_thread(PDFService.merge, upload_paths)
    out_name = "merged.pdf"
    output_meta = await StorageService.save_output(result_path, out_name)

    return make_file_response(
        output_meta["file_id"], out_name, output_meta["download_url"],
        output_meta["size"], t0,
    )


# ── Split ──────────────────────────────────────────────────────────────────────

@router.post("/split", summary="Split a PDF into multiple files")
@limiter.limit("15/hour")
async def split_pdf(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form(default="ranges", description="'ranges', 'every_n', or 'pages'"),
    ranges: str | None = Form(default=None, description="e.g. '1-3,4-6,7-9'"),
    every_n: int | None = Form(default=None, description="Split every N pages"),
    pages: str | None = Form(default=None, description="Comma-separated page numbers"),
    user: dict | None = Depends(get_optional_user),
):
    """
    Split a PDF by page ranges, every N pages, or extract specific pages.

    Examples:
    - ranges="1-3,4-6"  → Part 1 (pp.1-3), Part 2 (pp.4-6)
    - every_n=5         → Chunks of 5 pages each
    - pages="1,5,10"    → Extract pages 1, 5, and 10 into one file
    """
    t0 = time.time()
    data = await validate_upload(file, allowed_mimes=PDF_ONLY, is_pro=_is_pro(user))
    upload_meta = await StorageService.save_upload(data, file.filename or "upload.pdf")
    stem = Path(file.filename or "split").stem

    if mode == "ranges":
        if not ranges:
            from fastapi import HTTPException
            raise HTTPException(400, detail="'ranges' חובה למצב 'ranges'")
        range_list = [r.strip() for r in ranges.split(",") if r.strip()]
        result_paths = await asyncio.to_thread(
            PDFService.split_by_ranges, upload_meta["path"], range_list
        )
    elif mode == "every_n":
        if not every_n or every_n < 1:
            from fastapi import HTTPException
            raise HTTPException(400, detail="'every_n' חובה להיות מספר חיובי")
        result_paths = await asyncio.to_thread(
            PDFService.split_every_n, upload_meta["path"], every_n
        )
    elif mode == "pages":
        if not pages:
            from fastapi import HTTPException
            raise HTTPException(400, detail="'pages' חובה למצב 'pages'")
        page_list = [int(p) for p in pages.split(",") if p.strip().isdigit()]
        result_paths = [
            await asyncio.to_thread(PDFService.extract_pages, upload_meta["path"], page_list)
        ]
    else:
        from fastapi import HTTPException
        raise HTTPException(400, detail=f"מצב לא מוכר: {mode}")

    parts = []
    for i, path in enumerate(result_paths):
        out_name = f"{stem}_part{i+1}.pdf"
        meta = await StorageService.save_output(path, out_name)
        parts.append(make_file_response(
            meta["file_id"], out_name, meta["download_url"], meta["size"], t0
        ))

    return {"parts": parts, "total_parts": len(parts)}


# ── Compress ───────────────────────────────────────────────────────────────────

@router.post("/compress", response_model=FileResult, summary="Compress a PDF")
@limiter.limit("20/hour")
async def compress_pdf(
    request: Request,
    file: UploadFile = File(...),
    level: str = Form(default="medium", description="low | medium | high | extreme"),
    user: dict | None = Depends(get_optional_user),
):
    """
    Reduce PDF file size by recompressing images and cleaning the document structure.

    Compression levels:
    - **low** → ~10% reduction, near-lossless
    - **medium** → ~40% reduction, minimal quality loss
    - **high** → ~65% reduction, visible on images
    - **extreme** → ~80% reduction, best for text-only documents
    """
    t0 = time.time()
    if level not in ("low", "medium", "high", "extreme"):
        from fastapi import HTTPException
        raise HTTPException(400, detail=f"רמת דחיסה לא חוקית: {level}")

    data = await validate_upload(file, allowed_mimes=PDF_ONLY, is_pro=_is_pro(user))
    original_size = len(data)
    upload_meta = await StorageService.save_upload(data, file.filename or "upload.pdf")

    result_path = await asyncio.to_thread(PDFService.compress, upload_meta["path"], level)
    out_name = f"{Path(file.filename or 'compressed').stem}_compressed.pdf"
    output_meta = await StorageService.save_output(result_path, out_name)

    compressed_size = output_meta["size"]
    saved_pct = round((1 - compressed_size / original_size) * 100, 1) if original_size else 0

    return {
        **make_file_response(
            output_meta["file_id"], out_name, output_meta["download_url"],
            compressed_size, t0
        ),
        "original_size_bytes": original_size,
        "saved_percentage": saved_pct,
    }


# ── Rotate ─────────────────────────────────────────────────────────────────────

@router.post("/rotate", response_model=FileResult, summary="Rotate PDF pages")
@limiter.limit("30/hour")
async def rotate_pdf(
    request: Request,
    file: UploadFile = File(...),
    angle: int = Form(..., description="Rotation angle: 90, 180, or 270"),
    pages: str | None = Form(default=None, description="Comma-separated page numbers. Empty = all."),
    user: dict | None = Depends(get_optional_user),
):
    """Rotate all or selected PDF pages by 90, 180, or 270 degrees."""
    t0 = time.time()
    if angle not in (90, 180, 270):
        from fastapi import HTTPException
        raise HTTPException(400, detail="זווית חייבת להיות 90, 180, או 270")

    page_list = [int(p) for p in pages.split(",") if p.strip().isdigit()] if pages else None

    data = await validate_upload(file, allowed_mimes=PDF_ONLY, is_pro=_is_pro(user))
    upload_meta = await StorageService.save_upload(data, file.filename or "upload.pdf")
    result_path = await asyncio.to_thread(
        PDFService.rotate, upload_meta["path"], angle, page_list
    )
    out_name = f"{Path(file.filename or 'rotated').stem}_rotated.pdf"
    output_meta = await StorageService.save_output(result_path, out_name)
    return make_file_response(
        output_meta["file_id"], out_name, output_meta["download_url"],
        output_meta["size"], t0,
    )
