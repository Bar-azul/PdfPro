"""
Edit router — /api/edit/*
Watermark, digital signature, password protection, redaction.
"""

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Request, File, Form, UploadFile

from ..middleware.rate_limit import limiter
from ..models.schemas import FileResult
from .auth import get_optional_user
from ..services.pdf_service import PDFService
from ..services.storage_service import StorageService
from ..utils.file_utils import PDF_ONLY, IMAGE_TYPES, make_file_response, validate_upload

router = APIRouter()


def _is_pro(user): return user is not None and user.get("plan") in ("pro", "enterprise")


def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert '#RRGGBB' to (r, g, b) floats 0-1."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (r / 255, g / 255, b / 255)


# ── Watermark ──────────────────────────────────────────────────────────────────

@router.post("/watermark", response_model=FileResult, summary="Add text or image watermark")
@limiter.limit("20/hour")
async def add_watermark(
    request: Request,
    file: UploadFile = File(..., description="PDF file"),
    type: str = Form(default="text", description="'text' or 'image'"),
    text: str | None = Form(default=None, description="Watermark text (for type=text)"),
    opacity: float = Form(default=0.3),
    font_size: int = Form(default=48),
    color: str = Form(default="#CCCCCC", description="Hex color, e.g. '#CCCCCC'"),
    rotation: int = Form(default=-45),
    pages: str | None = Form(default=None, description="Comma-separated page numbers. Empty = all."),
    watermark_image: UploadFile | None = File(default=None, description="Image file (for type=image)"),
    user: dict | None = Depends(get_optional_user),
):
    """Add a text or image watermark to a PDF."""
    t0 = time.time()
    page_list = [int(p) for p in pages.split(",") if p.strip().isdigit()] if pages else None

    data = await validate_upload(file, allowed_mimes=PDF_ONLY, is_pro=_is_pro(user))
    upload_meta = await StorageService.save_upload(data, file.filename or "upload.pdf")

    if type == "image" and watermark_image:
        wm_data = await validate_upload(watermark_image, allowed_mimes=IMAGE_TYPES)
        wm_meta = await StorageService.save_upload(wm_data, watermark_image.filename or "wm.png")
        result_path = await asyncio.to_thread(
            PDFService.add_image_watermark,
            upload_meta["path"], wm_meta["path"], opacity, "center", page_list,
        )
    else:
        if not text:
            from fastapi import HTTPException
            raise HTTPException(400, detail="נדרש טקסט לסימן מים מסוג 'text'")
        rgb = _hex_to_rgb(color)
        result_path = await asyncio.to_thread(
            PDFService.add_text_watermark,
            upload_meta["path"], text, opacity, font_size, rgb, rotation, "center", page_list,
        )

    out_name = f"{Path(file.filename or 'watermarked').stem}_watermarked.pdf"
    output_meta = await StorageService.save_output(result_path, out_name)
    return make_file_response(
        output_meta["file_id"], out_name, output_meta["download_url"],
        output_meta["size"], t0,
    )


# ── Digital Signature ──────────────────────────────────────────────────────────

@router.post("/sign", response_model=FileResult, summary="Add a digital signature")
@limiter.limit("20/hour")
async def sign_pdf(
    request: Request,
    file: UploadFile = File(...),
    signature_text: str | None = Form(default=None, description="Name to render as a typed signature"),
    signature_image: UploadFile | None = File(default=None, description="Drawn signature (PNG, transparent background)"),
    x: float = Form(default=0.6, description="Horizontal position of the box's left edge (0.0–1.0)"),
    y: float = Form(default=0.85, description="Vertical position of the box's top edge (0.0–1.0)"),
    page: int = Form(default=-1, description="Page number (1-based). -1 = last page."),
    add_date: bool = Form(default=True, description="Print today's date under the signature"),
    user: dict | None = Depends(get_optional_user),
):
    """Add a visible signature to a PDF page: a drawn signature image, or a typed name."""
    import fitz
    from fastapi import HTTPException
    t0 = time.time()

    text = (signature_text or "").strip()
    if signature_image is None and not text:
        raise HTTPException(400, detail="Provide signature_image or signature_text")

    data = await validate_upload(file, allowed_mimes=PDF_ONLY, is_pro=_is_pro(user))
    img_bytes = None
    if signature_image is not None:
        img_bytes = await validate_upload(signature_image, allowed_mimes={"image/png", "image/jpeg"}, max_mb=2)
    upload_meta = await StorageService.save_upload(data, file.filename or "upload.pdf")

    def _sign(pdf_path: Path) -> Path:
        from ..services.pdf_service import _temp_pdf
        with fitz.open(pdf_path) as doc:
            page_idx = (doc.page_count - 1) if page == -1 else (page - 1)
            page_idx = max(0, min(page_idx, doc.page_count - 1))
            pg = doc[page_idx]

            r = pg.rect
            sig_w = r.width * 0.25
            sig_h = 50 if img_bytes else 40
            # keep the whole box (and the date line) inside the page
            sig_x = min(max(r.width * x, 10), r.width - sig_w - 10)
            sig_y = min(max(r.height * y, 10), r.height - sig_h - 20)
            rect = fitz.Rect(sig_x, sig_y, sig_x + sig_w, sig_y + sig_h)

            if img_bytes:
                # Drawn signature: placed as-is, no frame, like ink on paper
                try:
                    pg.insert_image(rect, stream=img_bytes, keep_proportion=True, overlay=True)
                except Exception:
                    raise ValueError("bad_image")
            else:
                pg.draw_rect(rect, color=(0, 0, 0.6), width=0.5)
                # insert_htmlbox handles Hebrew/Arabic fonts and RTL order
                # (insert_text's built-in font has no Hebrew glyphs and renders dots).
                import html as _html
                pg.insert_htmlbox(
                    fitz.Rect(sig_x + 4, sig_y + 2, sig_x + sig_w - 4, sig_y + sig_h - 2),
                    f'<div dir="auto" style="font-size:18px;color:#0000b3;text-align:center">'
                    f'{_html.escape(text, quote=False)}</div>',
                )

            if add_date:
                from datetime import datetime
                pg.insert_text(
                    fitz.Point(sig_x + 6, sig_y + sig_h + 12),
                    datetime.now().strftime("%d/%m/%Y"),
                    fontsize=8,
                    color=(0.4, 0.4, 0.4),
                )

            out = _temp_pdf("signed")
            doc.save(out, deflate=True)
        return out

    try:
        result_path = await asyncio.to_thread(_sign, upload_meta["path"])
    except ValueError:
        raise HTTPException(400, detail="The signature image could not be read")
    out_name = f"{Path(file.filename or 'signed').stem}_signed.pdf"
    output_meta = await StorageService.save_output(result_path, out_name)
    return make_file_response(
        output_meta["file_id"], out_name, output_meta["download_url"],
        output_meta["size"], t0,
    )


# ── Password Protection ────────────────────────────────────────────────────────

@router.post("/protect", response_model=FileResult, summary="Password-protect a PDF")
@limiter.limit("20/hour")
async def protect_pdf(
    request: Request,
    file: UploadFile = File(...),
    password: str = Form(..., min_length=4, description="Password to set"),
    owner_password: str | None = Form(default=None),
    allow_print: bool = Form(default=True),
    allow_copy: bool = Form(default=False),
    allow_edit: bool = Form(default=False),
    user: dict | None = Depends(get_optional_user),
):
    """Encrypt a PDF with AES-256 password protection."""
    t0 = time.time()
    data = await validate_upload(file, allowed_mimes=PDF_ONLY, is_pro=_is_pro(user))
    upload_meta = await StorageService.save_upload(data, file.filename or "upload.pdf")
    result_path = await asyncio.to_thread(
        PDFService.protect,
        upload_meta["path"], password, owner_password, allow_print, allow_copy, allow_edit,
    )
    out_name = f"{Path(file.filename or 'protected').stem}_protected.pdf"
    output_meta = await StorageService.save_output(result_path, out_name)
    return make_file_response(
        output_meta["file_id"], out_name, output_meta["download_url"],
        output_meta["size"], t0,
    )


@router.post("/unlock", response_model=FileResult, summary="Remove PDF password")
@limiter.limit("20/hour")
async def unlock_pdf(
    request: Request,
    file: UploadFile = File(...),
    password: str = Form(..., description="Current password"),
    user: dict | None = Depends(get_optional_user),
):
    """Remove password protection from a PDF (requires the correct password)."""
    t0 = time.time()
    data = await validate_upload(file, allowed_mimes=PDF_ONLY, is_pro=_is_pro(user))
    upload_meta = await StorageService.save_upload(data, file.filename or "upload.pdf")
    try:
        result_path = await asyncio.to_thread(PDFService.unlock, upload_meta["path"], password)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(403, detail="סיסמה שגויה")
    out_name = f"{Path(file.filename or 'unlocked').stem}_unlocked.pdf"
    output_meta = await StorageService.save_output(result_path, out_name)
    return make_file_response(
        output_meta["file_id"], out_name, output_meta["download_url"],
        output_meta["size"], t0,
    )


# ── Redact ─────────────────────────────────────────────────────────────────────

@router.post("/redact", response_model=FileResult, summary="Black out sensitive text")
@limiter.limit("20/hour")
async def redact_pdf(
    request: Request,
    file: UploadFile = File(...),
    texts: str = Form(..., description="Comma-separated list of text strings to redact"),
    case_sensitive: bool = Form(default=False),
    pages: str | None = Form(default=None),
    user: dict | None = Depends(get_optional_user),
):
    """Permanently black out all occurrences of the specified text strings."""
    t0 = time.time()
    text_list = [t.strip() for t in texts.split(",") if t.strip()]
    page_list = [int(p) for p in pages.split(",") if p.strip().isdigit()] if pages else None

    data = await validate_upload(file, allowed_mimes=PDF_ONLY, is_pro=_is_pro(user))
    upload_meta = await StorageService.save_upload(data, file.filename or "upload.pdf")
    result_path = await asyncio.to_thread(
        PDFService.redact, upload_meta["path"], text_list, case_sensitive, page_list,
    )
    out_name = f"{Path(file.filename or 'redacted').stem}_redacted.pdf"
    output_meta = await StorageService.save_output(result_path, out_name)
    return make_file_response(
        output_meta["file_id"], out_name, output_meta["download_url"],
        output_meta["size"], t0,
    )
