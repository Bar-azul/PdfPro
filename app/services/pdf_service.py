"""
PDFService
==========
Core PDF manipulation using PyMuPDF (fitz).
Handles: merge, split, compress, rotate, watermark, password, redact.
"""

import io
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF

from ..config import settings

logger = logging.getLogger(__name__)

# ── Compression settings ──────────────────────────────────────────────────────
COMPRESS_PRESETS = {
    "low":     {"deflate": True, "clean": True, "garbage": 1, "linear": False},
    "medium":  {"deflate": True, "clean": True, "garbage": 2, "linear": True},
    "high":    {"deflate": True, "clean": True, "garbage": 3, "linear": True},
    "extreme": {"deflate": True, "clean": True, "garbage": 4, "linear": True},
}

IMAGE_QUALITY = {
    "low": 40, "medium": 60, "high": 40, "extreme": 25,
}


class PDFService:

    # ── Merge ──────────────────────────────────────────────────────────────────

    @staticmethod
    def merge(pdf_paths: list[Path]) -> Path:
        """Merge multiple PDFs into a single file."""
        t0 = time.time()
        result = fitz.open()

        for path in pdf_paths:
            with fitz.open(path) as src:
                result.insert_pdf(src)

        out = _temp_pdf("merged")
        result.save(out, deflate=True, garbage=2)
        result.close()

        logger.info(f"Merged {len(pdf_paths)} PDFs → {out} in {_ms(t0)}ms")
        return out

    # ── Split ──────────────────────────────────────────────────────────────────

    @staticmethod
    def split_by_ranges(pdf_path: Path, ranges: list[str]) -> list[Path]:
        """
        Split a PDF by page ranges.
        ranges example: ["1-3", "4", "5-7"]  (1-based, inclusive)
        """
        t0 = time.time()
        outputs = []

        with fitz.open(pdf_path) as src:
            total = src.page_count
            for i, rng in enumerate(ranges):
                pages = _parse_range(rng, total)
                part = fitz.open()
                part.insert_pdf(src, from_page=pages[0], to_page=pages[-1])
                out = _temp_pdf(f"split_part{i+1}")
                part.save(out, deflate=True)
                part.close()
                outputs.append(out)

        logger.info(f"Split into {len(outputs)} parts in {_ms(t0)}ms")
        return outputs

    @staticmethod
    def split_every_n(pdf_path: Path, n: int) -> list[Path]:
        """Split a PDF into chunks of N pages each."""
        t0 = time.time()
        outputs = []

        with fitz.open(pdf_path) as src:
            total = src.page_count
            chunk_start = 0
            i = 0
            while chunk_start < total:
                chunk_end = min(chunk_start + n - 1, total - 1)
                part = fitz.open()
                part.insert_pdf(src, from_page=chunk_start, to_page=chunk_end)
                out = _temp_pdf(f"split_chunk{i+1}")
                part.save(out, deflate=True)
                part.close()
                outputs.append(out)
                chunk_start += n
                i += 1

        logger.info(f"Split every {n} pages → {len(outputs)} parts in {_ms(t0)}ms")
        return outputs

    @staticmethod
    def extract_pages(pdf_path: Path, pages: list[int]) -> Path:
        """Extract specific pages (1-based) into a new PDF."""
        t0 = time.time()
        with fitz.open(pdf_path) as src:
            result = fitz.open()
            for p in pages:
                if 1 <= p <= src.page_count:
                    result.insert_pdf(src, from_page=p - 1, to_page=p - 1)
            out = _temp_pdf("extracted")
            result.save(out, deflate=True)
            result.close()
        logger.info(f"Extracted {len(pages)} pages in {_ms(t0)}ms")
        return out

    # ── Compress ───────────────────────────────────────────────────────────────

    @staticmethod
    def compress(pdf_path: Path, level: str = "medium") -> Path:
        """Compress a PDF by downsampling images and cleaning the file."""
        t0 = time.time()
        preset = COMPRESS_PRESETS.get(level, COMPRESS_PRESETS["medium"])
        quality = IMAGE_QUALITY.get(level, 60)

        with fitz.open(pdf_path) as doc:
            # Recompress embedded images
            for page in doc:
                image_list = page.get_images(full=True)
                for img in image_list:
                    xref = img[0]
                    try:
                        base_img = doc.extract_image(xref)
                        img_bytes = base_img["image"]
                        from PIL import Image

                        pil_img = Image.open(io.BytesIO(img_bytes))
                        if pil_img.mode in ("RGBA", "P"):
                            pil_img = pil_img.convert("RGB")

                        buf = io.BytesIO()
                        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
                        doc.update_stream(xref, buf.getvalue())
                    except Exception:
                        pass  # Skip images we can't recompress

            out = _temp_pdf("compressed")
            doc.save(
                out,
                deflate=preset["deflate"],
                clean=preset["clean"],
                garbage=preset["garbage"],
                linear=preset["linear"],
            )

        original_size = pdf_path.stat().st_size
        compressed_size = out.stat().st_size
        ratio = (1 - compressed_size / original_size) * 100 if original_size else 0
        logger.info(
            f"Compressed ({level}): {original_size:,}B → {compressed_size:,}B "
            f"({ratio:.1f}% reduction) in {_ms(t0)}ms"
        )
        return out

    # ── Rotate ─────────────────────────────────────────────────────────────────

    @staticmethod
    def rotate(pdf_path: Path, angle: int, pages: list[int] | None = None) -> Path:
        """Rotate pages by 90, 180, or 270 degrees."""
        t0 = time.time()
        with fitz.open(pdf_path) as doc:
            target_pages = [p - 1 for p in pages] if pages else range(doc.page_count)
            for i in target_pages:
                if 0 <= i < doc.page_count:
                    doc[i].set_rotation(angle)
            out = _temp_pdf("rotated")
            doc.save(out, deflate=True)
        logger.info(f"Rotated {angle}° in {_ms(t0)}ms")
        return out

    # ── Watermark ──────────────────────────────────────────────────────────────

    @staticmethod
    def add_text_watermark(
        pdf_path: Path,
        text: str,
        opacity: float = 0.3,
        font_size: int = 48,
        color: tuple = (0.7, 0.7, 0.7),
        rotation: int = -45,
        position: str = "center",
        pages: list[int] | None = None,
    ) -> Path:
        t0 = time.time()
        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)
            for i in target:
                if 0 <= i < doc.page_count:
                    page = doc[i]
                    rect = page.rect
                    # Center point
                    cx = rect.width / 2
                    cy = rect.height / 2

                    # Insert text as a transparent annotation
                    page.insert_text(
                        fitz.Point(cx - font_size * len(text) * 0.3, cy),
                        text,
                        fontsize=font_size,
                        color=color,
                        rotate=rotation,
                        render_mode=0,
                        overlay=True,
                    )

            out = _temp_pdf("watermarked")
            doc.save(out, deflate=True)
        logger.info(f"Watermark added in {_ms(t0)}ms")
        return out

    @staticmethod
    def add_image_watermark(
        pdf_path: Path,
        image_path: Path,
        opacity: float = 0.3,
        position: str = "center",
        pages: list[int] | None = None,
    ) -> Path:
        t0 = time.time()
        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)
            for i in target:
                if 0 <= i < doc.page_count:
                    page = doc[i]
                    rect = page.rect
                    wm_w = rect.width * 0.4
                    wm_h = rect.height * 0.4
                    wm_rect = fitz.Rect(
                        (rect.width - wm_w) / 2,
                        (rect.height - wm_h) / 2,
                        (rect.width + wm_w) / 2,
                        (rect.height + wm_h) / 2,
                    )
                    page.insert_image(wm_rect, filename=str(image_path), overlay=True)

            out = _temp_pdf("watermarked")
            doc.save(out, deflate=True)
        logger.info(f"Image watermark added in {_ms(t0)}ms")
        return out

    # ── Password / Security ────────────────────────────────────────────────────

    @staticmethod
    def protect(
        pdf_path: Path,
        password: str,
        owner_password: str | None = None,
        allow_print: bool = True,
        allow_copy: bool = False,
        allow_edit: bool = False,
    ) -> Path:
        """Encrypt PDF with a user password."""
        t0 = time.time()
        owner_pw = owner_password or password + "_owner"

        perm = fitz.PDF_PERM_ACCESSIBILITY
        if allow_print:
            perm |= fitz.PDF_PERM_PRINT | fitz.PDF_PERM_PRINT_HQ
        if allow_copy:
            perm |= fitz.PDF_PERM_COPY
        if allow_edit:
            perm |= fitz.PDF_PERM_MODIFY | fitz.PDF_PERM_ANNOTATE

        with fitz.open(pdf_path) as doc:
            out = _temp_pdf("protected")
            doc.save(
                out,
                encryption=fitz.PDF_ENCRYPT_AES_256,
                user_pw=password,
                owner_pw=owner_pw,
                permissions=perm,
                deflate=True,
            )
        logger.info(f"PDF protected in {_ms(t0)}ms")
        return out

    @staticmethod
    def unlock(pdf_path: Path, password: str) -> Path:
        """Remove password protection (requires correct password)."""
        t0 = time.time()
        with fitz.open(pdf_path) as doc:
            if doc.is_encrypted:
                success = doc.authenticate(password)
                if not success:
                    raise ValueError("Incorrect password")
            out = _temp_pdf("unlocked")
            doc.save(out, encryption=fitz.PDF_ENCRYPT_NONE, deflate=True)
        logger.info(f"PDF unlocked in {_ms(t0)}ms")
        return out

    # ── Redact ─────────────────────────────────────────────────────────────────

    @staticmethod
    def redact(
        pdf_path: Path,
        texts: list[str],
        case_sensitive: bool = False,
        pages: list[int] | None = None,
    ) -> Path:
        """Black-out all occurrences of the given text strings."""
        t0 = time.time()
        flags = 0 if case_sensitive else fitz.TEXT_SEARCH_IGNORECASE
        total_redactions = 0

        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)
            for i in target:
                if 0 <= i < doc.page_count:
                    page = doc[i]
                    for text in texts:
                        rects = page.search_for(text, quads=False)
                        for rect in rects:
                            page.add_redact_annot(rect, fill=(0, 0, 0))
                            total_redactions += 1
                    page.apply_redactions()

            out = _temp_pdf("redacted")
            doc.save(out, deflate=True)

        logger.info(f"Redacted {total_redactions} occurrences in {_ms(t0)}ms")
        return out

    # ── Metadata ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_info(pdf_path: Path) -> dict:
        """Return metadata and page info for a PDF."""
        with fitz.open(pdf_path) as doc:
            meta = doc.metadata
            return {
                "page_count": doc.page_count,
                "is_encrypted": doc.is_encrypted,
                "has_forms": doc.is_pdf and bool(doc.get_sigflags() != -1),
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "creator": meta.get("creator", ""),
                "producer": meta.get("producer", ""),
                "creation_date": meta.get("creationDate", ""),
                "pages": [
                    {
                        "page": i + 1,
                        "width_pt": round(doc[i].rect.width, 2),
                        "height_pt": round(doc[i].rect.height, 2),
                    }
                    for i in range(doc.page_count)
                ],
            }

    # ── Render pages as images ─────────────────────────────────────────────────

    @staticmethod
    def render_pages(
        pdf_path: Path,
        dpi: int = 150,
        fmt: str = "jpg",
        quality: int = 85,
        pages: list[int] | None = None,
    ) -> list[Path]:
        """Render PDF pages as images. Returns list of image file paths."""
        t0 = time.time()
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        outputs = []

        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)
            for i in target:
                if 0 <= i < doc.page_count:
                    pix = doc[i].get_pixmap(matrix=matrix, alpha=False)
                    out = _temp_file(f"page{i+1}", f".{fmt}")
                    if fmt == "jpg":
                        pix.save(str(out), jpg_quality=quality)
                    else:
                        pix.save(str(out))
                    outputs.append(out)

        logger.info(f"Rendered {len(outputs)} pages at {dpi}dpi in {_ms(t0)}ms")
        return outputs


# ── Helpers ────────────────────────────────────────────────────────────────────

def _temp_pdf(prefix: str) -> Path:
    import uuid
    p = Path(tempfile.gettempdir()) / f"{prefix}_{uuid.uuid4().hex[:8]}.pdf"
    return p


def _temp_file(prefix: str, suffix: str) -> Path:
    import uuid
    p = Path(tempfile.gettempdir()) / f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
    return p


def _ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)


def _parse_range(rng: str, total: int) -> list[int]:
    """Convert "3-7" → [2,3,4,5,6] (0-based)."""
    rng = rng.strip()
    if "-" in rng:
        parts = rng.split("-", 1)
        start = max(1, int(parts[0])) - 1
        end = min(total, int(parts[1])) - 1
        return list(range(start, end + 1))
    else:
        p = int(rng) - 1
        if 0 <= p < total:
            return [p]
        return []
