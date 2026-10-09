"""
PDFService
==========
Core PDF manipulation using PyMuPDF (fitz).
Handles: merge, split, compress, rotate, watermark, password, redact.
"""

import io
import logging
import math
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF
from PIL import Image

from ..config import settings

logger = logging.getLogger(__name__)

# ── Compression settings ──────────────────────────────────────────────────────
# (dpi_threshold, dpi_target, jpeg_quality): images shown above the threshold are
# downsampled to the target resolution, then re-encoded as JPEG at that quality.
# Higher level = smaller file. Text and vector graphics are never touched.
COMPRESS_LEVELS = {
    "low":     (300, 220, 85),
    "medium":  (200, 150, 75),
    "high":    (150, 120, 60),
    "extreme": (120, 96, 45),
}
COMPRESS_MIN_IMAGE_BYTES = 20_000   # icons and small logos aren't worth re-encoding


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
        """Shrink a PDF by downsampling and re-encoding its images.

        Each image is decoded with its real colour space (RGB, gray, CMYK, ICC, inverted
        CMYK), downsampled only if it's shown above the level's DPI threshold, and stored
        as a plain JPEG with a matching image dictionary. Images with transparency,
        stencil masks or 1-bit scans are left untouched. If the result isn't smaller,
        or loses pages, the original file is returned unchanged.
        """
        t0 = time.time()
        threshold, target, quality = COMPRESS_LEVELS.get(level, COMPRESS_LEVELS["medium"])
        out = _temp_pdf("compressed")

        with fitz.open(pdf_path) as doc:
            page_count = doc.page_count
            widths = _image_display_widths(doc)
            seen: set[int] = set()
            for page in doc:
                for item in page.get_images(full=True):
                    xref = item[0]
                    if xref in seen:            # shared images (a logo on every page) once only
                        continue
                    seen.add(xref)
                    try:
                        _recompress_image(doc, xref, widths.get(xref), threshold, target, quality)
                    except Exception as exc:    # leave that image exactly as it was
                        logger.warning(f"compress: skipped image xref {xref}: {exc}")
                    fitz.TOOLS.store_shrink(100)  # drop MuPDF's decoded-image cache, keeps memory flat
            doc.save(out, garbage=3, deflate=True, use_objstms=1)

        with fitz.open(out) as check:
            pages_ok = check.page_count == page_count
        original_size = pdf_path.stat().st_size
        if not pages_ok or out.stat().st_size >= original_size:
            import shutil
            shutil.copyfile(pdf_path, out)      # never hand back something broken or bigger

        compressed_size = out.stat().st_size
        ratio = (1 - compressed_size / original_size) * 100 if original_size else 0
        logger.info(
            f"Compressed ({level}): {original_size:,}B → {compressed_size:,}B "
            f"({ratio:.1f}% reduction) in {_ms(t0)}ms"
        )
        return out

    @staticmethod
    def rotate(pdf_path: Path, angle: int, pages: list[int] | None = None) -> Path:
        """Rotate pages by 90, 180, or 270 degrees."""
        t0 = time.time()
        with fitz.open(pdf_path) as doc:
            target_pages = [p - 1 for p in pages] if pages else range(doc.page_count)
            for i in target_pages:
                if 0 <= i < doc.page_count:
                    # relative to the current rotation (scans often already carry /Rotate)
                    doc[i].set_rotation((doc[i].rotation + angle) % 360)
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
        """Diagonal text watermark. Supports Hebrew/Arabic (RTL) and any rotation angle.

        The text is rendered once with insert_htmlbox (font fallback + bidi), then
        stamped onto each page with show_pdf_page, which accepts arbitrary angles.
        """
        t0 = time.time()
        opacity = max(0.05, min(float(opacity), 1.0))
        hex_color = "#%02x%02x%02x" % tuple(int(max(0, min(c, 1)) * 255) for c in color)

        stamp = fitz.open()
        sp = stamp.new_page(width=3000, height=font_size * 3)
        sp.insert_htmlbox(
            sp.rect,
            f'<div dir="auto" style="font-size:{font_size}px;color:{hex_color};'
            f'white-space:nowrap;text-align:center;font-weight:bold">{_html_escape(text)}</div>',
            opacity=opacity,
        )
        clip = fitz.Rect()
        for b in sp.get_text("blocks"):
            clip |= fitz.Rect(b[:4])
        if clip.is_empty:
            raise ValueError("Watermark text could not be rendered")
        clip = (clip + (-4, -4, 4, 4)) & sp.rect

        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)
            for i in target:
                if 0 <= i < doc.page_count:
                    page = doc[i]
                    r = page.rect
                    box = fitz.Rect(r.x0 + r.width * 0.08, r.y0 + r.height * 0.08,
                                    r.x1 - r.width * 0.08, r.y1 - r.height * 0.08)
                    # show_pdf_page rotates counter-clockwise; the API's rotation is clockwise-negative
                    page.show_pdf_page(box, stamp, 0, clip=clip, rotate=-rotation, overlay=True)
            out = _temp_pdf("watermarked")
            doc.save(out, deflate=True)
        stamp.close()
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
        # random owner password: knowing the open password must not unlock the permissions
        owner_pw = owner_password or secrets.token_urlsafe(24)

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
        # search_for() is case-insensitive; for case-sensitive requests keep only exact hits.
        total_redactions = 0

        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)
            for i in target:
                if 0 <= i < doc.page_count:
                    page = doc[i]
                    for text in texts:
                        rects = page.search_for(text, quads=False)
                        if case_sensitive:
                            # Drop only hits that are clearly a different-case match on one line.
                            # Fragments of a match split across lines don't contain the whole text,
                            # so they are kept: when in doubt, redact rather than leak.
                            def _wrong_case(r):
                                """True if this hit's line holds the text, but only in another case."""
                                box = page.get_textbox(r + (-1, -1, 1, 1))
                                return text.lower() in box.lower() and text not in box
                            rects = [r for r in rects if not _wrong_case(r)]
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

# ── Compression helpers ───────────────────────────────────────────────────────

def _image_display_widths(doc):
    """xref -> largest width (in points) the image is drawn at anywhere in the document.
    The largest placement needs the most pixels, so it decides how far we may downsample.
    Uses get_image_info() without hashes, which doesn't decode the images (cheap on big scans);
    placements are matched to xrefs by pixel size."""
    widths = {}
    for page in doc:
        by_size = {}
        for item in page.get_images(full=True):
            by_size.setdefault((item[2], item[3]), []).append(item[0])
        for info in page.get_image_info():
            a, b = info["transform"][0], info["transform"][1]
            shown = (a * a + b * b) ** 0.5          # drawn width of the image's x axis, handles rotation
            for xref in by_size.get((info["width"], info["height"]), []):
                widths[xref] = max(widths.get(xref, 0.0), shown)
    return widths


def _scale_for(width_px, shown_width_pt, threshold, target):
    """Resize factor for an image: target/actual DPI if shown above the threshold, else 1."""
    dpi = width_px / (shown_width_pt / 72.0)
    return target / dpi if dpi > threshold else 1.0


def _finish(img, scale):
    """Apply the remaining resize (LANCZOS) after any cheap pre-shrink; no-op near 1.0."""
    if scale < 0.98:
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        if img.size != size:
            img = img.resize(size, Image.LANCZOS)
    return img


def _image_colorspace(doc, xref):
    """('DeviceRGB'|'DeviceGray'|'ICCBased', components, icc_profile_bytes) or None for anything else
    (CMYK, Lab, Separation, Indexed...), which must go through MuPDF's colour conversion."""
    kind, value = doc.xref_get_key(xref, "ColorSpace")
    if kind == "name" and value in ("/DeviceRGB", "/DeviceGray"):
        return value[1:], (3 if value == "/DeviceRGB" else 1), None
    text = value
    if kind == "xref":
        text = doc.xref_object(int(value.split()[0]), compressed=True)
    m = re.search(r"/ICCBased\s*(\d+)\s+0\s+R", text or "")
    if not m:
        return None
    icc = int(m.group(1))
    n = doc.xref_get_key(icc, "N")[1]
    if n not in ("1", "3"):
        return None
    return "ICCBased", int(n), doc.xref_stream(icc)


def _decode_jpeg_scaled(doc, xref, shown_width_pt, threshold, target):
    """Plain RGB/gray JPEGs (most scans and photos): let libjpeg decode straight to a smaller
    size (draft mode), so a 600 dpi page never has to sit in memory at full resolution.
    Only for DeviceRGB/DeviceGray/ICC RGB-or-gray; ICC RGB is converted to sRGB afterwards."""
    if doc.xref_get_key(xref, "Filter")[1] != "/DCTDecode" or doc.xref_get_key(xref, "Decode")[0] != "null":
        return None
    cs = _image_colorspace(doc, xref)
    if cs is None:
        return None
    name, n, icc = cs
    img = Image.open(io.BytesIO(doc.xref_stream_raw(xref)))
    if img.mode != ("RGB" if n == 3 else "L"):
        return None                          # CMYK/YCCK or mismatched JPEGs go through MuPDF
    scale = _scale_for(img.width, shown_width_pt, threshold, target)
    w, h = img.size
    if scale < 0.98:
        img.draft(img.mode, (int(w * scale) + 1, int(h * scale) + 1))
    img.load()
    img = _finish(img, scale * w / img.width)
    if name == "ICCBased" and n == 3 and icc:
        from PIL import ImageCms
        img = ImageCms.profileToProfile(
            img, ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.createProfile("sRGB"), outputMode="RGB")
    return img


def _decode_with_mupdf(doc, xref, shown_width_pt, threshold, target):
    """Decode any image MuPDF understands and return an RGB or gray PIL image at the target
    size, or None if it has transparency. Non-Device colour spaces are converted by MuPDF."""
    pix = fitz.Pixmap(doc, xref)            # Flate/JPX/CMYK/ICC/Decode arrays, decoded by MuPDF
    if pix.alpha or pix.colorspace is None:
        return None
    scale = _scale_for(pix.width, shown_width_pt, threshold, target)
    if scale < 0.5:                          # cheap power-of-two shrink first, keeps memory low
        n = int(math.floor(math.log2(1 / scale)))
        pix.shrink(n)
        scale *= 2 ** n
    # Convert by colour-space *name*: Lab is 3 components and Separation is 1, but neither is RGB/gray.
    name = pix.colorspace.name
    if name not in ("DeviceRGB", "DeviceGray"):
        gray = pix.n == 1 and (name.startswith("ICCBased(Gray") or "Gray" in name)
        pix = fitz.Pixmap(fitz.csGRAY if gray else fitz.csRGB, pix)
    mode = "L" if pix.n == 1 else "RGB"
    # frombuffer shares the pixmap's memory instead of copying it; release that view
    # before the pixmap goes away (otherwise PyMuPDF can't free its buffer).
    view = Image.frombuffer(mode, (pix.width, pix.height), pix.samples_mv, "raw", mode, pix.stride, 1)
    img = _finish(view, scale)
    if img is view:
        img = view.copy()
    del view
    del pix
    return img


def _recompress_image(doc, xref, shown_width_pt, threshold, target, quality):
    """Replace one image stream with a smaller JPEG and a matching image dictionary.
    Skips masks/transparency, 1-bit images, small or unplaced images, and anything that
    wouldn't get smaller. Raises on unexpected errors; the caller leaves the image as is."""
    obj = doc.xref_object(xref, compressed=True)
    # Leave alone anything we can't faithfully re-encode: masks/transparency, stencils, 1-bit scans.
    if any(k in obj for k in ("/SMask", "/Mask", "/ImageMask")):
        return
    if "/BitsPerComponent 1" in obj.replace("\n", " "):
        return
    raw_len = len(doc.xref_stream_raw(xref) or b"")
    if raw_len < COMPRESS_MIN_IMAGE_BYTES:
        return
    if not shown_width_pt:                   # image not placed on any page
        return
    img = _decode_jpeg_scaled(doc, xref, shown_width_pt, threshold, target)
    if img is None:
        img = _decode_with_mupdf(doc, xref, shown_width_pt, threshold, target)
    if img is None:
        return
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    if len(data) >= raw_len:                 # never make an image bigger
        return
    doc.update_stream(xref, data, compress=False)
    doc.xref_set_key(xref, "Filter", "/DCTDecode")
    doc.xref_set_key(xref, "DecodeParms", "null")
    doc.xref_set_key(xref, "Decode", "null")      # the pixmap already has any Decode array applied (e.g. inverted CMYK)
    doc.xref_set_key(xref, "ColorSpace", "/DeviceGray" if img.mode == "L" else "/DeviceRGB")
    doc.xref_set_key(xref, "BitsPerComponent", "8")
    doc.xref_set_key(xref, "Width", str(img.width))
    doc.xref_set_key(xref, "Height", str(img.height))



def _html_escape(text: str) -> str:
    import html
    return html.escape(text or "", quote=False)


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
