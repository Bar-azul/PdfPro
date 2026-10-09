"""
ConvertService — with memory optimizations for Render 512MB.
"""

import gc
import logging
import os
import shutil
import subprocess
import tempfile
import re
import time
import uuid
from pathlib import Path

import fitz
from PIL import Image

from ..config import settings
from . import progress
from ..services.pdf_service import PDFService, _temp_pdf, _temp_file, _ms

logger = logging.getLogger(__name__)

OFFICE_EXTENSIONS = {
    ".doc", ".docx", ".odt", ".rtf",
    ".xls", ".xlsx", ".ods",
    ".ppt", ".pptx", ".odp",
}


def _resolve_libreoffice_path() -> str:
    configured = getattr(settings, "LIBREOFFICE_PATH", None)
    if configured:
        configured = str(configured).strip()
        if configured and Path(configured).exists():
            return configured
        found = shutil.which(configured)
        if found:
            return found

    windows_path = r"C:\Program Files\LibreOffice\program\soffice.exe"
    if Path(windows_path).exists():
        return windows_path

    for cmd in ("soffice", "libreoffice"):
        found = shutil.which(cmd)
        if found:
            return found

    raise RuntimeError(
        "LibreOffice not found. Install LibreOffice or set LIBREOFFICE_PATH."
    )


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


_RTL_RE = re.compile(r"[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]")
_NUMBER_RE = re.compile(r"^(-)?\s*([\d,]*\d(?:\.\d+)?)\s*(-)?$")


def _has_rtl(text: str) -> bool:
    return bool(_RTL_RE.search(text or ""))


def _cell_text(page, bbox) -> str:
    """Text inside one table cell, in reading order (PyMuPDF orders RTL text correctly)."""
    if not bbox:
        return ""
    text = page.get_text("text", clip=fitz.Rect(bbox)).strip()
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def _excel_value(raw: str):
    """'12,450.00' → 12450.0 with a thousands format; '412.90-' (RTL minus) → -412.9."""
    m = _NUMBER_RE.match(raw.replace("\u200f", "").replace("\u200e", "").strip()) if raw else None
    if not m or (m.group(1) and m.group(3)):
        return raw, None
    digits = m.group(2)
    if "," in digits and not re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", digits):
        return raw, None  # commas that aren't thousands separators — leave as text
    number = float(digits.replace(",", ""))
    if m.group(1) or m.group(3):
        number = -number
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    if decimals == 0 and "," not in digits and len(digits) > 1 and digits.startswith("0"):
        return raw, None  # leading zeros (IDs, account numbers) stay text
    if len(digits.replace(",", "").replace(".", "")) > 15:
        return raw, None  # too long for Excel precision (card/account numbers)
    fmt = "#,##0" + ("." + "0" * decimals if decimals else "") if "," in digits or decimals else None
    return (int(number) if decimals == 0 else number), fmt


# EXIF orientation → counter-clockwise rotation for insert_image (no re-encoding needed)
_EXIF_ROTATE = {3: 180, 6: 270, 8: 90}


def _image_for_pdf(img_path: Path):
    """
    Returns (stream or None, upright width px, upright height px, rotate).
    Phone photos store "turn me" in EXIF instead of rotating the pixels; honour it so
    portrait photos don't come out lying on their side. Plain rotations keep the
    original bytes; mirrored orientations and formats PDF can't hold are re-encoded.
    """
    import io
    from PIL import ImageOps
    with Image.open(img_path) as im:
        try:
            orientation = im.getexif().get(0x0112, 1)
        except Exception:
            orientation = 1
        w, h = im.size
        if orientation in (1, None) or orientation not in range(1, 9):
            needs_reencode = im.format not in ("JPEG", "PNG")
            if not needs_reencode:
                return None, w, h, 0
        elif orientation in _EXIF_ROTATE and im.format in ("JPEG", "PNG"):
            rot = _EXIF_ROTATE[orientation]
            return (None, h, w, rot) if rot in (90, 270) else (None, w, h, rot)
        upright = ImageOps.exif_transpose(im)
        if upright.mode not in ("RGB", "L", "RGBA", "LA"):
            upright = upright.convert("RGBA" if "A" in upright.getbands() else "RGB")
        buf = io.BytesIO()
        upright.save(buf, format="PNG")
        return buf.getvalue(), upright.width, upright.height, 0


class _Pdf2DocxProgress(logging.Handler):
    """
    pdf2docx logs "(i/n) Page p" while it parses pages ([3/4]) and again while it
    writes them ([4/4]); turn those lines into real progress: parsing is the first
    half of the bar, writing the second.
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.phase = 0

    def emit(self, record):
        if "pdf2docx" not in record.pathname:
            return
        msg = str(record.msg)
        if "[3/4]" in msg:
            self.phase = 0
        elif "[4/4]" in msg:
            self.phase = 1
        elif msg.startswith("(%d/%d)") and len(record.args or ()) >= 2:
            i, n = record.args[0], record.args[1]
            progress.update(self.phase * n + i, 2 * n, "pages")


class ConvertService:

    @staticmethod
    def pdf_to_word(pdf_path: Path) -> Path:
        from pdf2docx import Converter
        t0 = time.time()
        pdf_path = Path(pdf_path).resolve()
        out = _temp_file("converted", ".docx")
        cv = Converter(str(pdf_path))
        handler = _Pdf2DocxProgress()
        root = logging.getLogger()
        root.addHandler(handler)
        old_level = root.level
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
        try:
            cv.convert(str(out), start=0, end=None)
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)
            cv.close()
            gc.collect()
        logger.info(f"PDF→Word in {_ms(t0)}ms")
        return out

    @staticmethod
    def pdf_to_excel(pdf_path: Path) -> Path:
        """
        Tables → one sheet per table (cells keep their row/column, numbers become real
        numbers). Pages without a ruled table go to a sheet with their text lines.
        Text is read with PyMuPDF so Hebrew/Arabic comes out in reading order.
        """
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from ..utils.errors import ApiError

        t0 = time.time()
        pdf_path = Path(pdf_path).resolve()
        out = _temp_file("converted", ".xlsx")
        wb = openpyxl.Workbook()
        wb.remove(wb.active)

        header_font = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
        header_fill = PatternFill(fill_type="solid", fgColor="0D1B2A")
        side = Side(style="thin")
        border = Border(left=side, right=side, top=side, bottom=side)
        any_text = False

        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                progress.update(page_num - 1, doc.page_count, "pages")
                page_text = page.get_text().strip()
                any_text = any_text or bool(page_text)
                try:
                    tables = page.find_tables().tables
                except Exception as e:  # table detection is best-effort
                    logger.warning(f"find_tables failed on page {page_num}: {e}")
                    tables = []
                tables = [t for t in tables if t.row_count >= 2 and t.col_count >= 2]

                for tbl_num, table in enumerate(tables, start=1):
                    name = f"Page {page_num}" + (f" table {tbl_num}" if len(tables) > 1 else "")
                    ws = wb.create_sheet(title=name[:31])
                    widths: dict[int, int] = {}
                    for row_idx, row in enumerate(table.rows, start=1):
                        for col_idx, bbox in enumerate(row.cells, start=1):
                            raw = _cell_text(page, bbox)
                            value, number_format = _excel_value(raw)
                            c = ws.cell(row=row_idx, column=col_idx, value=value)
                            c.border = border
                            if number_format:
                                c.number_format = number_format
                            c.alignment = Alignment(
                                wrap_text=True, vertical="top",
                                horizontal="right" if _has_rtl(raw) else None,
                            )
                            if row_idx == 1:
                                c.font = header_font
                                c.fill = header_fill
                            widths[col_idx] = max(widths.get(col_idx, 8), len(raw))
                    for col_idx, w in widths.items():
                        ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = min(w + 4, 60)

                if not tables and page_text:
                    ws = wb.create_sheet(title=f"Page {page_num} text"[:31])
                    ws.column_dimensions["A"].width = 100
                    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
                    for line_num, line in enumerate(lines, start=1):
                        c = ws.cell(row=line_num, column=1, value=line)
                        if _has_rtl(line):
                            c.alignment = Alignment(horizontal="right")
                gc.collect()

        if not any_text:
            _safe_unlink(out)
            raise ApiError(400, "no_text",
                           "This PDF has no selectable text (it looks like a scan). Run OCR first, then convert.")

        wb.save(str(out))
        gc.collect()
        logger.info(f"PDF→Excel in {_ms(t0)}ms")
        return out

    @staticmethod
    def pdf_to_pptx(pdf_path: Path, dpi: int = 120) -> Path:  # ← הורדנו מ-150 ל-120
        from pptx import Presentation
        t0 = time.time()
        pdf_path = Path(pdf_path).resolve()
        image_paths = PDFService.render_pages(pdf_path, dpi=dpi, fmt="png")

        prs = Presentation()
        with fitz.open(pdf_path) as doc:
            first_page = doc[0].rect
            prs.slide_width  = int(first_page.width  / 72 * 914400)
            prs.slide_height = int(first_page.height / 72 * 914400)

        blank_layout = prs.slide_layouts[6]
        progress.stage("saving")
        try:
            for img_path in image_paths:
                slide = prs.slides.add_slide(blank_layout)
                slide.shapes.add_picture(
                    str(img_path), left=0, top=0,
                    width=prs.slide_width, height=prs.slide_height,
                )
                gc.collect()
        finally:
            for img_path in image_paths:
                _safe_unlink(Path(img_path))

        out = _temp_file("converted", ".pptx")
        prs.save(str(out))
        gc.collect()
        logger.info(f"PDF→PPTX ({len(image_paths)} slides) in {_ms(t0)}ms")
        return out

    @staticmethod
    def pdf_to_images(
        pdf_path: Path,
        dpi: int = 120,   # ← הורדנו מ-150
        fmt: str = "jpg",
        quality: int = 80,  # ← הורדנו מ-85
        pages: list[int] | None = None,
    ) -> list[Path]:
        return PDFService.render_pages(pdf_path, dpi=dpi, fmt=fmt, quality=quality, pages=pages)

    @staticmethod
    def pdf_to_text(pdf_path: Path) -> str:
        with fitz.open(pdf_path) as doc:
            parts = [page.get_text() for page in doc]
        return "\n\n--- עמוד חדש ---\n\n".join(parts)

    @staticmethod
    def office_to_pdf(input_path: Path) -> Path:
        """Convert Word/Excel/PPT to PDF using LibreOffice."""
        t0 = time.time()
        input_path = Path(input_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        suffix = input_path.suffix.lower().strip()
        if suffix not in OFFICE_EXTENSIONS:
            raise RuntimeError(f"Unsupported extension: {suffix}")

        libreoffice_path = _resolve_libreoffice_path()
        work_dir: Path | None = None

        try:
            work_dir = Path(tempfile.mkdtemp(prefix="office_pdf_")).resolve()
            input_dir  = work_dir / "input"
            output_dir = work_dir / "output"
            profile_dir = work_dir / "profile"

            for d in (input_dir, output_dir, profile_dir):
                d.mkdir(parents=True, exist_ok=True)

            safe_input = input_dir / f"input_{uuid.uuid4().hex}{suffix}"
            shutil.copy2(str(input_path), str(safe_input))

            cmd = [
                libreoffice_path,
                "--headless", "--nologo", "--nofirststartwizard",
                "--nolockcheck", "--nodefault", "--norestore",
                f"-env:UserInstallation={profile_dir.as_uri()}",
                "--convert-to", "pdf",
                "--outdir", str(output_dir),
                str(safe_input),
            ]

            env = os.environ.copy()
            env["HOME"] = str(work_dir)

            progress.stage("converting")
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=120, cwd=str(work_dir), env=env,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"LibreOffice failed (code {result.returncode})\n"
                    f"STDERR: {result.stderr[:500]}"
                )

            pdf_files = list(output_dir.glob("*.pdf"))
            if not pdf_files:
                raise RuntimeError("LibreOffice produced no PDF output")

            out = _temp_pdf("from_office")
            shutil.copy2(str(pdf_files[0]), str(out))

            logger.info(f"Office→PDF in {_ms(t0)}ms")
            return out

        except subprocess.TimeoutExpired:
            raise RuntimeError("LibreOffice timed out after 120 seconds")

        finally:
            if work_dir and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            gc.collect()

    @staticmethod
    def images_to_pdf(image_paths: list[Path], page_size: str = "fit") -> Path:
        t0 = time.time()
        doc = fitz.open()
        sizes = {"A4":(595,842),"Letter":(612,792),"Legal":(612,1008),"A3":(842,1190)}

        try:
            for k, img_path in enumerate(image_paths, 1):
                progress.update(k - 1, len(image_paths), "images")
                stream, w_px, h_px, rotate = _image_for_pdf(img_path)
                w_pt = w_px * 72 / 96
                h_pt = h_px * 72 / 96

                if page_size == "fit":
                    rect = fitz.Rect(0, 0, w_pt, h_pt)
                else:
                    pw, ph = sizes.get(page_size, (595, 842))
                    rect = fitz.Rect(0, 0, pw, ph)

                page = doc.new_page(width=rect.width, height=rect.height)
                scale = min(rect.width / w_pt, rect.height / h_pt)
                img_rect = fitz.Rect(
                    (rect.width  - w_pt * scale) / 2,
                    (rect.height - h_pt * scale) / 2,
                    (rect.width  + w_pt * scale) / 2,
                    (rect.height + h_pt * scale) / 2,
                )
                if stream is None:
                    page.insert_image(img_rect, filename=str(img_path), rotate=rotate)
                else:
                    page.insert_image(img_rect, stream=stream, rotate=rotate)
                gc.collect()

            out = _temp_pdf("from_images")
            doc.save(out, deflate=True)
        finally:
            doc.close()
            gc.collect()

        logger.info(f"Images→PDF in {_ms(t0)}ms")
        return out