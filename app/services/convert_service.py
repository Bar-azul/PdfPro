"""
ConvertService — with memory optimizations for Render 512MB.
"""

import gc
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import fitz
from PIL import Image

from ..config import settings
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


class ConvertService:

    @staticmethod
    def pdf_to_word(pdf_path: Path) -> Path:
        from pdf2docx import Converter
        t0 = time.time()
        pdf_path = Path(pdf_path).resolve()
        out = _temp_file("converted", ".docx")
        cv = Converter(str(pdf_path))
        try:
            cv.convert(str(out), start=0, end=None)
        finally:
            cv.close()
            gc.collect()
        logger.info(f"PDF→Word in {_ms(t0)}ms")
        return out

    @staticmethod
    def pdf_to_excel(pdf_path: Path) -> Path:
        import pdfplumber
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

        t0 = time.time()
        pdf_path = Path(pdf_path).resolve()
        out = _temp_file("converted", ".xlsx")
        wb = openpyxl.Workbook()
        wb.remove(wb.active)

        header_font = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
        header_fill = PatternFill(fill_type="solid", fgColor="0D1B2A")
        border = Border(
            left=Side(style="thin"), right=Side(style="thin"),
            top=Side(style="thin"),  bottom=Side(style="thin"),
        )
        found_any_table = False

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                if tables:
                    found_any_table = True
                    for tbl_num, table in enumerate(tables, start=1):
                        ws_name = f"עמ' {page_num}" + (f" טבלה {tbl_num}" if tbl_num > 1 else "")
                        ws = wb.create_sheet(title=ws_name[:31])
                        for row_idx, row in enumerate(table, start=1):
                            for col_idx, cell in enumerate(row, start=1):
                                c = ws.cell(row=row_idx, column=col_idx, value=cell or "")
                                c.border = border
                                c.alignment = Alignment(wrap_text=True, horizontal="right")
                                if row_idx == 1:
                                    c.font = header_font
                                    c.fill = header_fill
                        for col in ws.columns:
                            max_len = max((len(str(c.value)) if c.value else 0 for c in col), default=10)
                            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)
                else:
                    text = page.extract_text() or ""
                    ws = wb.create_sheet(title=f"עמ' {page_num} טקסט"[:31])
                    for line_num, line in enumerate(text.splitlines(), start=1):
                        ws.cell(row=line_num, column=1, value=line)
                gc.collect()  # ← נקה בין עמודים

        if not wb.worksheets:
            ws = wb.create_sheet(title="ריק")
            ws.cell(row=1, column=1, value="לא נמצאו טבלאות")

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
            for img_path in image_paths:
                with Image.open(img_path) as pil_img:
                    w_px, h_px = pil_img.size
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
                page.insert_image(img_rect, filename=str(img_path))
                gc.collect()

            out = _temp_pdf("from_images")
            doc.save(out, deflate=True)
        finally:
            doc.close()
            gc.collect()

        logger.info(f"Images→PDF in {_ms(t0)}ms")
        return out