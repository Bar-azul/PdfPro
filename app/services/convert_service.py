"""
ConvertService
==============
Handles all file format conversions:
  PDF → Word, Excel, PowerPoint, JPG/PNG
  Word / Excel / PowerPoint / Images → PDF
"""

import io
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from ..config import settings
from ..services.pdf_service import PDFService, _temp_pdf, _temp_file, _ms

logger = logging.getLogger(__name__)


class ConvertService:

    # ────────────────────────────────────────────────────────────────────────
    # PDF → Other formats
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def pdf_to_word(pdf_path: Path) -> Path:
        """Convert PDF to .docx using pdf2docx."""
        from pdf2docx import Converter
        t0 = time.time()
        out = _temp_file("converted", ".docx")
        cv = Converter(str(pdf_path))
        cv.convert(str(out), start=0, end=None)
        cv.close()
        logger.info(f"PDF→Word in {_ms(t0)}ms → {out}")
        return out

    @staticmethod
    def pdf_to_excel(pdf_path: Path) -> Path:
        """
        Convert PDF tables to .xlsx using pdfplumber for table extraction.
        Falls back to text extraction if no tables found.
        """
        import pdfplumber
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

        t0 = time.time()
        out = _temp_file("converted", ".xlsx")
        wb = openpyxl.Workbook()
        wb.remove(wb.active)  # Remove default sheet

        header_font = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
        header_fill = PatternFill(fill_type="solid", fgColor="0D1B2A")
        border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        with pdfplumber.open(str(pdf_path)) as pdf:
            found_any_table = False

            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()

                if tables:
                    found_any_table = True
                    for tbl_num, table in enumerate(tables, start=1):
                        ws_name = f"עמ' {page_num}"
                        if tbl_num > 1:
                            ws_name += f" טבלה {tbl_num}"
                        ws = wb.create_sheet(title=ws_name[:31])

                        for row_idx, row in enumerate(table, start=1):
                            for col_idx, cell in enumerate(row, start=1):
                                c = ws.cell(row=row_idx, column=col_idx, value=cell or "")
                                c.border = border
                                c.alignment = Alignment(wrap_text=True, horizontal="right")
                                if row_idx == 1:
                                    c.font = header_font
                                    c.fill = header_fill

                        # Auto column width
                        for col in ws.columns:
                            max_len = max(
                                (len(str(c.value)) if c.value else 0 for c in col), default=10
                            )
                            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)

                else:
                    # No tables — extract raw text into a sheet
                    text = page.extract_text() or ""
                    ws = wb.create_sheet(title=f"עמ' {page_num} טקסט"[:31])
                    for line_num, line in enumerate(text.splitlines(), start=1):
                        ws.cell(row=line_num, column=1, value=line)

            if not wb.worksheets:
                ws = wb.create_sheet(title="ריק")
                ws.cell(row=1, column=1, value="לא נמצאו טבלאות במסמך")

        wb.save(str(out))
        logger.info(f"PDF→Excel in {_ms(t0)}ms (found tables: {found_any_table})")
        return out

    @staticmethod
    def pdf_to_pptx(pdf_path: Path, dpi: int = 150) -> Path:
        """
        Convert PDF to .pptx: render each page as an image, insert into slides.
        Preserves visual appearance exactly.
        """
        from pptx import Presentation
        from pptx.util import Pt, Emu

        t0 = time.time()
        image_paths = PDFService.render_pages(pdf_path, dpi=dpi, fmt="png")

        prs = Presentation()

        with fitz.open(pdf_path) as doc:
            first_page = doc[0].rect
            # Set slide size to match PDF page aspect ratio (widescreen default)
            width_emu = int(first_page.width / 72 * 914400)
            height_emu = int(first_page.height / 72 * 914400)
            prs.slide_width = width_emu
            prs.slide_height = height_emu

        blank_layout = prs.slide_layouts[6]

        for img_path in image_paths:
            slide = prs.slides.add_slide(blank_layout)
            slide.shapes.add_picture(
                str(img_path),
                left=0, top=0,
                width=prs.slide_width,
                height=prs.slide_height,
            )
            img_path.unlink(missing_ok=True)

        out = _temp_file("converted", ".pptx")
        prs.save(str(out))
        logger.info(f"PDF→PPTX ({len(image_paths)} slides) in {_ms(t0)}ms")
        return out

    @staticmethod
    def pdf_to_images(
        pdf_path: Path,
        dpi: int = 150,
        fmt: str = "jpg",
        quality: int = 85,
        pages: list[int] | None = None,
    ) -> list[Path]:
        """Convert PDF pages to image files. Returns list of paths."""
        return PDFService.render_pages(pdf_path, dpi=dpi, fmt=fmt, quality=quality, pages=pages)

    @staticmethod
    def pdf_to_text(pdf_path: Path) -> str:
        """Extract all text from PDF."""
        with fitz.open(pdf_path) as doc:
            parts = [page.get_text() for page in doc]
        return "\n\n--- עמוד חדש ---\n\n".join(parts)

    # ────────────────────────────────────────────────────────────────────────
    # Other formats → PDF
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def office_to_pdf(input_path: Path) -> Path:
        """
        Convert Word / Excel / PowerPoint to PDF using LibreOffice.
        LibreOffice must be installed: `apt install libreoffice`
        """
        t0 = time.time()
        out_dir = Path(tempfile.mkdtemp())

        cmd = [
            settings.LIBREOFFICE_PATH,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(out_dir),
            str(input_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            logger.error(f"LibreOffice error: {result.stderr}")
            raise RuntimeError(f"LibreOffice conversion failed: {result.stderr[:300]}")

        # Find the generated PDF
        pdf_files = list(out_dir.glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError("LibreOffice did not produce a PDF output")

        out = _temp_pdf("from_office")
        shutil.move(str(pdf_files[0]), str(out))
        shutil.rmtree(out_dir, ignore_errors=True)

        logger.info(f"Office→PDF in {_ms(t0)}ms")
        return out

    @staticmethod
    def images_to_pdf(
        image_paths: list[Path],
        page_size: str = "fit",
    ) -> Path:
        """Combine multiple images into a single PDF."""
        t0 = time.time()
        doc = fitz.open()

        for img_path in image_paths:
            # Open image to get dimensions
            with Image.open(img_path) as pil_img:
                w_px, h_px = pil_img.size

            # Convert pixels (assuming 96 DPI) to points
            w_pt = w_px * 72 / 96
            h_pt = h_px * 72 / 96

            if page_size == "fit":
                rect = fitz.Rect(0, 0, w_pt, h_pt)
            else:
                # Standard sizes in points
                sizes = {
                    "A4": (595, 842), "Letter": (612, 792),
                    "Legal": (612, 1008), "A3": (842, 1190),
                }
                pw, ph = sizes.get(page_size, (595, 842))
                rect = fitz.Rect(0, 0, pw, ph)

            page = doc.new_page(width=rect.width, height=rect.height)

            # Scale image to fill page
            scale = min(rect.width / w_pt, rect.height / h_pt)
            img_rect = fitz.Rect(
                (rect.width - w_pt * scale) / 2,
                (rect.height - h_pt * scale) / 2,
                (rect.width + w_pt * scale) / 2,
                (rect.height + h_pt * scale) / 2,
            )
            page.insert_image(img_rect, filename=str(img_path))

        out = _temp_pdf("from_images")
        doc.save(out, deflate=True)
        doc.close()
        logger.info(f"Images→PDF ({len(image_paths)} pages) in {_ms(t0)}ms")
        return out

    @staticmethod
    def html_to_pdf(html_content: str) -> Path:
        """Convert HTML string to PDF using PyMuPDF's story API."""
        t0 = time.time()
        story = fitz.Story(html=html_content)
        writer = fitz.DocumentWriter(_temp_file("from_html", ".pdf").as_posix())
        mediabox = fitz.paper_rect("A4")
        where = mediabox + (36, 36, -36, -36)  # 0.5" margins
        more = True
        while more:
            device = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(device)
            writer.end_page()
        writer.close()
        out = _temp_pdf("from_html")
        logger.info(f"HTML→PDF in {_ms(t0)}ms")
        return out
