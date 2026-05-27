"""
OCRService
==========
Extract text from scanned PDFs or images using Tesseract OCR.
Supports Hebrew, English, Arabic, and 40+ other languages.
"""

import io
import logging
import time
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from ..config import settings
from ..services.pdf_service import _temp_pdf, _temp_file, _ms

logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH


def _image_to_pdf(image_path: Path) -> Path:
    """Convert any image file to a single-page PDF using Pillow."""
    with Image.open(image_path) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        if img.mode == "RGBA":
            img = img.convert("RGB")
        out = _temp_pdf("img_as_pdf")
        img.save(str(out), format="PDF", resolution=150)
    return out


def _ensure_pdf(path: Path) -> tuple[Path, bool]:
    """
    Return (pdf_path, is_temp).
    If path is already a PDF, return it unchanged.
    If it's an image, convert it and return the temp PDF path.
    """
    try:
        doc = fitz.open(path)
        is_pdf = doc.is_pdf
        doc.close()
        if is_pdf:
            return path, False
    except Exception:
        pass
    # It's an image — convert
    logger.info(f"Converting image to PDF for OCR: {path.name}")
    return _image_to_pdf(path), True


class OCRService:

    @staticmethod
    def extract_text(
        pdf_path: Path,
        language: str = "heb+eng",
        dpi: int = 300,
        pages: list[int] | None = None,
    ) -> list[dict]:
        t0 = time.time()
        results = []
        matrix = fitz.Matrix(dpi / 72, dpi / 72)

        actual_path, is_temp = _ensure_pdf(pdf_path)

        try:
            with fitz.open(actual_path) as doc:
                target = [p - 1 for p in pages] if pages else range(doc.page_count)
                for i in target:
                    if not (0 <= i < doc.page_count):
                        continue
                    page = doc[i]

                    native_text = page.get_text().strip()
                    if native_text and len(native_text) > 50:
                        results.append({
                            "page": i + 1,
                            "text": native_text,
                            "confidence": 1.0,
                            "source": "native",
                        })
                        continue

                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                    data = pytesseract.image_to_data(
                        img,
                        lang=language,
                        config="--oem 3 --psm 3",
                        output_type=pytesseract.Output.DICT,
                    )

                    words = [
                        w for w, c in zip(data["text"], data["conf"])
                        if w.strip() and int(c) > 20
                    ]
                    page_text = " ".join(words)

                    valid_confs = [int(c) for c in data["conf"] if int(c) > 0]
                    avg_conf = (sum(valid_confs) / len(valid_confs) / 100) if valid_confs else 0.0

                    results.append({
                        "page": i + 1,
                        "text": page_text,
                        "confidence": round(avg_conf, 3),
                        "source": "ocr",
                    })
        finally:
            if is_temp and actual_path.exists():
                actual_path.unlink(missing_ok=True)

        logger.info(f"OCR: {len(results)} pages, lang={language} in {_ms(t0)}ms")
        return results

    @staticmethod
    def extract_to_txt(pdf_path: Path, language: str = "heb+eng", dpi: int = 300) -> Path:
        results = OCRService.extract_text(pdf_path, language=language, dpi=dpi)
        out = _temp_file("ocr_output", ".txt")
        lines = []
        for r in results:
            lines.append(f"=== עמוד {r['page']} ===")
            lines.append(r["text"])
            lines.append("")
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    @staticmethod
    def extract_to_searchable_pdf(
        pdf_path: Path, language: str = "heb+eng", dpi: int = 300
    ) -> Path:
        """Create a searchable PDF with invisible OCR text layer."""
        t0 = time.time()
        actual_path, is_temp = _ensure_pdf(pdf_path)

        try:
            results = OCRService.extract_text(actual_path, language=language, dpi=dpi)
            text_by_page = {r["page"]: r["text"] for r in results}

            with fitz.open(actual_path) as doc:
                for page_num, text in text_by_page.items():
                    if not text.strip():
                        continue
                    page = doc[page_num - 1]
                    try:
                        page.insert_text(
                            fitz.Point(10, 20),
                            text,
                            fontsize=1,
                            color=(1, 1, 1),
                            overlay=False,
                        )
                    except Exception as e:
                        logger.warning(f"Could not insert text on page {page_num}: {e}")

                out = _temp_pdf("searchable")
                doc.save(out, deflate=True)
        finally:
            if is_temp and actual_path.exists():
                actual_path.unlink(missing_ok=True)

        logger.info(f"Searchable PDF created in {_ms(t0)}ms")
        return out

    @staticmethod
    def extract_to_docx(pdf_path: Path, language: str = "heb+eng", dpi: int = 300) -> Path:
        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        results = OCRService.extract_text(pdf_path, language=language, dpi=dpi)

        doc = Document()
        title = doc.add_heading("מסמך מחולץ — OCR", level=1)
        title.alignment = WD_ALIGN_PARAGRAPH.RIGHT

        for r in results:
            doc.add_heading(f"עמוד {r['page']}", level=2)
            para = doc.add_paragraph(r["text"])
            para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            for run in para.runs:
                run.font.name = "David"
                run.font.size = Pt(12)
            doc.add_paragraph()

        out = _temp_file("ocr_output", ".docx")
        doc.save(str(out))
        return out

    @staticmethod
    def ocr_image(image_path: Path, language: str = "heb+eng") -> dict:
        t0 = time.time()
        with Image.open(image_path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            data = pytesseract.image_to_data(
                img,
                lang=language,
                config="--oem 3 --psm 3",
                output_type=pytesseract.Output.DICT,
            )

        words = [
            w for w, c in zip(data["text"], data["conf"])
            if w.strip() and int(c) > 20
        ]
        text = " ".join(words)
        valid_confs = [int(c) for c in data["conf"] if int(c) > 0]
        avg_conf = (sum(valid_confs) / len(valid_confs) / 100) if valid_confs else 0.0

        logger.info(f"Image OCR in {_ms(t0)}ms, conf={avg_conf:.2f}")
        return {"text": text, "confidence": round(avg_conf, 3)}

    @staticmethod
    def get_available_languages() -> list[str]:
        try:
            langs = pytesseract.get_languages(config="")
            return [l for l in langs if l != "osd"]
        except Exception:
            return settings.OCR_LANGUAGES