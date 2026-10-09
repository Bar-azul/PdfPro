"""
OCRService — memory optimized + auto-rotation + image enhancement.
"""

import gc
import logging
import re
import time
from pathlib import Path

import fitz
import pytesseract
from PIL import Image, ImageEnhance

from ..config import settings
from ..services.pdf_service import _temp_pdf, _temp_file, _ms

logger = logging.getLogger(__name__)
pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".gif"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _image_to_pdf(image_path: Path) -> Path:
    from PIL import ImageOps
    with Image.open(image_path) as raw:
        img = ImageOps.exif_transpose(raw)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        out = _temp_pdf("img_as_pdf")
        img.save(str(out), format="PDF", resolution=150)
    return out


def _prepare_image(img: Image.Image, with_angle: bool = False):
    """
    Prepare image for best OCR accuracy:
    1. Resize if too large
    2. Auto-detect and fix rotation
    3. Convert to grayscale
    4. Enhance contrast and sharpness
    """
    # Step 1 — Resize if too large
    max_px = 3000
    if max(img.width, img.height) > max_px:
        ratio = max_px / max(img.width, img.height)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)),
            Image.LANCZOS,
        )

    # Step 2 — Convert to RGB for OSD
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Step 3 — Auto-detect and fix rotation
    angle = 0
    try:
        osd = pytesseract.image_to_osd(
            img,
            output_type=pytesseract.Output.DICT,
            config="--psm 0",
        )
        angle = osd.get("rotate", 0)
        if angle and angle != 0:
            img = img.rotate(-angle, expand=True)
            logger.info(f"Auto-rotated image by {angle}°")
    except Exception as e:
        logger.debug(f"OSD failed (continuing without rotation): {e}")

    # Step 4 — Grayscale + enhance
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    return (img, angle or 0) if with_angle else img


_RTL_CHARS = re.compile(r"[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]")
_FONT_FILES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_layer_font = None


def _get_layer_font() -> fitz.Font:
    """A font with Hebrew/Arabic glyphs if one is installed (the text is invisible, so
    the look doesn't matter, but real glyphs give viewers correct word widths)."""
    global _layer_font
    if _layer_font is None:
        for path in _FONT_FILES:
            if Path(path).exists():
                _layer_font = fitz.Font(fontfile=path)
                break
        else:
            _layer_font = fitz.Font("helv")
    return _layer_font


def _text_layer(img_w: int, img_h: int, data: dict, w_pt: float, h_pt: float):
    """
    Build a one-page PDF (w_pt x h_pt, the OCR image's frame) holding the OCR
    result as invisible text, one string per detected line. Text is written in
    logical order with right_to_left set for Hebrew/Arabic lines, so copy and
    search return the words the right way round.
    """
    font = _get_layer_font()
    sx, sy = w_pt / img_w, h_pt / img_h
    lines: dict[tuple, list[int]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip() or float(data["conf"][i]) < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)
    if not lines:
        return None

    layer = fitz.open()
    page = layer.new_page(width=w_pt, height=h_pt)
    tw = fitz.TextWriter(page.rect)
    for idx in lines.values():
        words = [data["text"][i].strip() for i in idx]
        left = min(data["left"][i] for i in idx)
        right = max(data["left"][i] + data["width"][i] for i in idx)
        top = min(data["top"][i] for i in idx)
        bottom = max(data["top"][i] + data["height"][i] for i in idx)
        rtl = sum(bool(_RTL_CHARS.search(w)) for w in words) * 2 >= len(words)
        text = " ".join(words)
        box_w, box_h = (right - left) * sx, (bottom - top) * sy
        length = font.text_length(text, 1)
        size = min(box_h, box_w / length) if length else box_h
        if size <= 0:
            continue
        try:
            tw.append((left * sx, bottom * sy - box_h * 0.15), text,
                      font=font, fontsize=size, right_to_left=rtl)
        except Exception as e:  # a glyph the font can't encode — skip that line only
            logger.debug(f"text layer line skipped: {e}")
    tw.write_text(page, render_mode=3)  # 3 = invisible
    return layer


class OCRService:

    @staticmethod
    def extract_text(
        pdf_path: Path,
        language: str = "heb+eng",
        dpi: int = 200,
        pages: list[int] | None = None,
    ) -> list[dict]:
        t0 = time.time()

        # ── IMAGE: direct Tesseract — no PDF overhead ─────────────────────────
        if _is_image(pdf_path):
            result = OCRService.ocr_image(pdf_path, language=language)
            logger.info(f"Image OCR (direct) in {_ms(t0)}ms")
            return [{
                "page": 1,
                "text": result["text"],
                "confidence": result["confidence"],
                "source": "ocr",
            }]

        # ── PDF: page by page ─────────────────────────────────────────────────
        results = []
        matrix = fitz.Matrix(dpi / 72, dpi / 72)

        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)
            for i in target:
                if not (0 <= i < doc.page_count):
                    continue
                page = doc[i]

                # Use native text if available
                native_text = page.get_text().strip()
                if native_text and len(native_text) > 50:
                    results.append({
                        "page": i + 1,
                        "text": native_text,
                        "confidence": 1.0,
                        "source": "native",
                    })
                    continue

                # Render → prepare → OCR
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                del pix

                img = _prepare_image(img)

                data = pytesseract.image_to_data(
                    img,
                    lang=language,
                    config="--oem 3 --psm 3",
                    output_type=pytesseract.Output.DICT,
                )
                del img

                words = [
                    w for w, c in zip(data["text"], data["conf"])
                    if w.strip() and int(c) > 20
                ]
                valid_confs = [int(c) for c in data["conf"] if int(c) > 0]
                avg_conf = (sum(valid_confs) / len(valid_confs) / 100) if valid_confs else 0.0

                results.append({
                    "page": i + 1,
                    "text": " ".join(words),
                    "confidence": round(avg_conf, 3),
                    "source": "ocr",
                })
                gc.collect()

        gc.collect()
        logger.info(f"OCR: {len(results)} pages, lang={language} in {_ms(t0)}ms")
        return results

    @staticmethod
    def extract_to_txt(
        pdf_path: Path, language: str = "heb+eng", dpi: int = 200
    ) -> Path:
        results = OCRService.extract_text(pdf_path, language=language, dpi=dpi)
        out = _temp_file("ocr_output", ".txt")
        lines = []
        for r in results:
            lines.append(f"=== עמוד {r['page']} ===")
            lines.append(r["text"])
            lines.append("")
        out.write_text("\n".join(lines), encoding="utf-8")
        gc.collect()
        return out

    @staticmethod
    def extract_to_searchable_pdf(
        pdf_path: Path, language: str = "heb+eng", dpi: int = 200
    ) -> Path:
        """
        Keep every page exactly as it looks and add an invisible text layer on top,
        word lines placed where Tesseract found them — so Ctrl+F, copy and
        screen readers work, in Hebrew/Arabic too.
        """
        t0 = time.time()
        converted = None
        if _is_image(pdf_path):
            converted = pdf_path = _image_to_pdf(pdf_path)

        try:
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    # pages that already have real text are searchable as they are
                    if len(page.get_text().strip()) > 50:
                        continue
                    pix = page.get_pixmap(matrix=matrix, alpha=False)  # as it looks (rotation applied)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    del pix
                    img, osd_angle = _prepare_image(img, with_angle=True)
                    data = pytesseract.image_to_data(
                        img, lang=language, config="--oem 3 --psm 3",
                        output_type=pytesseract.Output.DICT,
                    )
                    vis_w, vis_h = page.rect.width, page.rect.height
                    if osd_angle % 180:
                        vis_w, vis_h = vis_h, vis_w
                    layer = _text_layer(img.width, img.height, data, vis_w, vis_h)
                    del img
                    if layer is None:
                        continue
                    with layer:
                        target = (page.rect * page.derotation_matrix).normalize()
                        page.show_pdf_page(
                            target, layer, 0, overlay=True,
                            rotate=(page.rotation + osd_angle) % 360,
                        )
                    gc.collect()
                out = _temp_pdf("searchable")
                doc.save(out, deflate=True, garbage=3)
        finally:
            if converted is not None:
                converted.unlink(missing_ok=True)
            gc.collect()

        logger.info(f"Searchable PDF in {_ms(t0)}ms")
        return out

    @staticmethod
    def extract_to_docx(
        pdf_path: Path, language: str = "heb+eng", dpi: int = 200
    ) -> Path:
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
        gc.collect()
        return out

    @staticmethod
    def ocr_image(image_path: Path, language: str = "heb+eng") -> dict:
        """
        Run Tesseract directly on an image file.
        Auto-detects rotation and enhances image quality before OCR.
        """
        t0 = time.time()

        from PIL import ImageOps
        with Image.open(image_path) as raw:
            img = _prepare_image(ImageOps.exif_transpose(raw))
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

        gc.collect()
        logger.info(f"Image OCR in {_ms(t0)}ms, conf={avg_conf:.2f}")
        return {"text": text, "confidence": round(avg_conf, 3)}

    @staticmethod
    def get_available_languages() -> list[str]:
        try:
            langs = pytesseract.get_languages(config="")
            return [l for l in langs if l != "osd"]
        except Exception:
            return settings.OCR_LANGUAGES