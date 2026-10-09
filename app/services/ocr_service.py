"""
OCRService — memory optimized + auto-rotation + image enhancement.
"""

import gc
import logging
import re
import time
from pathlib import Path

import os

# Tesseract's OpenMP threads only fight each other on the server's fraction of a CPU
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

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


# Render budget per page. A4 at 300 dpi is ~8.7 MP; phone-scan PDFs often have pages
# sized in pixels-as-points (4000x3000 pt), which at 300 dpi would be 100+ MP and
# blow past the server's 512 MB. Grey 8-bit keeps one page at ~10 MB.
_MAX_RENDER_PIXELS = 10_000_000
_MAX_RENDER_SIDE = 4000


def _render_page_gray(page: "fitz.Page", dpi: int) -> Image.Image:
    """Render a page as it looks (rotation applied) in 8-bit grey, capped in size."""
    w_in, h_in = page.rect.width / 72, page.rect.height / 72
    scale = dpi
    if w_in * h_in * scale * scale > _MAX_RENDER_PIXELS:
        scale = (_MAX_RENDER_PIXELS / (w_in * h_in)) ** 0.5
    scale = min(scale, _MAX_RENDER_SIDE / max(w_in, h_in))
    pix = page.get_pixmap(matrix=fitz.Matrix(scale / 72, scale / 72),
                          colorspace=fitz.csGRAY, alpha=False)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    del pix
    # MuPDF keeps decoded page images in its cache; on scans that is a full page
    # of pixels per page, which piled up to 500+ MB on an 8-page scan.
    fitz.TOOLS.store_shrink(100)
    return img


def _invisible_text_spans(page: "fitz.Page") -> tuple[int, list]:
    """
    (visible character count, bboxes of invisible text spans).
    Scanner apps (Adobe Scan, Microsoft Lens, some phone galleries) add their own
    invisible OCR layer, which is often garbage for Hebrew. Text that is drawn
    (render mode != 3) is real text and is kept.
    """
    visible, invisible = 0, []
    try:
        for span in page.get_texttrace():
            n = len(span.get("chars", ()))
            if span.get("type") == 3 or span.get("opacity", 1) == 0:
                invisible.append(fitz.Rect(span["bbox"]))
            else:
                visible += n
    except Exception:
        visible = len(page.get_text().strip())
    return visible, invisible


def _needs_ocr(page: "fitz.Page") -> bool:
    visible, _ = _invisible_text_spans(page)
    return visible <= 50


def _data_to_text(data: dict) -> str:
    """Tesseract word boxes → text with the original line and paragraph breaks."""
    out, last_par, line_words, last_line = [], None, [], None
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word or float(data["conf"][i]) < 0:
            continue
        par = (data["block_num"][i], data["par_num"][i])
        line = par + (data["line_num"][i],)
        if line != last_line and line_words:
            out.append(" ".join(line_words)); line_words = []
        if last_par is not None and par != last_par:
            out.append("")
        line_words.append(word)
        last_line, last_par = line, par
    if line_words:
        out.append(" ".join(line_words))
    return "\n".join(out).strip()


def _confidence(data: dict) -> float:
    confs = [float(c) for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
    return round(sum(confs) / len(confs) / 100, 3) if confs else 0.0


_RTL_LANGS = {"heb", "ara", "fas", "yid"}
_HEB_ARA = re.compile(r"[\u0590-\u06FF]")
_LATIN = re.compile(r"[A-Za-z]")
_BIDI_MARKS = dict.fromkeys(map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e"))


def _box(d: dict, i: int) -> tuple:
    return (d["left"][i], d["top"][i], d["left"][i] + d["width"][i], d["top"][i] + d["height"][i])


def _overlap(a: tuple, b: tuple) -> float:
    """Share of box a covered by box b."""
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / max(1, (a[2] - a[0]) * (a[3] - a[1]))


def _fix_latin_misreads(mixed: dict, rtl_only: dict) -> int:
    """
    With heb+eng, Tesseract regularly reads a Hebrew word as Latin junk
    ("ביום" → "ova", "עם" → "OY"). For Latin-only words inside mostly-Hebrew
    lines, take the Hebrew-only reading of the same spot when it is more
    confident. Real English words (emails, URLs, names read confidently) stay.
    Edits `mixed` in place; returns how many words were replaced.
    """
    lines: dict[tuple, list[int]] = {}
    for i, w in enumerate(mixed["text"]):
        if w.strip() and float(mixed["conf"][i]) >= 0:
            lines.setdefault((mixed["block_num"][i], mixed["par_num"][i], mixed["line_num"][i]), []).append(i)
    rtl_words = [j for j, w in enumerate(rtl_only["text"]) if w.strip() and float(rtl_only["conf"][j]) >= 0]
    replaced = 0
    for idx in lines.values():
        rtl_count = sum(bool(_HEB_ARA.search(mixed["text"][i])) for i in idx)
        if rtl_count * 2 < len(idx):
            continue  # an English line: leave it to the English model
        for i in idx:
            word, conf = mixed["text"][i], float(mixed["conf"][i])
            if not _LATIN.search(word) or _HEB_ARA.search(word) or conf >= 90:
                continue
            if "@" in word or "www" in word.lower() or "://" in word:
                continue
            b = _box(mixed, i)
            cand = [j for j in rtl_words
                    if _overlap(b, _box(rtl_only, j)) > 0.5 or _overlap(_box(rtl_only, j), b) > 0.5]
            if not cand:
                continue
            cand_conf = sum(float(rtl_only["conf"][j]) for j in cand) / len(cand)
            text = " ".join(rtl_only["text"][j].strip() for j in cand)
            if cand_conf > conf and _HEB_ARA.search(text):
                mixed["text"][i] = text
                mixed["conf"][i] = cand_conf
                replaced += 1
    return replaced


def _ocr_pil(img: Image.Image, language: str):
    """Prepare (rotate/enhance) and OCR one image → (prepared image, osd angle, data)."""
    img, angle = _prepare_image(img, with_angle=True)
    data = pytesseract.image_to_data(
        img, lang=language, config="--oem 3 --psm 3",
        output_type=pytesseract.Output.DICT,
    )
    langs = language.split("+")
    rtl = [l for l in langs if l in _RTL_LANGS]
    if rtl and len(rtl) < len(langs):
        rtl_data = pytesseract.image_to_data(
            img, lang="+".join(rtl), config="--oem 3 --psm 3",
            output_type=pytesseract.Output.DICT,
        )
        n = _fix_latin_misreads(data, rtl_data)
        if n:
            logger.info(f"OCR: {n} Latin misreads replaced from the {'+'.join(rtl)} pass")
    # Tesseract sprinkles LRM/RLM marks into RTL output; they break search and copy
    data["text"] = [w.translate(_BIDI_MARKS) for w in data["text"]]
    return img, angle, data


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
        # orientation needs far less detail than reading: a ~1500 px copy is enough
        probe = img
        if max(img.size) > 1500:
            r = 1500 / max(img.size)
            probe = img.resize((int(img.width * r), int(img.height * r)), Image.BILINEAR)
        osd = pytesseract.image_to_osd(
            probe,
            output_type=pytesseract.Output.DICT,
            config="--psm 0",
        )
        angle = osd.get("rotate", 0)
        # a guess on a page with little text can be wrong; only turn the page when sure
        if float(osd.get("orientation_conf", 0)) < 2:
            angle = 0
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
        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)
            for i in target:
                if not (0 <= i < doc.page_count):
                    continue
                page = doc[i]

                # Real (visible) text: use it as is. Invisible text from a scanner
                # app's own OCR is ignored and the page is read again.
                if not _needs_ocr(page):
                    results.append({
                        "page": i + 1,
                        "text": page.get_text().strip(),
                        "confidence": 1.0,
                        "source": "native",
                    })
                    continue

                img = _render_page_gray(page, dpi)
                img, _angle, data = _ocr_pil(img, language)
                del img
                results.append({
                    "page": i + 1,
                    "text": _data_to_text(data),
                    "confidence": _confidence(data),
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
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    visible, invisible = _invisible_text_spans(page)
                    # pages that already have real text are searchable as they are
                    if visible > 50:
                        continue
                    img = _render_page_gray(page, dpi)  # as it looks (rotation applied)
                    img, osd_angle, data = _ocr_pil(img, language)
                    vis_w, vis_h = page.rect.width, page.rect.height
                    if osd_angle % 180:
                        vis_w, vis_h = vis_h, vis_w
                    layer = _text_layer(img.width, img.height, data, vis_w, vis_h)
                    del img
                    if layer is None:
                        continue
                    if invisible:
                        # drop the scanner app's own (usually wrong for Hebrew) OCR layer
                        for r in invisible:
                            page.add_redact_annot(r)
                        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                                              graphics=fitz.PDF_REDACT_LINE_ART_NONE)
                    with layer:
                        target = (page.rect * page.derotation_matrix).normalize()
                        page.show_pdf_page(
                            target, layer, 0, overlay=True,
                            rotate=(page.rotation + osd_angle) % 360,
                        )
                    fitz.TOOLS.store_shrink(100)
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

        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement

        results = OCRService.extract_text(pdf_path, language=language, dpi=dpi)
        doc = Document()

        def add_par(text: str, size: int = 12, bold: bool = False):
            rtl = bool(_RTL_CHARS.search(text))
            para = doc.add_paragraph()
            if rtl:
                # paragraph direction RTL, so periods/brackets land on the correct side in Word
                ppr = para._p.get_or_add_pPr()
                ppr.append(OxmlElement("w:bidi"))
                para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            for line in [text]:
                run = para.add_run(line)
                run.font.size = Pt(size)
                run.bold = bold
                run.font.name = "David" if rtl else "Calibri"
                rpr = run._r.get_or_add_rPr()
                fonts = rpr.find(qn("w:rFonts"))
                if fonts is None:
                    fonts = OxmlElement("w:rFonts"); rpr.append(fonts)
                fonts.set(qn("w:cs"), "David")  # Hebrew/Arabic use the complex-script font
                if rtl:
                    rpr.append(OxmlElement("w:rtl"))
            return para

        for n, r in enumerate(results):
            if len(results) > 1:
                add_par(f"— {r['page']} —", size=10, bold=True)
            for block in [b for b in r["text"].split("\n\n") if b.strip()]:
                lines = block.split("\n")
                for k, line in enumerate(lines):
                    para = add_par(line)
                    para.paragraph_format.space_after = Pt(8 if k == len(lines) - 1 else 0)
            if n < len(results) - 1:
                from docx.enum.text import WD_BREAK
                doc.paragraphs[-1].add_run().add_break(WD_BREAK.PAGE)

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
            raw.draft("L", (_MAX_RENDER_SIDE, _MAX_RENDER_SIDE))  # JPEG: decode smaller, cheaper
            img = ImageOps.exif_transpose(raw).convert("L")
        img, _angle, data = _ocr_pil(img, language)
        del img
        text = _data_to_text(data)
        avg_conf = _confidence(data)
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