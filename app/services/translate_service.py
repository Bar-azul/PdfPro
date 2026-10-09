"""
TranslateService
================
Translates PDF content using deep-translator (Google Translate, free tier).
Supports 40+ languages including Hebrew, Arabic, and RTL languages.
"""

import html
import io
import logging
import time
from pathlib import Path

import fitz
from deep_translator import GoogleTranslator

from ..services.pdf_service import _temp_pdf, _ms
from ..utils.errors import ApiError

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 4500
RTL_LANGS = {"iw", "he", "ar", "fa", "ur", "yi"}
# line direction (from get_text "dict") → insert_htmlbox rotate value
_DIR_TO_ROTATE = {(1, 0): 0, (0, -1): 90, (-1, 0): 180, (0, 1): 270}


def _html_block(text: str, font_size: float, is_rtl: bool) -> str:
    """One paragraph of translated text as HTML; dir=auto lets mixed lines order themselves."""
    body = html.escape(text, quote=False).replace("\n", "<br>")
    # no text-align: the default "start" is right for rtl and left for ltr
    direction = "rtl" if is_rtl else "auto"
    return (f'<p dir="{direction}" style="margin:0 0 6px 0;font-size:{font_size:.1f}px;'
            f'line-height:1.2">{body}</p>')


class TranslateService:

    @staticmethod
    def translate_pdf(
        pdf_path: Path,
        target_language: str,
        source_language: str = "auto",
        preserve_layout: bool = True,
        pages: list[int] | None = None,
    ) -> Path:
        t0 = time.time()
        translator = GoogleTranslator(source=source_language, target=target_language)
        is_rtl = target_language in RTL_LANGS
        stats = {"tried": 0, "failed": 0}

        if preserve_layout:
            out = TranslateService._translate_overlay(
                pdf_path, translator, is_rtl, pages, stats
            )
        else:
            out = TranslateService._translate_clean(
                pdf_path, translator, is_rtl, pages, stats
            )

        if stats["tried"] == 0:
            out.unlink(missing_ok=True)
            raise ApiError(400, "translate_no_text",
                           "This PDF has no selectable text to translate. If it is a scan, run OCR first.")
        if stats["failed"] == stats["tried"]:
            out.unlink(missing_ok=True)
            raise ApiError(502, "translate_failed",
                           "The translation service is not responding right now. Please try again in a few minutes.")

        logger.info(
            f"Translated PDF ({source_language}→{target_language}) in {_ms(t0)}ms, "
            f"{stats['failed']}/{stats['tried']} blocks failed"
        )
        return out

    # ── Overlay strategy ──────────────────────────────────────────────────────

    @staticmethod
    def _translate_overlay(
        pdf_path: Path,
        translator: GoogleTranslator,
        is_rtl: bool,
        pages: list[int] | None,
        stats: dict,
    ) -> Path:
        """Cover each text block with white, then lay the translation into the same box."""
        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)

            for i in target:
                if not (0 <= i < doc.page_count):
                    continue
                page = doc[i]
                placed = []
                for block in page.get_text("dict")["blocks"]:
                    if block.get("type") != 0 or not block.get("lines"):
                        continue
                    text = "\n".join(
                        "".join(span["text"] for span in line["spans"]) for line in block["lines"]
                    ).strip()
                    if len(text) < 2:
                        continue

                    stats["tried"] += 1
                    translated = _safe_translate(translator, text)
                    if translated is None:
                        stats["failed"] += 1
                        continue
                    if not translated or translated == text:
                        continue
                    sizes = [span["size"] for line in block["lines"] for span in line["spans"] if span["text"].strip()]
                    font_size = max(6, min(sorted(sizes)[len(sizes) // 2] if sizes else 11, 28))
                    # follow the direction of the original lines (text can run sideways on
                    # scanned/rotated pages); coordinates here are unrotated page space
                    dx, dy = block["lines"][0]["dir"]
                    angle = _DIR_TO_ROTATE.get((round(dx), round(dy)), 0)
                    placed.append((fitz.Rect(block["bbox"]), translated, font_size, angle))

                if not placed:
                    continue
                # Remove the original words (a white box alone would leave them selectable
                # underneath), but keep images and drawings.
                for rect, *_rest in placed:
                    page.add_redact_annot(rect, fill=(1, 1, 1))
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                                      graphics=fitz.PDF_REDACT_LINE_ART_NONE)
                for rect, translated, font_size, angle in placed:
                    # insert_htmlbox shapes Hebrew/Arabic, orders bidi text and wraps lines
                    # correctly, and shrinks the text if the translation is longer.
                    page.insert_htmlbox(
                        rect, _html_block(translated, font_size, is_rtl),
                        rotate=angle, scale_low=0,
                    )

            out = _temp_pdf("translated")
            doc.save(out, deflate=True, garbage=3)
        return out

    # ── Clean strategy ────────────────────────────────────────────────────────

    @staticmethod
    def _translate_clean(
        pdf_path: Path,
        translator: GoogleTranslator,
        is_rtl: bool,
        pages: list[int] | None,
        stats: dict,
    ) -> Path:
        """Create a new clean PDF with the translated text only (flows onto extra pages if needed)."""
        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)
            new_doc = fitz.open()

            for i in target:
                if not (0 <= i < doc.page_count):
                    continue
                page = doc[i]
                original_text = page.get_text().strip()
                if not original_text:
                    new_doc.new_page(width=page.rect.width, height=page.rect.height)
                    continue

                stats["tried"] += 1
                translated = _safe_translate(translator, original_text)
                if translated is None:
                    stats["failed"] += 1
                    translated = original_text

                html = "".join(
                    _html_block(par, 11, is_rtl) for par in translated.split("\n\n") if par.strip()
                )
                mediabox = fitz.Rect(0, 0, page.rect.width, page.rect.height)
                where = mediabox + (50, 50, -50, -50)
                story = fitz.Story(html=html)
                buf = io.BytesIO()
                writer = fitz.DocumentWriter(buf)
                more = 1
                while more:
                    dev = writer.begin_page(mediabox)
                    more, _ = story.place(where)
                    story.draw(dev)
                    writer.end_page()
                writer.close()
                with fitz.open("pdf", buf.getvalue()) as part:
                    new_doc.insert_pdf(part)

            if new_doc.page_count == 0:
                new_doc.new_page()
            out = _temp_pdf("translated")
            new_doc.save(out, deflate=True, garbage=3)
            new_doc.close()

        return out

    @staticmethod
    def translate_text(
        text: str,
        target_language: str,
        source_language: str = "auto",
    ) -> str:
        translator = GoogleTranslator(source=source_language, target=target_language)
        result = _safe_translate(translator, text)
        if result is None:
            raise ApiError(502, "translate_failed",
                           "The translation service is not responding right now. Please try again in a few minutes.")
        return result

    @staticmethod
    def get_supported_languages() -> dict:
        try:
            return GoogleTranslator().get_supported_languages(as_dict=True)
        except Exception:
            from ..models.schemas import SUPPORTED_LANGUAGES
            return SUPPORTED_LANGUAGES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_translate(translator: GoogleTranslator, text: str) -> str | None:
    text = text.strip()
    if not text:
        return text
    try:
        if len(text) <= _CHUNK_SIZE:
            return translator.translate(text)
        chunks = [text[i:i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)]
        return " ".join(translator.translate(chunk) for chunk in chunks)
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return None