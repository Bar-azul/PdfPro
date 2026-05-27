"""
TranslateService
================
Translates PDF content using deep-translator (Google Translate, free tier).
Supports 40+ languages including Hebrew, Arabic, and RTL languages.
"""

import logging
import os
import time
from pathlib import Path

import fitz
from deep_translator import GoogleTranslator

try:
    from bidi.algorithm import get_display
    from arabic_reshaper import reshape
    BIDI_AVAILABLE = True
except ImportError:
    BIDI_AVAILABLE = False

from ..services.pdf_service import _temp_pdf, _ms

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 4500
RTL_LANGS = {"iw", "he", "ar", "fa", "ur", "yi"}

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/Arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/Tahoma.ttf",
    "C:/Windows/Fonts/times.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _get_font_path() -> str | None:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _fix_rtl(text: str, lang: str) -> str:
    """
    Fix RTL text display in PDF:
    - If python-bidi is installed: use proper BiDi algorithm (best quality)
    - Fallback: reverse words per line (works for Hebrew without library)
    """
    if not text:
        return text

    if BIDI_AVAILABLE:
        try:
            # Arabic needs reshaping before BiDi
            if lang == "ar":
                text = reshape(text)
            return get_display(text)
        except Exception as e:
            logger.debug(f"BiDi failed, using fallback: {e}")

    # Fallback — reverse words per line
    lines = text.splitlines()
    fixed = []
    for line in lines:
        words = line.split()
        if words:
            fixed.append(" ".join(reversed(words)))
        else:
            fixed.append(line)
    return "\n".join(fixed)


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
        font_path = _get_font_path()

        if not BIDI_AVAILABLE and is_rtl:
            logger.warning(
                "python-bidi not installed — RTL text may appear reversed. "
                "Run: pip install python-bidi arabic-reshaper"
            )

        if preserve_layout:
            out = TranslateService._translate_overlay(
                pdf_path, translator, is_rtl, target_language, pages, font_path
            )
        else:
            out = TranslateService._translate_clean(
                pdf_path, translator, is_rtl, target_language, pages, font_path
            )

        logger.info(f"Translated PDF ({source_language}→{target_language}) in {_ms(t0)}ms")
        return out

    # ── Overlay strategy ──────────────────────────────────────────────────────

    @staticmethod
    def _translate_overlay(
        pdf_path: Path,
        translator: GoogleTranslator,
        is_rtl: bool,
        lang: str,
        pages: list[int] | None,
        font_path: str | None,
    ) -> Path:
        """Cover each text block with white, then overlay translated text."""
        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)

            for i in target:
                if not (0 <= i < doc.page_count):
                    continue
                page = doc[i]
                blocks = page.get_text("blocks")

                for block in blocks:
                    x0, y0, x1, y1, text, *_ = block
                    text = text.strip()
                    if not text or len(text) < 2:
                        continue

                    translated = _safe_translate(translator, text)
                    if not translated or translated == text:
                        continue

                    # Fix RTL direction
                    if is_rtl:
                        translated = _fix_rtl(translated, lang)

                    rect = fitz.Rect(x0, y0, x1, y1)
                    page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), overlay=True)

                    lines = max(text.count("\n") + 1, 1)
                    font_size = max(7, min((y1 - y0) / lines * 0.75, 13))

                    try:
                        kwargs = dict(
                            fontsize=font_size,
                            color=(0, 0, 0),
                            align=2 if is_rtl else 0,
                            overlay=True,
                        )
                        if font_path:
                            kwargs["fontfile"] = font_path
                            kwargs["fontname"] = "custom"
                        page.insert_textbox(rect, translated, **kwargs)
                    except Exception as e:
                        logger.debug(f"insert_textbox failed: {e}")
                        try:
                            page.insert_text(
                                fitz.Point(x0 + 2, y0 + font_size + 2),
                                translated,
                                fontsize=font_size,
                                color=(0, 0, 0),
                            )
                        except Exception:
                            pass

            out = _temp_pdf("translated")
            doc.save(out, deflate=True)
        return out

    # ── Clean strategy ────────────────────────────────────────────────────────

    @staticmethod
    def _translate_clean(
        pdf_path: Path,
        translator: GoogleTranslator,
        is_rtl: bool,
        lang: str,
        pages: list[int] | None,
        font_path: str | None,
    ) -> Path:
        """Create a new clean PDF with translated text only."""
        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)
            new_doc = fitz.open()

            for i in target:
                if not (0 <= i < doc.page_count):
                    continue

                page = doc[i]
                original_text = page.get_text()
                translated = _safe_translate(translator, original_text) or original_text

                # Fix RTL direction
                if is_rtl:
                    translated = _fix_rtl(translated, lang)

                new_page = new_doc.new_page(
                    width=page.rect.width,
                    height=page.rect.height,
                )

                margin = 50
                text_rect = fitz.Rect(
                    margin, margin,
                    new_page.rect.width - margin,
                    new_page.rect.height - margin,
                )

                try:
                    kwargs = dict(
                        fontsize=11,
                        color=(0, 0, 0),
                        align=2 if is_rtl else 0,
                    )
                    if font_path:
                        kwargs["fontfile"] = font_path
                        kwargs["fontname"] = "custom"
                    new_page.insert_textbox(text_rect, translated, **kwargs)
                except Exception as e:
                    logger.debug(f"insert_textbox failed on page {i}: {e}")
                    y = margin + 15
                    for line in translated.splitlines():
                        if y > new_page.rect.height - margin:
                            break
                        try:
                            new_page.insert_text(
                                fitz.Point(margin, y),
                                line,
                                fontsize=11,
                                color=(0, 0, 0),
                            )
                        except Exception:
                            pass
                        y += 16

            out = _temp_pdf("translated")
            new_doc.save(out, deflate=True)
            new_doc.close()

        return out

    @staticmethod
    def translate_text(
        text: str,
        target_language: str,
        source_language: str = "auto",
    ) -> str:
        translator = GoogleTranslator(source=source_language, target=target_language)
        result = _safe_translate(translator, text) or text
        if target_language in RTL_LANGS:
            result = _fix_rtl(result, target_language)
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