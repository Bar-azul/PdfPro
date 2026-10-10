"""
TranslateService
================
Translates PDF content using deep-translator (Google Translate, free tier).
Supports 40+ languages including Hebrew, Arabic, and RTL languages.
"""

import html
import io
import json
import logging
import re
import time
from pathlib import Path

import fitz
import httpx
from deep_translator import GoogleTranslator

from ..config import settings
from . import progress
from ..services.pdf_service import _temp_pdf, _ms
from ..utils.errors import ApiError

logger = logging.getLogger(__name__)

RTL_LANGS = {"iw", "he", "ar", "fa", "ur", "yi"}
# line direction (from get_text "dict") → insert_htmlbox rotate value
_DIR_TO_ROTATE = {(1, 0): 0, (0, -1): 90, (-1, 0): 180, (0, 1): 270}


def _html_block(text: str, font_size: float, is_rtl: bool, color: int = 0, bold: bool = False,
                gap: float = 0) -> str:
    """One paragraph of translated text as HTML; dir=auto lets mixed lines order themselves."""
    body = html.escape(text, quote=False).replace("\n", "<br>")
    # no text-align: the default "start" is right for rtl and left for ltr
    direction = "rtl" if is_rtl else "auto"
    weight = "font-weight:bold;" if bold else ""
    return (f'<p dir="{direction}" style="margin:0 0 {gap:.0f}px 0;font-size:{font_size:.1f}px;'
            f'line-height:1.15;color:#{color & 0xFFFFFF:06x};{weight}">{body}</p>')


_PUA = re.compile(r"[\uE000-\uF8FF]")


def _text_units(page: "fitz.Page", erase_icons: bool = False) -> list[dict]:
    """
    The pieces of a page to translate, one per line — except lines that wrap
    into the next one (same size, near the full block width), which are joined
    into one sentence. Translating whole MuPDF blocks sent entire menu columns
    as one text: the model garbled it and the result was squeezed into one box.

    Each unit: text, rect to write into (from the line's start to the block's
    edge, so a longer translation still fits), font size, colour, bold, angle,
    and the exact span boxes to erase. Icon glyphs (private-use characters) stay
    in place, except for right-to-left targets, where the text moves to the
    right edge and an icon left behind would sit on top of it.
    """
    units = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = block["bbox"]
        bw = max(1.0, bx1 - bx0)
        current = None
        for line in block["lines"]:
            spans = [sp for sp in line["spans"] if _PUA.sub("", sp["text"]).strip()]
            if not spans:
                continue
            text = _PUA.sub("", "".join(sp["text"] for sp in line["spans"])).strip()
            size = sorted(sp["size"] for sp in spans)[len(spans) // 2]
            lx0, ly0, lx1, ly1 = line["bbox"]
            boxes = [fitz.Rect(sp["bbox"]) for sp in (line["spans"] if erase_icons else spans)]
            if (current and abs(size - current["size"]) < 0.6
                    and abs(lx0 - current["x0"]) < 6
                    and ly0 - current["y1"] < size * 0.8
                    and current["last_w"] >= 0.8 * bw):
                current["text"] += " " + text
                current["y1"] = ly1
                current["last_w"] = lx1 - lx0
                current["redact"] += boxes
                continue
            sp0 = spans[0]
            dx, dy = line["dir"]
            current = {
                "text": text, "size": size, "x0": lx0, "y0": ly0, "y1": ly1,
                "last_w": lx1 - lx0, "redact": boxes, "block_x1": bx1,
                "color": sp0.get("color", 0), "bold": bool(sp0.get("flags", 0) & 16),
                "angle": _DIR_TO_ROTATE.get((round(dx), round(dy)), 0),
            }
            units.append(current)
    out = []
    prev_bottom = {}
    for u in units:
        if len(u["text"]) < 2:
            continue
        # PDF line boxes can overlap the line above (tall ascenders); start below it
        key = round(u["x0"])
        if u["angle"] == 0 and key in prev_bottom and u["y0"] < prev_bottom[key] < u["y1"]:
            u["y0"] = prev_bottom[key]
        prev_bottom[key] = u["y1"]
        if u["angle"] == 0:
            rect = fitz.Rect(u["x0"], u["y0"], max(u["block_x1"], u["x0"] + 20), u["y1"])
        else:  # sideways text: keep the original box
            rect = fitz.Rect(u["redact"][0])
            for r in u["redact"][1:]:
                rect |= r
        out.append({**u, "rect": rect, "size": max(5, min(u["size"], 36))})
    return out


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
        translator = _Translator(source_language, target_language)
        is_rtl = target_language in RTL_LANGS
        stats = {"tried": 0, "failed": 0}

        try:
            if preserve_layout:
                out = TranslateService._translate_overlay(
                    pdf_path, translator, is_rtl, pages, stats
                )
            else:
                out = TranslateService._translate_clean(
                    pdf_path, translator, is_rtl, pages, stats
                )
        finally:
            translator.close()

        if stats["tried"] == 0:
            out.unlink(missing_ok=True)
            raise ApiError(400, "translate_no_text",
                           "This PDF has no selectable text to translate. If it is a scan, run OCR first.")
        if stats["failed"] == stats["tried"]:
            out.unlink(missing_ok=True)
            logger.error(f"Translation failed for every block; last error: {translator.last_error}")
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
        translator: "_Translator",
        is_rtl: bool,
        pages: list[int] | None,
        stats: dict,
    ) -> Path:
        """Cover each text block with white, then lay the translation into the same box."""
        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)

            target = list(target)
            for k, i in enumerate(target):
                progress.update(k, len(target), "translating")
                if not (0 <= i < doc.page_count):
                    continue
                page = doc[i]
                units = _text_units(page, erase_icons=is_rtl)
                translator.on_progress = (
                    lambda f, k=k, n=len(target): progress.update(k + 0.9 * f, n, "translating"))
                stats["tried"] += len(units)
                translations = translator.translate_many([u["text"] for u in units]) if units else []
                placed = []
                for unit, translated in zip(units, translations):
                    if translated is None:
                        stats["failed"] += 1
                        continue
                    if not translated or translated == unit["text"]:
                        continue
                    placed.append((unit, translated))

                if not placed:
                    continue
                # Remove the original words (a white box alone would leave them selectable
                # underneath), but keep images and drawings.
                for unit, _t in placed:
                    for r in unit["redact"]:
                        page.add_redact_annot(r, fill=False)
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                                      graphics=fitz.PDF_REDACT_LINE_ART_NONE)
                for unit, translated in placed:
                    # insert_htmlbox shapes Hebrew/Arabic, orders bidi text and wraps lines
                    # correctly, and shrinks the text if the translation is longer.
                    page.insert_htmlbox(
                        unit["rect"],
                        _html_block(translated, unit["size"], is_rtl, unit["color"], unit["bold"]),
                        rotate=unit["angle"], scale_low=0,
                    )

            out = _temp_pdf("translated")
            doc.save(out, deflate=True, garbage=3)
        return out

    # ── Clean strategy ────────────────────────────────────────────────────────

    @staticmethod
    def _translate_clean(
        pdf_path: Path,
        translator: "_Translator",
        is_rtl: bool,
        pages: list[int] | None,
        stats: dict,
    ) -> Path:
        """Create a new clean PDF with the translated text only (flows onto extra pages if needed)."""
        with fitz.open(pdf_path) as doc:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)
            new_doc = fitz.open()

            target = list(target)
            for k, i in enumerate(target):
                progress.update(k, len(target), "translating")
                if not (0 <= i < doc.page_count):
                    continue
                page = doc[i]
                paragraphs = [b[4].strip() for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
                if not paragraphs:
                    new_doc.new_page(width=page.rect.width, height=page.rect.height)
                    continue

                stats["tried"] += len(paragraphs)
                translated = translator.translate_many(paragraphs)
                stats["failed"] += sum(t is None for t in translated)
                translated = [t if t is not None else o for t, o in zip(translated, paragraphs)]

                html = "".join(_html_block(par, 11, is_rtl, gap=6) for par in translated if par.strip())
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
        translator = _Translator(source_language, target_language)
        try:
            result = translator.translate(text)
        finally:
            translator.close()
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


# ── Translation backend ───────────────────────────────────────────────────────

_GTX_URL = "https://translate.googleapis.com/translate_a/single"
_BATCH_CHARS = 1500  # per request; keeps the GET URL well under URL-length limits


class _Translator:
    """
    Translation routes, best first: Azure AI Translator, Google Cloud
    Translation and Cloudflare Workers AI (a chat model, then m2m100) when their keys are set
    (official APIs, work from cloud servers),
    then Google's public "gtx" endpoint and deep-translator's page scraper —
    those two are free but Google blocks them from datacenter IPs (Render). One request per block — the old way — gets an
    IP throttled after a few dozen blocks.
    """

    def __init__(self, source: str = "auto", target: str = "en"):
        self.source = source or "auto"
        self.target = target
        self.last_error: str | None = None
        self._client = httpx.Client(timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        self._fallback = None
        self._latin_lang: str | None = None
        self.on_progress = None  # called with the fraction (0-1) of the current batch done  # document's Latin-script language, detected once
        # circuit breaker: after 3 failures in a row a route is skipped for this
        # document, so a blocked service fails fast instead of retrying every block
        self._fails: dict[str, int] = {}

    # ── official APIs (used when a key is configured) ─────────────────────────

    def _routes_many(self):
        """Routes that take a list of texts, best first."""
        routes = []
        if settings.AZURE_TRANSLATOR_KEY:
            routes.append(self._azure)
        if settings.GOOGLE_TRANSLATE_API_KEY:
            routes.append(self._gcloud)
        if settings.CLOUDFLARE_ACCOUNT_ID and settings.CLOUDFLARE_API_TOKEN:
            if settings.CLOUDFLARE_LLM_MODEL:
                routes.append(self._cloudflare_llm)
            routes.append(self._cloudflare)
        return routes

    def _azure(self, texts: list[str]) -> list[str]:
        """Azure AI Translator v3: up to 1000 texts / 50k chars per call (we send ≤ 1500 chars)."""
        params = {"api-version": "3.0", "to": _azure_code(self.target)}
        if self.source and self.source != "auto":
            params["from"] = _azure_code(self.source)
        headers = {"Ocp-Apim-Subscription-Key": settings.AZURE_TRANSLATOR_KEY,
                   "Content-Type": "application/json"}
        if settings.AZURE_TRANSLATOR_REGION:
            headers["Ocp-Apim-Subscription-Region"] = settings.AZURE_TRANSLATOR_REGION
        r = self._client.post("https://api.cognitive.microsofttranslator.com/translate",
                              params=params, headers=headers, json=[{"Text": t} for t in texts])
        if r.status_code != 200:
            raise RuntimeError(f"azure HTTP {r.status_code}: {r.text[:200]}")
        return [item["translations"][0]["text"] for item in r.json()]

    def _gcloud(self, texts: list[str]) -> list[str]:
        """Google Cloud Translation v2 (official, keyed)."""
        body = {"q": texts, "target": self.target, "format": "text"}
        if self.source and self.source != "auto":
            body["source"] = self.source
        r = self._client.post("https://translation.googleapis.com/language/translate/v2",
                              params={"key": settings.GOOGLE_TRANSLATE_API_KEY}, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"gcloud HTTP {r.status_code}: {r.text[:200]}")
        return [html.unescape(t["translatedText"]) for t in r.json()["data"]["translations"]]

    def _cloudflare_llm(self, texts: list[str]) -> list[str]:
        """
        Cloudflare Workers AI chat model (Gemma by default). Much better than m2m100
        on short, name-heavy text such as menus: m2m100 translates dish names word by
        word and loops ("tomato tomato tomato"). The whole batch goes in one call as
        numbered JSON so the model sees the context; any missing or broken item
        fails the batch and the next route (m2m100) takes it.
        """
        from ..models.schemas import SUPPORTED_LANGUAGES
        url = (f"https://api.cloudflare.com/client/v4/accounts/{settings.CLOUDFLARE_ACCOUNT_ID}"
               f"/ai/run/{settings.CLOUDFLARE_LLM_MODEL}")
        headers = {"Authorization": f"Bearer {settings.CLOUDFLARE_API_TOKEN}"}
        target = f"{SUPPORTED_LANGUAGES.get(self.target, self.target)} (language code {self.target})"
        source = ("" if self.source in ("", "auto") else
                  f" from {SUPPORTED_LANGUAGES.get(self.source, self.source)}")
        system = (
            f"You are a professional translator. Translate each value of the JSON object{source} "
            f"into {target}. Rules: translate meaning naturally, not word by word. "
            "Proper names, brand names and foreign dish names (e.g. Thai or Italian dishes) "
            "are transliterated into the target script, not translated literally. "
            "Keep numbers, prices, currency codes, units, times, e-mail addresses, URLs and "
            "markers such as (V), (N), (P) exactly as they are. Keep the same keys. "
            "Reply with the JSON object only, no explanations."
        )
        payload = {str(i + 1): t for i, t in enumerate(texts)}
        body = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "temperature": 0.1,
            "max_tokens": min(8000, 400 + 4 * sum(len(t) for t in texts)),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = self._client.post(url, headers=headers, json=body, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"cloudflare-llm HTTP {r.status_code}: {r.text[:200]}")
        data = _llm_json(_llm_text(r.json()))
        out = []
        for i, src in enumerate(texts):
            val = data.get(str(i + 1))
            if not isinstance(val, str) or not val.strip() or _degenerate(src, val):
                raise RuntimeError(f"cloudflare-llm: bad item {i + 1}")
            out.append(val.strip())
        if self.on_progress:
            self.on_progress(1.0)
        return out

    def _cloudflare(self, texts: list[str]) -> list[str]:
        """
        Cloudflare Workers AI, model m2m100-1.2b (free daily allowance). One text per
        call, so calls run 16 at a time. m2m100 needs the source language: with
        "auto" it is detected by script per block, and for Latin-script blocks from
        all the document's text at once (short menu items alone get misdetected).
        """
        from concurrent.futures import ThreadPoolExecutor
        url = (f"https://api.cloudflare.com/client/v4/accounts/{settings.CLOUDFLARE_ACCOUNT_ID}"
               f"/ai/run/@cf/meta/m2m100-1.2b")
        headers = {"Authorization": f"Bearer {settings.CLOUDFLARE_API_TOKEN}"}
        target = _m2m_code(self.target)

        if self.source and self.source != "auto":
            fixed_src = _m2m_code(self.source)
        else:
            fixed_src = None
            if self._latin_lang is None:
                latin = " ".join(t for t in texts if _script_lang(t) is None)
                self._latin_lang = _detect_lang(latin) if latin.strip() else "en"

        def one(text: str) -> str:
            src = fixed_src or _script_lang(text) or self._latin_lang
            if src == target:
                return text
            r = self._client.post(url, headers=headers,
                                  json={"text": text, "source_lang": src, "target_lang": target})
            if r.status_code != 200:
                raise RuntimeError(f"cloudflare HTTP {r.status_code}: {r.text[:200]}")
            out = r.json()["result"]["translated_text"]
            return text if _degenerate(text, out) else out

        results = []
        with ThreadPoolExecutor(max_workers=16) as pool:
            for r in pool.map(one, texts):
                results.append(r)
                if self.on_progress:
                    self.on_progress(len(results) / len(texts))
        return results

    def _try_many(self, texts: list[str]) -> list[str] | None:
        """Batch through the official APIs; None if none is configured or all failed."""
        for route in self._routes_many():
            name = route.__name__
            if not self._available(name):
                continue
            try:
                out = route(texts)
                if len(out) == len(texts):
                    self._record(name, True)
                    return out
                raise RuntimeError(f"{name}: got {len(out)} results for {len(texts)} texts")
            except Exception as e:
                self._record(name, False)
                self.last_error = f"{name}: {e}"
                logger.warning(f"Translation failed via {name}: {e}")
        return None

    def _available(self, name: str) -> bool:
        return self._fails.get(name, 0) < 3

    def _record(self, name: str, ok: bool):
        self._fails[name] = 0 if ok else self._fails.get(name, 0) + 1

    def close(self):
        self._client.close()

    def _gtx(self, text: str) -> str:
        """One request to the gtx endpoint, with a short retry on throttling/5xx."""
        params = {"client": "gtx", "sl": self.source, "tl": self.target, "dt": "t", "q": text}
        for attempt in range(3):
            r = self._client.get(_GTX_URL, params=params)
            if r.status_code == 200:
                data = r.json()
                return "".join(seg[0] for seg in (data[0] or []) if seg and seg[0])
            if r.status_code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"gtx HTTP {r.status_code}: {r.text[:200]}")
        raise RuntimeError("gtx: retries exhausted")

    def _deep(self, text: str) -> str:
        """deep-translator (scrapes translate.google.com/m) as a second route."""
        if self._fallback is None:
            self._fallback = GoogleTranslator(source=self.source, target=self.target)
        return self._fallback.translate(text)

    def translate(self, text: str) -> str | None:
        """Translate one text (any length); None if every route failed."""
        text = text.strip()
        if not text:
            return text
        chunks = _chunks(text, _BATCH_CHARS)
        out = []
        for chunk in chunks:
            done = None
            official = self._try_many([chunk])
            if official:
                out.append(official[0])
                continue
            for route in (self._gtx, self._deep):
                name = route.__name__
                if not self._available(name):
                    continue
                try:
                    done = route(chunk)
                    self._record(name, bool(done))
                    if done:
                        break
                except Exception as e:  # keep the reason for the log / error message
                    self._record(name, False)
                    self.last_error = f"{name}: {e}"
                    logger.warning(f"Translation failed via {name}: {e}")
            if not done:
                return None
            out.append(done)
        return " ".join(out)

    def translate_many(self, texts: list[str]) -> list[str | None]:
        """
        Translate many blocks. Blocks without letters (prices, numbers, dates) are
        kept as they are, and repeated blocks (menu headings, "Spicy", footers on
        every page) are translated once — on a menu that is often half the calls.
        """
        out: list[str | None] = [None] * len(texts)
        unique: dict[str, list[int]] = {}
        for i, t in enumerate(texts):
            key = " ".join(t.split())
            if not _HAS_LETTER.search(key):
                out[i] = t
            else:
                unique.setdefault(key, []).append(i)
        if unique:
            keys = list(unique)
            for key, done in zip(keys, self._translate_unique(keys)):
                for i in unique[key]:
                    out[i] = done
        return out

    def _translate_unique(self, texts: list[str]) -> list[str | None]:
        """Translate distinct blocks with few requests: one line per block, joined by newlines."""
        results: list[str | None] = [None] * len(texts)
        batch: list[int] = []
        size = 0

        def flush():
            nonlocal batch, size
            if not batch:
                return
            # a block's line breaks are just where the PDF wrapped it: send it as one
            # sentence (as the gtx path does) so it translates and re-wraps cleanly
            official = self._try_many([" ".join(texts[i].split()) for i in batch])
            if official:
                for i, part in zip(batch, official):
                    results[i] = part.strip()
                batch, size = [], 0
                return
            lines = [" ".join(texts[i].split()) for i in batch]
            joined = None
            if self._available("_gtx"):
                try:
                    joined = self._gtx("\n".join(lines))
                    self._record("_gtx", True)
                except Exception as e:
                    self._record("_gtx", False)
                    self.last_error = f"_gtx: {e}"
                    logger.warning(f"Batch translation failed: {e}")
            parts = joined.split("\n") if joined else []
            if len(parts) == len(batch):
                for i, part in zip(batch, parts):
                    results[i] = part.strip()
            else:  # line count changed (or the batch failed): translate one by one
                for i in batch:
                    results[i] = self.translate(texts[i])
            batch, size = [], 0

        for i, t in enumerate(texts):
            n = len(t)
            if n > _BATCH_CHARS:
                flush()
                results[i] = self.translate(t)
                continue
            if size + n > _BATCH_CHARS:
                flush()
            batch.append(i)
            size += n + 1
        flush()
        return results


_HAS_LETTER = re.compile(r"[^\W\d_]")


def _degenerate(src: str, out: str) -> bool:
    """
    True for machine-translation loops: the same word three or more times in a
    row ("tomato tomato tomato") when the source has no such run, or an output
    far longer than its source.
    """
    def run(text: str) -> int:
        words = re.findall(r"[^\W\d_]+", text.lower())
        best = cur = 1 if words else 0
        for a, b in zip(words, words[1:]):
            cur = cur + 1 if a == b else 1
            best = max(best, cur)
        return best
    if run(out) >= 3 and run(src) < 3:
        return True
    return len(out) > 4 * len(src) + 40


def _llm_text(resp: dict) -> str:
    """Reply text from Workers AI: classic {result: {response}} or OpenAI-style choices."""
    res = resp.get("result", resp)
    if isinstance(res.get("response"), str):
        return res["response"]
    if isinstance(res.get("response"), dict):  # some models parse JSON replies already
        return json.dumps(res["response"], ensure_ascii=False)
    choices = res.get("choices") or []
    if choices:
        return choices[0].get("message", {}).get("content") or choices[0].get("text") or ""
    raise RuntimeError(f"cloudflare-llm: unexpected reply {str(resp)[:200]}")


def _llm_json(text: str) -> dict:
    """The JSON object in a model reply (tolerates code fences or a stray thought block)."""
    text = re.sub(r"<(think|thought)>.*?</\1>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("cloudflare-llm: no JSON in reply")
    return json.loads(text[start:end + 1])


def _m2m_code(code: str) -> str:
    """Google-style codes → m2m100 codes."""
    return {"iw": "he", "zh-CN": "zh", "zh-TW": "zh", "jw": "jv"}.get(code, code)


_SCRIPTS = [
    (re.compile(r"[\u0590-\u05FF]"), "he"), (re.compile(r"[\u0600-\u06FF]"), "ar"),
    (re.compile(r"[\u0E00-\u0E7F]"), "th"), (re.compile(r"[\u0400-\u04FF]"), "ru"),
    (re.compile(r"[\uAC00-\uD7AF]"), "ko"), (re.compile(r"[\u3040-\u30FF]"), "ja"),
    (re.compile(r"[\u4E00-\u9FFF]"), "zh"), (re.compile(r"[\u0370-\u03FF]"), "el"),
]


def _script_lang(text: str) -> str | None:
    """Language implied by a non-Latin script (Hebrew, Thai, Arabic...), else None."""
    n, code = max((len(rx.findall(text)), code) for rx, code in _SCRIPTS)
    return code if n >= 2 else None


def _detect_lang(text: str) -> str:
    """Source language for m2m100: by script, then langdetect for Latin-script text."""
    code = _script_lang(text)
    if code:
        return code
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        found = detect(text)
        return {"zh-cn": "zh", "zh-tw": "zh"}.get(found, found)
    except Exception:
        return "en"


def _azure_code(code: str) -> str:
    """Google-style codes → Azure codes where they differ."""
    return {"iw": "he", "zh": "zh-Hans", "zh-CN": "zh-Hans", "zh-TW": "zh-Hant", "jw": "jv"}.get(code, code)


def _chunks(text: str, limit: int) -> list[str]:
    """Split long text at line or sentence ends, never mid-word, into pieces <= limit."""
    if len(text) <= limit:
        return [text]
    pieces, cur = [], ""
    for part in re.split(r"(?<=[\n.!?։׃])\s+", text):
        while len(part) > limit:  # one huge sentence: cut at the last space
            cut = part.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append(part[:cut]); part = part[cut:].lstrip()
        if cur and len(cur) + 1 + len(part) > limit:
            pieces.append(cur); cur = part
        else:
            cur = f"{cur} {part}" if cur else part
    if cur:
        pieces.append(cur)
    return pieces


