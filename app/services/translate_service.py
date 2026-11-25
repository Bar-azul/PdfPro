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
import unicodedata
from pathlib import Path

import fitz
import httpx
from deep_translator import GoogleTranslator

from ..config import settings
from . import progress
from ..services.pdf_service import _temp_pdf, _ms
from ..utils.errors import ApiError, daily_quota_error, is_daily_quota

logger = logging.getLogger(__name__)

RTL_LANGS = {"iw", "he", "ar", "fa", "ur", "yi"}
# line direction (from get_text "dict") → insert_htmlbox rotate value
_DIR_TO_ROTATE = {(1, 0): 0, (0, -1): 90, (-1, 0): 180, (0, 1): 270}


def _html_block(text: str, font_size: float, is_rtl: bool, color: int = 0, bold: bool = False,
                gap: float = 0, align_right: bool = False) -> str:
    """One paragraph of translated text as HTML; dir=auto lets mixed lines order themselves."""
    if is_rtl:
        # "(V)" after Latin or a price in a right-to-left line: without marks the bidi
        # algorithm gives the closing bracket the line's direction and it lands at
        # the wrong end, mirrored ("(Tom Yam Hed (V"). LRMs on both sides keep the
        # group together.
        text = _LATIN_GROUP.sub("\u200e(\\1)\u200e", text)
    body = html.escape(text, quote=False).replace("\n", "<br>")
    # no text-align: the default "start" is right for rtl and left for ltr
    direction = "rtl" if is_rtl else "auto"
    weight = "font-weight:bold;" if bold else ""
    if align_right and not is_rtl:
        weight += "text-align:right;"
    return (f'<p dir="{direction}" style="margin:0 0 {gap:.0f}px 0;font-size:{font_size:.1f}px;'
            f'line-height:1.15;color:#{color & 0xFFFFFF:06x};{weight}">{body}</p>')


def _compact(doc: "fitz.Document"):
    """
    insert_htmlbox embeds a full copy of its fonts in every box, so a translated
    4-page menu came out at 9.6 MB from a 150 KB original. Subsetting the fonts
    (plus garbage=4 on save, which merges identical objects) brings it to ~0.3 MB.
    """
    try:
        doc.subset_fonts()
    except Exception as e:  # smaller file is nice to have, never a reason to fail
        logger.warning(f"Font subsetting failed: {e}")


_COMPACT_EVERY = 8   # pages


def _compacted(doc: "fitz.Document") -> "fitz.Document":
    """
    The same document with its embedded fonts subset and duplicates merged, reopened
    from memory. Every translated box embeds a full copy of its fonts (about 0.2 MB),
    and they were only merged when saving at the end: a 104-page file peaked at
    ~550 MB and a long document brought the server down mid-request (the browser
    only saw a dropped connection). Doing it every few pages keeps memory flat.
    """
    _compact(doc)
    data = doc.tobytes(deflate=True, garbage=4)
    doc.close()
    return fitz.open("pdf", data)


_RTL_TEXT = re.compile(r"[\u0590-\u05FF\u0600-\u06FF]")
_LATIN_GROUP = re.compile(r"\(([^()\u0590-\u05FF\u0600-\u06FF]*)\)")
_PUA = re.compile(r"[\uE000-\uF8FF]")


_BULLETS = set("•◦▪▫●○■□‣⁃∙➢➤►▶✓✔")
_NUMBERED = re.compile(r"^(\d{1,2}|[a-zA-Zא-ת])[.)]\s")
_MARKER_ONLY = re.compile(r"^[\s.)(\d" + "".join(sorted(_BULLETS)) + r"]{1,4}$")
_CELL_GAP = 1.5     # a horizontal gap wider than this many font sizes separates two cells
_SPACE_GAP = 0.22   # a gap wider than this (with no space character in it) is a word space


def _line_pieces(page: "fitz.Page", block: dict, erase_icons: bool, page_rtl: bool = False) -> list[dict]:
    """
    The horizontal text of one block as pieces: one per table cell / text run on a row.

    MuPDF's lines do not follow the page's cells. In a Hebrew table it put a course
    name and the credits column in one line ("מבוא לכלכלה4"), and it split one
    sentence into three lines at each change of direction. Translated line by line,
    cells ended up in each other's columns and the sentence came out scrambled.
    So the characters are regrouped: lines that share a row are pooled, the row is
    cut wherever there is a cell-sized gap, and each piece is read in logical order
    (right-to-left runs reversed back, numbers and English kept).
    """
    from .rtl_docx import _visual_to_logical_order, rtl_majority

    lines = []
    for line in block["lines"]:
        if tuple(round(v) for v in line["dir"]) != (1, 0):
            continue
        # clusters: a letter with its combining marks (Thai vowels and tones, Hebrew
        # points). Marks sit above the letter and sorting them by position on their
        # own scrambled Thai words.
        chars = []
        for k, sp in enumerate(line["spans"]):
            for ch in sp["chars"]:
                c = ch["c"]
                if chars and unicodedata.category(c) in ("Mn", "Me"):
                    chars[-1]["s"] += c
                    chars[-1]["bbox"] |= fitz.Rect(ch["bbox"])
                    continue
                chars.append({"c": c, "s": c, "bbox": fitz.Rect(ch["bbox"]), "size": sp["size"],
                              "span": (id(line), k),
                              "base": ch["origin"][1], "color": sp.get("color", 0),
                              "flags": sp.get("flags", 0), "alpha": sp.get("alpha", 255)})
        ink = [c for c in chars if c["c"].strip()]
        if ink:
            bases = sorted(c["base"] for c in ink)
            lines.append({"bbox": fitz.Rect(line["bbox"]), "chars": chars, "base": bases[len(bases) // 2],
                          "size": max(c["size"] for c in ink)})

    # rows: lines on the same baseline (line boxes can be much taller than the text
    # and overlap the next line, so the boxes themselves are not a reliable test)
    rows = []
    for ln in sorted(lines, key=lambda l: l["base"]):
        for row in rows:
            if abs(ln["base"] - row["base"]) < 0.4 * min(ln["size"], row["size"]):
                row["lines"].append(ln)
                break
        else:
            rows.append({"base": ln["base"], "size": ln["size"], "lines": [ln]})

    pieces = []
    for row in rows:
        chars = sorted((c for ln in row["lines"] for c in ln["chars"]), key=lambda c: c["bbox"].x0)
        y0 = min(ln["bbox"].y0 for ln in row["lines"])
        y1 = max(ln["bbox"].y1 for ln in row["lines"])
        groups, cur, last = [], [], None
        for c in chars:
            ink = c["c"].strip() and not _PUA.match(c["c"])
            bullet = c["c"] in _BULLETS
            if cur and (bullet or (last is not None and last["c"] in _BULLETS) or
                        (ink and last is not None and c["bbox"].x0 - last["bbox"].x1 > _CELL_GAP * c["size"])):
                # a list bullet is a piece of its own: it is not translated and stays put
                groups.append(cur)
                cur = []
            cur.append(c)
            if ink:
                last = c
        groups.append(cur)
        row_pieces = []
        for g in groups:
            ink = [c for c in g if c["c"].strip() and not _PUA.match(c["c"])]
            if not ink:
                continue
            seq, prev = [], None
            for c in g:
                if _PUA.match(c["c"]):
                    continue
                if (prev is not None and c["c"] != " " and prev["c"] != " "
                        and c["bbox"].x0 - prev["bbox"].x1 > _SPACE_GAP * c["size"]):
                    seq.append((" ", " "))
                if not (c["c"] == " " and seq and seq[-1][0] == " "):
                    seq.append((c["c"], c["s"]))
                prev = c
            visual = "".join(b for b, _ in seq)
            if _RTL_TEXT.search(visual):
                # paragraph direction: the piece's own majority, or the page's when the
                # piece is mostly English inside a Hebrew page ("צד שרת (Python + FastAPI)")
                base = "R" if rtl_majority(visual) or page_rtl else "L"
                order = _visual_to_logical_order(visual, base)
                text = "".join(seq[i][1] for i in order)
            else:
                text = "".join(full for _, full in seq)
            text = re.sub(r"\s+", " ", text).strip()
            # one vote per text run, not per character: a heading with small Thai and
            # large English letters keeps the heading size
            runs = {}
            for c in ink:
                runs.setdefault(c["span"], c["size"])
            sizes = sorted(runs.values())
            box = fitz.Rect(ink[0]["bbox"])
            for c in ink[1:]:
                box |= c["bbox"]
            erase = fitz.Rect(box)
            if erase_icons:
                for c in g:
                    erase |= c["bbox"]
            first = ink[0] if not _RTL_TEXT.search(text) else ink[-1]
            row_pieces.append({
                "text": text, "size": sizes[len(sizes) // 2],
                "x0": box.x0, "x1": box.x1, "y0": y0, "y1": y1,
                "redact": [fitz.Rect(erase.x0, y0, erase.x1, y1)],
                "color": first["color"],
                # bold only when most of the piece is: a bold lead-in ("לחשוב בקול.") shouldn't make the line bold
                "bold": sum(1 for c in ink if c["flags"] & 16) * 2 > len(ink),
                "hidden": all(c["alpha"] == 0 for c in ink),
            })
        # alone on its row (a bullet or list number next to it doesn't count)
        texts = [pc for pc in row_pieces if not _MARKER_ONLY.match(pc["text"])]
        for pc in row_pieces:
            pc["alone"] = len(texts) <= 1
        pieces += row_pieces
    pieces.sort(key=lambda pc: (pc["y0"], pc["x0"]))
    return pieces


def _list_marks(page: "fitz.Page", blocks: list[dict]) -> list["fitz.Rect"]:
    """List bullets on the page: bullet glyphs and small drawn dots/squares."""
    marks = []
    for b in blocks:
        for line in b.get("lines", []):
            for sp in line["spans"]:
                for ch in sp["chars"]:
                    if ch["c"] in _BULLETS:
                        marks.append(fitz.Rect(ch["bbox"]))
    try:
        for d in page.get_drawings():
            r = d["rect"]
            if 1 < r.width <= 10 and 1 < r.height <= 10 and 0.5 <= r.width / r.height <= 2:
                marks.append(fitz.Rect(r))
    except Exception:
        pass
    return marks


def _vertical_edges(page: "fitz.Page") -> list[tuple[float, float, float]]:
    """Vertical lines drawn on the page (table borders) as (x, y0, y1)."""
    edges = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return edges
    for d in drawings:
        for item in d.get("items", []):
            if item[0] == "l":
                a, b = item[1], item[2]
                if abs(a.x - b.x) < 1 and abs(a.y - b.y) > 4:
                    edges.append(((a.x + b.x) / 2, min(a.y, b.y), max(a.y, b.y)))
            elif item[0] == "re":
                r = item[1]
                if r.width < 2 and r.height > 4:      # a thin filled bar drawn as a line
                    edges.append(((r.x0 + r.x1) / 2, r.y0, r.y1))
                elif r.height > 4 and d.get("color") is not None:   # a stroked cell
                    edges += [(r.x0, r.y0, r.y1), (r.x1, r.y0, r.y1)]
    return edges


def _trace_rows(page: "fitz.Page") -> list[dict] | None:
    """
    Horizontal text rebuilt from the page's glyph trace, as rawdict-like blocks (one
    per baseline), or None when MuPDF's normal extraction already has every glyph.

    Some PDFs (Google Docs exports with Hebrew) draw each letter as its own text
    object, right to left. MuPDF's text extraction then drops a large share of the
    letters as if they were overprinted duplicates; whole phrases vanished, were never
    erased and stayed in Hebrew under the English. The glyph trace still has every
    letter with its Unicode value and position.
    """
    try:
        trace = page.get_texttrace()
    except Exception:
        return None
    area = page.rect
    glyphs, seen = [], set()
    for sp in trace:
        if sp.get("type") == 2 or tuple(round(v) for v in sp["dir"]) != (1, 0) or sp.get("wmode"):
            continue
        col = sp.get("color") or (0,)
        if len(col) >= 3:
            color = (int(col[0] * 255) << 16) | (int(col[1] * 255) << 8) | int(col[2] * 255)
        else:
            g = int(col[0] * 255) if col else 0
            color = (g << 16) | (g << 8) | g
        alpha = 0 if sp.get("type") == 3 or sp.get("opacity", 1) == 0 else 255
        for u, _gid, origin, bbox in sp["chars"]:
            if u < 32 and u != 9:
                continue
            r = fitz.Rect(bbox)
            if r.is_empty and chr(u) != " ":
                continue
            if not r.intersects(area):
                continue
            key = (u, round(origin[0]), round(origin[1]))
            if key in seen:            # drawn twice (fake bold): keep one
                continue
            seen.add(key)
            glyphs.append({"c": chr(u), "origin": origin, "bbox": r, "size": sp["size"],
                           "color": color, "flags": sp.get("flags", 0), "alpha": alpha})
    ink = sum(1 for g in glyphs if g["c"].strip())
    have = sum(1 for b in page.get_text("rawdict")["blocks"] for ln in b.get("lines", [])
               if tuple(round(v) for v in ln["dir"]) == (1, 0)
               for sp in ln["spans"] for c in sp["chars"] if c["c"].strip())
    if not ink or have >= 0.97 * ink:
        return None

    x0 = min(g["bbox"].x0 for g in glyphs if g["c"].strip())
    x1 = max(g["bbox"].x1 for g in glyphs if g["c"].strip())
    rows = []
    for g in sorted(glyphs, key=lambda g: g["origin"][1]):
        for row in rows:
            if abs(g["origin"][1] - row["base"]) < 0.3 * min(g["size"], row["size"]):
                row["glyphs"].append(g)
                break
        else:
            rows.append({"base": g["origin"][1], "size": g["size"], "glyphs": [g]})
    blocks = []
    for row in rows:
        gl = row["glyphs"]
        box = fitz.Rect(gl[0]["bbox"])
        for g in gl[1:]:
            box |= g["bbox"]
        spans = [{"size": g["size"], "color": g["color"], "flags": g["flags"], "alpha": g["alpha"],
                  "bbox": tuple(g["bbox"]),
                  "chars": [{"c": g["c"], "bbox": tuple(g["bbox"]), "origin": g["origin"]}]} for g in gl]
        blocks.append({"type": 0, "bbox": (x0, box.y0, x1, box.y1),
                       "lines": [{"dir": (1.0, 0.0), "bbox": tuple(box), "spans": spans}]})
    return blocks


def _text_units(page: "fitz.Page", erase_icons: bool = False) -> list[dict]:
    """
    The pieces of a page to translate: one per table cell or line — except lines
    that wrap into the next one (same size, near the full block width, alone on
    their row), which are joined into one sentence. Translating whole MuPDF blocks
    sent entire menu columns as one text: the model garbled it and the result was
    squeezed into one box.

    Each unit: text, rect to write into (from the line's start edge to the nearest
    neighbour on its row, or the block's edge), font size, colour, bold, angle, and
    the boxes to erase. Icon glyphs (private-use characters) stay in place, except
    for right-to-left targets, where the text moves to the right edge and an icon
    left behind would sit on top of it.
    """
    from .rtl_docx import rtl_majority

    raw = page.get_text("rawdict")["blocks"]
    marks = _list_marks(page, raw)
    page_rtl = rtl_majority("".join(c["c"] for b in raw for ln in b.get("lines", [])
                                    for sp in ln["spans"] for c in sp["chars"]))

    def starts_item(pc, rtl):
        """A bullet (glyph or small drawn shape) just before the line's start edge."""
        mid, gap = (pc["y0"] + pc["y1"]) / 2, 3 * pc["size"]
        for m in marks:
            if not (m.y0 - 2 <= mid <= m.y1 + 2 or pc["y0"] <= (m.y0 + m.y1) / 2 <= pc["y1"]):
                continue
            if (rtl and 0 <= m.x0 - pc["x1"] <= gap) or (not rtl and 0 <= pc["x0"] - m.x1 <= gap):
                return True
        return False

    traced = _trace_rows(page)
    pieces = []
    for block in (traced if traced is not None else raw):
        if block.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = block["bbox"]
        for pc in _line_pieces(page, block, erase_icons, page_rtl):
            pc["rtl"] = bool(_RTL_TEXT.search(pc["text"]) and rtl_majority(pc["text"]))
            pc.update(block_x0=bx0, block_x1=bx1, angle=0)
            pieces.append(pc)

    def edge(pc):
        return pc["x1"] if pc["rtl"] else pc["x0"]

    def column_width(pc):
        """
        What a full line is here: the typical width of nearby lines that start at the
        same edge (the median, so one wide intro line above a column doesn't count).
        """
        widths = sorted(q["x1"] - q["x0"] for q in pieces
                        if q["alone"] and q["rtl"] == pc["rtl"] and abs(edge(q) - edge(pc)) < 6
                        and abs(q["y0"] - pc["y0"]) < 15 * pc["size"])
        return widths[len(widths) // 2] if widths else pc["x1"] - pc["x0"]

    def hanging(cur, pc):
        """The second line of a numbered item starts after the number ("1. ")."""
        if not cur.get("numbered"):
            return False
        indent = edge(cur) - edge(pc) if pc["rtl"] else edge(pc) - edge(cur)
        return 0 < indent < 3 * pc["size"]

    # Lines that wrap into the next one are joined into one sentence, also across
    # MuPDF blocks (a paragraph's second line is often a block of its own): same
    # column edge, same size, right below, the line above is full, and the new line
    # does not start a list item.
    units = []
    for pc in pieces:
        target = None
        if (pc["alone"] and not starts_item(pc, pc["rtl"]) and not _NUMBERED.match(pc["text"])
                and not _MARKER_ONLY.match(pc["text"])):
            for cur in reversed(units[-8:]):
                if (cur["alone"] and cur["angle"] == 0 and not _MARKER_ONLY.match(cur["text"]) and cur["rtl"] == pc["rtl"]
                        and abs(pc["size"] - cur["size"]) < 0.6
                        and (abs(edge(pc) - edge(cur)) < 6 or hanging(cur, pc))
                        and -0.3 * pc["size"] < pc["y0"] - cur["y1"] < pc["size"] * 0.8
                        and cur["last_w"] >= 0.85 * cur["col_w"]):
                    target = cur
                    break
        if target:
            target["text"] += " " + pc["text"]
            target["y1"] = pc["y1"]
            target["x0"] = min(target["x0"], pc["x0"])
            target["x1"] = max(target["x1"], pc["x1"])
            target["last_w"] = pc["x1"] - pc["x0"]
            target["redact"] += pc["redact"]
            target["hidden"] = target["hidden"] and pc["hidden"]
            target["block_x0"] = min(target["block_x0"], pc["block_x0"])
            target["block_x1"] = max(target["block_x1"], pc["block_x1"])
            continue
        units.append({**pc, "last_w": pc["x1"] - pc["x0"], "col_w": column_width(pc),
                      "numbered": bool(_NUMBERED.match(pc["text"]))})

    for block in raw:
        if block.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = block["bbox"]
        # sideways text: one unit per line, in its original box
        for line in block["lines"]:
            dx, dy = (round(v) for v in line["dir"])
            if (dx, dy) == (1, 0):
                continue
            spans = [sp for sp in line["spans"] if _PUA.sub("", "".join(c["c"] for c in sp["chars"])).strip()]
            if not spans:
                continue
            text = _PUA.sub("", "".join(c["c"] for sp in line["spans"] for c in sp["chars"])).strip()
            boxes = [fitz.Rect(sp["bbox"]) for sp in (line["spans"] if erase_icons else spans)]
            lx0, ly0, lx1, ly1 = line["bbox"]
            units.append({
                "text": text, "size": sorted(sp["size"] for sp in spans)[len(spans) // 2],
                "x0": lx0, "x1": lx1, "y0": ly0, "y1": ly1, "block_x0": bx0, "block_x1": bx1,
                "last_w": lx1 - lx0, "redact": boxes, "alone": True, "rtl": False,
                "color": spans[0].get("color", 0), "bold": bool(spans[0].get("flags", 0) & 16),
                "angle": _DIR_TO_ROTATE.get((dx, dy), 0),
                "hidden": all(sp.get("alpha", 255) == 0 for sp in spans),
            })

    # room on the row: a cell's translation may use its whole cell (up to the table's
    # vertical border, when there is one) but may not run into the next cell
    edges = _vertical_edges(page)
    flat = [u for u in units if u["angle"] == 0]
    for u in flat:
        u["left"], u["right"] = u["block_x0"], u["block_x1"]
        h = u["y1"] - u["y0"]
        mid = (u["y0"] + u["y1"]) / 2
        crossing = [x for x, ey0, ey1 in edges if ey0 <= mid <= ey1]
        left_edges = [x for x in crossing if x < u["x0"] - 1]
        right_edges = [x for x in crossing if x > u["x1"] + 1]
        if left_edges:
            u["left"] = max(left_edges) + 3
        if right_edges:
            u["right"] = min(right_edges) - 3
        for v in flat:
            if v is u or min(u["y1"], v["y1"]) - max(u["y0"], v["y0"]) < 0.3 * min(h, v["y1"] - v["y0"]):
                continue
            if v["x1"] <= u["x0"]:
                u["left"] = max(u["left"], v["x1"] + 3)
            elif v["x0"] >= u["x1"]:
                u["right"] = min(u["right"], v["x0"] - 3)

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
        u["align_right"] = False
        if u["angle"] == 0 and not erase_icons and _RTL_TEXT.search(u["text"]):
            # right-to-left source into a left-to-right language: the line starts at
            # its right edge, so keep that edge and give the translation the room to
            # its left (a short right-aligned Hebrew line otherwise left a sliver
            # of space and the English shrank to nothing)
            left = min(u["left"], u["x0"])
            rect = fitz.Rect(left, u["y0"], max(u["x1"], left + 20), u["y1"])
            u["align_right"] = True
        elif u["angle"] == 0:
            right = max(u["right"], u["x1"])
            rect = fitz.Rect(u["x0"], u["y0"], max(right, u["x0"] + 20), u["y1"])
        else:  # sideways text: keep the original box
            rect = fitz.Rect(u["redact"][0])
            for r in u["redact"][1:]:
                rect |= r
        out.append({**u, "rect": rect, "size": max(5, min(u["size"], 36))})
    # OCR layers size each line to fit its width, so on a scan neighbouring lines
    # of the same body text come out at very different sizes (and a translation
    # set at those sizes looks like a ransom note). Keep scanned lines near the
    # page's typical size: headings may stay a little larger.
    hidden = sorted(u["size"] for u in out if u["hidden"])
    if hidden:
        typical = hidden[len(hidden) // 2]
        for u in out:
            if u["hidden"]:
                # one body size for the page; clearly larger lines stay a bit larger
                u["size"] = typical * 1.2 if u["size"] > typical * 1.3 else typical
        # A translation is often longer than the scanned line it replaces. Rather
        # than shrinking it to fit one line's height, let it wrap into the free
        # space below, up to the next line in the same column (at most 2 more lines).
        scanned = sorted((u for u in out if u["hidden"] and u["angle"] == 0), key=lambda u: u["rect"].y0)
        for i, u in enumerate(scanned):
            r = u["rect"]
            limit = r.y1 + 2.4 * u["size"]
            for v in scanned[i + 1:]:
                w = v["rect"]
                if w.y0 >= r.y1 - 1 and w.x0 < r.x1 and w.x1 > r.x0:
                    limit = min(limit, w.y0)
                    break
            u["rect"] = fitz.Rect(r.x0, r.y0, r.x1, max(r.y1, min(limit, page.rect.y1)))
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
            if translator.quota_exhausted:
                raise daily_quota_error()
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
        doc = fitz.open(pdf_path)
        try:
            target = [p - 1 for p in pages] if pages else range(doc.page_count)

            target = [i for i in target if 0 <= i < doc.page_count]
            n = len(target)
            # Read every page first and translate the whole document in one go: the
            # batches run in parallel, the model sees neighbouring lines, and text
            # repeated on every page (footers, notes) is translated once.
            page_units = [_text_units(doc[i], erase_icons=is_rtl) for i in target]
            all_units = [u for units in page_units for u in units]
            stats["tried"] += len(all_units)
            progress.update(0, n, "translating")
            translator.on_progress = lambda f: progress.update(0.95 * f * n, n, "translating")
            translations = translator.translate_many([u["text"] for u in all_units]) if all_units else []

            pos = 0
            for k, (i, units) in enumerate(zip(target, page_units)):
                if k and k % _COMPACT_EVERY == 0:
                    doc = _compacted(doc)
                page = doc[i]
                placed = []
                for unit, translated in zip(units, translations[pos:pos + len(units)]):
                    if translated is None:
                        stats["failed"] += 1
                        continue
                    if not translated or translated == unit["text"]:
                        continue
                    placed.append((unit, translated))
                pos += len(units)

                if not placed:
                    continue
                # Remove the original words (a white box alone would leave them selectable
                # underneath), but keep images and drawings.
                for unit, _t in placed:
                    for r in unit["redact"]:
                        # On a scanned page the original words are pixels in the image
                        # and the text is an invisible OCR layer: removing it would leave
                        # the scan's words under the translation, so paint them white.
                        if unit["hidden"]:
                            # OCR word boxes hug the letters; ink (bold, underlines,
                            # descenders) spills past them, so cover a little more
                            m = r.height * 0.25
                            r = fitz.Rect(r.x0 - 2, r.y0 - m, r.x1 + 2, r.y1 + m) & page.rect
                        else:
                            # line boxes include leading and overlap the rows above and
                            # below; a character is removed if its box touches the area,
                            # so erase only the middle band (a table lost a neighbouring
                            # cell's "92" this way)
                            r = fitz.Rect(r.x0 + 0.5, r.y0 + 0.3 * r.height, r.x1 - 0.5, r.y1 - 0.3 * r.height)
                        page.add_redact_annot(r, fill=(1, 1, 1) if unit["hidden"] else False)
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                                      graphics=fitz.PDF_REDACT_LINE_ART_NONE)
                for unit, translated in placed:
                    # insert_htmlbox shapes Hebrew/Arabic, orders bidi text and wraps lines
                    # correctly, and shrinks the text if the translation is longer.
                    page.insert_htmlbox(
                        unit["rect"],
                        _html_block(translated, unit["size"], is_rtl,
                                    0 if unit["hidden"] else unit["color"], unit["bold"],
                                    align_right=unit["align_right"]),
                        rotate=unit["angle"], scale_low=0,
                    )

            out = _temp_pdf("translated")
            _compact(doc)
            doc.save(out, deflate=True, garbage=4)
        finally:
            doc.close()
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
                # the same reading of the page as the in-place mode: logical order, wrapped
                # lines joined, letters that MuPDF's block text drops recovered
                paragraphs = [u["text"] for u in _text_units(page) if u["text"].strip()]
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
            _compact(new_doc)
            new_doc.save(out, deflate=True, garbage=4)
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
        self.quota_exhausted = False  # Workers AI free daily allowance used up
        self._client = httpx.Client(timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        self._fallback = None
        self._latin_lang: str | None = None
        self.on_progress = None  # called with the fraction (0-1) of translate_many's batches done
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
            f"into {target}. The values are consecutive lines of one document (headings, "
            "titles, body text); treat them consistently. Rules:\n"
            "- Translate meaning naturally, not word by word. Every word that has a meaning "
            "is translated, also in titles and short lines (\"Seasonal fruits\", \"Mixed "
            "Grilled Satay\", \"Mango\" are translated, not kept).\n"
            "- Proper names, brand names and foreign dish names (e.g. \"Tom Yam Goong\") are "
            "transliterated into the target script, not translated literally, followed by "
            "the original Latin spelling, written plainly with a space between them and "
            "no brackets or quotes (for Hebrew, \"Tom Yam Goong\" becomes \"טום יאם גונג "
            "Tom Yam Goong\"). Write each word once. Text in a third script (e.g. Thai) "
            "next to its Latin romanization is dropped; on its own it is transliterated.\n"
            "- Keep numbers, prices, currency codes, units, times, e-mail addresses, URLs "
            "and markers such as (V), (N), (P) exactly as they are.\n"
            "- Return every key, never an empty value. Reply with the JSON object only, "
            "no explanations."
        )
        payload = {str(i + 1): t for i, t in enumerate(texts)}
        body = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "temperature": 0,
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
            if "<" not in src and ">" not in src:  # the model sometimes copies <...> from the rules
                val = val.replace("<", "").replace(">", "")
            out.append(" ".join(val.split()) if "\n" not in src else val.strip())
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

        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(one, texts))

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
                if "HTTP 429" in str(e) and is_daily_quota(429, str(e)):
                    self.quota_exhausted = True
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
        """
        Translate distinct blocks with few requests. Blocks are grouped into batches
        of up to _BATCH_CHARS; with an official API the batches run 4 at a time
        (a chat model takes ~15 s per batch, so a 4-page menu went from ~70 s to
        ~20 s). Batches no official API could do go through gtx one line per block.
        """
        results: list[str | None] = [None] * len(texts)
        batches: list[list[int]] = []
        batch: list[int] = []
        size = 0
        for i, t in enumerate(texts):
            n = len(t)
            if n > _BATCH_CHARS:  # long block: on its own, split into chunks by translate()
                batches.append([i])
                continue
            if batch and size + n > _BATCH_CHARS:
                batches.append(batch)
                batch, size = [], 0
            batch.append(i)
            size += n + 1
        if batch:
            batches.append(batch)

        # a block's line breaks are just where the PDF wrapped it: send it as one
        # sentence so it translates and re-wraps cleanly
        flat = [" ".join(t.split()) for t in texts]
        done = 0

        def report():
            if self.on_progress:
                self.on_progress(done / len(batches))

        def official(idx: list[int]):
            if len(idx) == 1 and len(texts[idx[0]]) > _BATCH_CHARS:
                return None
            return self._try_many([flat[i] for i in idx])

        pending = []
        if self._routes_many():
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as pool:
                for idx, out in zip(batches, pool.map(official, batches)):
                    if out:
                        for i, part in zip(idx, out):
                            results[i] = part.strip()
                        done += 1
                        report()
                    else:
                        pending.append(idx)
        else:
            pending = batches

        for idx in pending:
            if len(idx) == 1 and len(texts[idx[0]]) > _BATCH_CHARS:
                results[idx[0]] = self.translate(texts[idx[0]])
            else:
                self._gtx_batch(idx, flat, texts, results)
            done += 1
            report()
        return results

    def _gtx_batch(self, idx: list[int], flat: list[str], texts: list[str],
                   results: list[str | None]):
        """One gtx request for a batch (one line per block), else block by block."""
        joined = None
        if self._available("_gtx"):
            try:
                joined = self._gtx("\n".join(flat[i] for i in idx))
                self._record("_gtx", True)
            except Exception as e:
                self._record("_gtx", False)
                self.last_error = f"_gtx: {e}"
                logger.warning(f"Batch translation failed: {e}")
        parts = joined.split("\n") if joined else []
        if len(parts) == len(idx):
            for i, part in zip(idx, parts):
                results[i] = part.strip()
        else:  # line count changed (or the batch failed): translate one by one
            for i in idx:
                results[i] = self.translate(texts[i])


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


