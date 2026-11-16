"""
Hebrew / Arabic support for PDF → Word (pdf2docx).

pdf2docx lays text out left to right. On a right-to-left page that breaks
every line that mixes scripts: MuPDF splits such a line into separate pieces
at each change of direction, pdf2docx writes the pieces in left-to-right page
order, and Word shows "(2026) עם מילה Word" as ") 2026 Word עם מילה (". It
also leaves the paragraphs left to right and gives the runs no complex-script
font or size, so Word falls back to its defaults for the Hebrew letters.

Two passes fix it:

* fix_rtl_layout() runs on pdf2docx's parsed pages, before the .docx is made.
  For each row of a text block that contains RTL letters it takes every
  character with its position, puts them in left-to-right page (visual) order,
  turns that back into reading (logical) order with the Unicode bidi algorithm,
  and rebuilds the row as one line whose spans follow that order. Each
  character keeps the span (font, size, colour, bold) it came from.

  The same pass also: restores word spaces that some PDF producers (Chrome's
  "Save as PDF") leave out and pdf2docx drops; keeps list bullets drawn as images
  in place; rejoins a line that pdf2docx split into two blocks or two fake
  columns; and joins the wrapped lines of a right-to-left paragraph back into
  one paragraph (pdf2docx makes each line its own paragraph). Rows that mix
  Hebrew and English take the document's main direction, unless the line
  clearly hugs the left or right side of its column.

* fix_rtl_docx() edits the finished .docx: paragraphs with RTL letters get
  <w:bidi/> (right-to-left paragraph) and runs get <w:rtl/> plus complex-script
  font, size, bold and italic copied from their Latin settings.
"""

import copy
import logging
import re
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

_RTL = re.compile(r"[֐-׿؀-ۿݐ-ݿיִ-﷿ﹰ-﻿]")


# ── visual → logical ──────────────────────────────────────────────────────────

def _visual_to_logical_order(text: str, base: str = "R") -> list[int]:
    """
    Indexes of `text` (characters in left-to-right page order) in reading order.

    Running the bidi reordering on a visual string of a right-to-left paragraph
    gives back its logical order: right-to-left runs are reversed again, left-to-
    right runs (English, numbers) keep their internal order. Mirroring is not
    applied: PDFs store the logical bracket and draw the mirrored glyph, so the
    character codes are already right for reading order.
    """
    from bidi.algorithm import (get_empty_storage, get_embedding_levels, explicit_embed_and_overrides,
                                resolve_weak_types, resolve_neutral_types, resolve_implicit_levels,
                                reorder_resolved_levels, PARAGRAPH_LEVELS)
    storage = get_empty_storage()
    storage["base_level"] = PARAGRAPH_LEVELS[base]
    storage["base_dir"] = base
    get_embedding_levels(text, storage, False, False)
    for i, ch in enumerate(storage["chars"]):
        ch["_i"] = i
    explicit_embed_and_overrides(storage, False)
    resolve_weak_types(storage, False)
    resolve_neutral_types(storage, False)
    resolve_implicit_levels(storage, False)
    reorder_resolved_levels(storage, False)
    return [ch["_i"] for ch in storage["chars"]]


_LATIN = re.compile(r"[A-Za-z]")
_OBJ = "\ufffc"  # stands in for an inline image (list bullets are often drawn as images)


class _Placeholder:
    """A non-text item in a row (an image span) with a position, ordered like a character."""
    def __init__(self, bbox):
        self.c = _OBJ
        self.bbox = bbox


def rtl_majority(text: str) -> bool:
    """More Hebrew/Arabic letters than Latin ones: a right-to-left paragraph."""
    return len(_RTL.findall(text)) >= len(_LATIN.findall(text))


_NO_SPACE_NEXT_TO = set(".,:;!?()[]{}")  # glyph gaps next to punctuation are kerning, not spaces
_SWAP_CHARS = "()[]{}"
_SWAP = str.maketrans("()[]{}", ")(][}{")


def _balanced(text: str) -> bool:
    """Brackets open before they close and all close."""
    depth = {"(": 0, "[": 0, "{": 0}
    closes = {")": "(", "]": "[", "}": "{"}
    for c in text:
        if c in depth:
            depth[c] += 1
        elif c in closes:
            depth[closes[c]] -= 1
            if depth[closes[c]] < 0:
                return False
    return all(v == 0 for v in depth.values())


# ── pass 1: parsed layout ─────────────────────────────────────────────────────

def _rows(lines: list) -> list[list]:
    """Group a block's lines into visual rows (lines whose heights overlap)."""
    rows: list[list] = []
    for line in sorted(lines, key=lambda l: (l.bbox[1] + l.bbox[3]) / 2):
        y0, y1 = line.bbox[1], line.bbox[3]
        if rows:
            r0 = min(l.bbox[1] for l in rows[-1])
            r1 = max(l.bbox[3] for l in rows[-1])
            overlap = min(y1, r1) - max(y0, r0)
            if overlap > 0.5 * min(y1 - y0, r1 - r0):
                rows[-1].append(line)
                continue
        rows.append([line])
    return rows


def _fix_row(row: list, base_hint: str = "R", col: tuple | None = None) -> bool:
    """Rebuild one row in reading order. False when the row is left as it is."""
    from pdf2docx.text.TextSpan import TextSpan
    from pdf2docx.image.ImageSpan import ImageSpan

    items = []  # (x0, x1, char, span)
    for line in row:
        if not line.is_horizontal_text:
            return False
        for span in line.spans:
            if isinstance(span, ImageSpan):  # e.g. a list bullet drawn as an image
                items.append((span.bbox[0], span.bbox[2], _Placeholder(tuple(span.bbox)), span))
                continue
            if not isinstance(span, TextSpan) or not span.chars:
                return False  # text without positions: leave the row alone
            for ch in span.chars:
                if len(ch.c) == 1:
                    items.append((ch.bbox[0], ch.bbox[2], ch, span))
                elif ch.c:  # pdf2docx glues a trailing space onto a char ("d "): split it
                    w = (ch.bbox[2] - ch.bbox[0]) / len(ch.c)
                    for k, cp in enumerate(ch.c):
                        part = copy.copy(ch)
                        part.c = cp
                        x = ch.bbox[0] + k * w
                        items.append((x, x + w, part, span))
    text = "".join(c.c for _, _, c, _ in items)
    if not items:
        return False
    has_rtl = bool(_RTL.search(text))
    # Paragraph direction: the document's main direction, unless the row clearly
    # hugs one side of its column (a short right-aligned or left-aligned line).
    base = base_hint
    if col:
        x0 = min(t[0] for t in items); x1 = max(t[1] for t in items)
        w = max(col[1] - col[0], 1)
        left_gap, right_gap = x0 - col[0], col[1] - x1
        if right_gap < 0.05 * w and left_gap > 0.15 * w:
            base = "R"
        elif left_gap < 0.05 * w and right_gap > 0.15 * w:
            base = "L"

    items.sort(key=lambda t: (t[0] + t[1]) / 2)
    # a gap between pieces with no space character in it: add one
    sized = [s.size for _, _, _, s in items if getattr(s, "size", 0)]
    size = sorted(sized)[len(sized) // 2] if sized else 10
    visual = []
    prev_x1 = None
    for x0, x1, ch, span in items:
        if (prev_x1 is not None and x0 - prev_x1 > 0.2 * size
                and ch.c != " " and visual and visual[-1][0].c != " "
                and not isinstance(ch, _Placeholder) and not isinstance(visual[-1][0], _Placeholder)
                and ch.c not in _NO_SPACE_NEXT_TO and visual[-1][0].c not in _NO_SPACE_NEXT_TO):
            space = copy.copy(ch)
            space.c = " "
            visual.append((space, visual[-1][1]))
        visual.append((ch, span))
        prev_x1 = x1

    added = len(visual) - len(items)
    if not has_rtl and not added:
        return False  # left-to-right row with its spaces: nothing to do
    # a "character" can hold more than one code point (ligatures, marks): order by
    # code point, then map each back to its item, keeping each item once
    owner = [k for k, (c, _) in enumerate(visual) for _ in c.c]
    seen, logical = set(), []
    order = (_visual_to_logical_order("".join(c.c for c, _ in visual), base) if has_rtl
             else range(len(owner)))
    for i in order:
        k = owner[i]
        if k not in seen:
            seen.add(k)
            logical.append(visual[k])

    # consecutive characters from the same span become one span, in reading order
    # Some PDF producers store the bracket character as drawn (mirrored) rather
    # than as read, which leaves ")2026(" in reading order. If the brackets only
    # balance when swapped, swap them.
    text = "".join(c.c for c, _ in logical)
    if not _balanced(text) and _balanced(text.translate(_SWAP)):
        swapped = []
        for ch, span in logical:
            if ch.c in _SWAP_CHARS:
                ch = copy.copy(ch)
                ch.c = ch.c.translate(_SWAP)
            swapped.append((ch, span))
        logical = swapped

    # pieces split at a change of direction often both carry the space between them
    collapsed = []
    for ch, span in logical:
        if ch.c == " " and (not collapsed or collapsed[-1][0].c == " "):
            continue
        collapsed.append((ch, span))
    # lines of a wrapped paragraph are joined without a separator: end each row with a space
    if collapsed and collapsed[-1][0].c not in (" ", "-", "\u2013", _OBJ) and not isinstance(collapsed[-1][0], _Placeholder):
        sp = copy.copy(next(c for c, _ in reversed(collapsed) if not isinstance(c, _Placeholder)))
        sp.c = " "
        collapsed.append((sp, collapsed[-1][1]))
    new_spans = []
    for ch, span in collapsed:
        if isinstance(ch, _Placeholder):
            new_spans.append((span, None))  # the image span itself, unchanged
            continue
        if new_spans and new_spans[-1][0] is span and new_spans[-1][1] is not None:
            new_spans[-1][1].append(ch)
        else:
            new_spans.append((span, [ch]))
    rebuilt = []
    for span, chars in new_spans:
        if chars is None:
            rebuilt.append(span)
            continue
        s = copy.copy(span)
        s.chars = chars
        rebuilt.append(s)

    first = row[0]
    x0 = min(l.bbox[0] for l in row); y0 = min(l.bbox[1] for l in row)
    x1 = max(l.bbox[2] for l in row); y1 = max(l.bbox[3] for l in row)
    first.spans.reset(rebuilt)
    first.update_bbox((x0, y0, x1, y1))
    first.line_break = any(getattr(l, "line_break", 0) for l in row)
    first.tab_stop = 0
    for line in row[1:]:
        line.spans.reset([])
    return True


def _block_text(blk) -> str:
    return "".join(getattr(s, "text", "") or "" for l in blk.lines for s in l.spans)


def _join_same_row(blocks: list) -> list:
    """
    pdf2docx sometimes splits one visual line of right-to-left text into two
    blocks side by side (e.g. at a number or an English word). Put such pieces
    back into one block so the row can be reordered as a whole.
    """
    from pdf2docx.text.TextBlock import TextBlock
    out = []
    for blk in blocks:
        prev = out[-1] if out else None
        if (isinstance(blk, TextBlock) and isinstance(prev, TextBlock) and blk.lines and prev.lines
                and _RTL.search(_block_text(prev) + _block_text(blk))):
            y0, y1 = max(prev.bbox[1], blk.bbox[1]), min(prev.bbox[3], blk.bbox[3])
            h = min(prev.bbox[3] - prev.bbox[1], blk.bbox[3] - blk.bbox[1])
            gap = max(blk.bbox[0] - prev.bbox[2], prev.bbox[0] - blk.bbox[2])
            same_row = h > 0 and (y1 - y0) > 0.6 * h and len(_rows(list(prev.lines))) == 1 \
                and len(_rows(list(blk.lines))) == 1
            if same_row and gap < 2 * h:
                prev.lines.extend(list(blk.lines))
                prev.update_bbox((min(prev.bbox[0], blk.bbox[0]), min(prev.bbox[1], blk.bbox[1]),
                                  max(prev.bbox[2], blk.bbox[2]), max(prev.bbox[3], blk.bbox[3])))
                continue
        out.append(blk)
    return out


def _false_columns(section) -> bool:
    """
    pdf2docx splits a page into columns where it sees a vertical gap. In right-to-left
    text it often cuts a single line in two (e.g. after a number), giving two
    "columns" whose blocks sit on the same rows, side by side. Real columns have
    a wide gutter; these touch.
    """
    from pdf2docx.text.TextBlock import TextBlock
    cols = list(section)
    for a, b in zip(cols, cols[1:]):
        for x in a.blocks:
            for y in b.blocks:
                if not (isinstance(x, TextBlock) and isinstance(y, TextBlock)):
                    continue
                h = min(x.bbox[3] - x.bbox[1], y.bbox[3] - y.bbox[1])
                overlap = min(x.bbox[3], y.bbox[3]) - max(x.bbox[1], y.bbox[1])
                gap = max(y.bbox[0] - x.bbox[2], x.bbox[0] - y.bbox[2])
                if h > 0 and overlap > 0.6 * h and gap < 0.3 * h:
                    return True
    return False


def _merge_wrapped(blocks: list, col_left: float, col_right: float) -> list:
    """
    pdf2docx often turns each line of a right-to-left paragraph into its own block
    (it expects lines to share a left edge), so Word gets one paragraph per line.
    Join a block into the previous one when the previous line is full (reaches the
    column's left side, i.e. it wrapped), both are right-to-left, share a right
    edge, follow each other closely and use the same text size.
    """
    from pdf2docx.text.TextBlock import TextBlock
    out = []
    width = max(col_right - col_left, 1)
    for blk in blocks:
        prev = out[-1] if out else None
        if (isinstance(blk, TextBlock) and isinstance(prev, TextBlock) and prev.lines and blk.lines
                and rtl_majority(_block_text(prev)) and _RTL.search(_block_text(prev))
                and rtl_majority(_block_text(blk)) and _RTL.search(_block_text(blk))):
            last = prev.lines[-1]
            h = max(last.bbox[3] - last.bbox[1], 1)
            gap = blk.bbox[1] - prev.bbox[3]
            sizes = lambda b: [s.size for l in b.lines for s in l.spans if getattr(s, "size", 0)]
            sp, sb = sizes(prev), sizes(blk)
            same_size = sp and sb and abs(max(sp) - max(sb)) <= 0.15 * max(sp)
            if (abs(prev.bbox[2] - blk.bbox[2]) <= 3 and -0.5 * h <= gap <= 0.6 * h and same_size
                    and last.bbox[0] - col_left <= 0.12 * width):
                for line in prev.lines:
                    line.line_break = 0
                prev.lines.extend(list(blk.lines))
                prev.update_bbox((min(prev.bbox[0], blk.bbox[0]), prev.bbox[1],
                                  max(prev.bbox[2], blk.bbox[2]), blk.bbox[3]))
                prev.after_space = getattr(blk, "after_space", prev.after_space)
                from pdf2docx.common.share import TextAlignment
                prev.alignment = TextAlignment.JUSTIFY
                prev.first_line_space = 0
                continue
        out.append(blk)
    return out


def fix_rtl_layout(converter) -> int:
    """Put RTL rows of every parsed page in reading order, restore missing word
    spaces and rejoin wrapped right-to-left paragraphs. Returns rows changed."""
    from pdf2docx.text.TextBlock import TextBlock
    changed = 0

    # the document's main direction decides rows that mix Hebrew and English
    all_text = []
    for page in converter.pages:
        for section in getattr(page, "sections", []):
            for column in section:
                for blk in column.blocks:
                    if isinstance(blk, TextBlock):
                        all_text.append(_block_text(blk))
    joined = "".join(all_text)
    doc_base = "R" if _RTL.search(joined) and rtl_majority(joined) else "L"

    if doc_base == "R":
        for page in converter.pages:
            for section in getattr(page, "sections", []):
                try:
                    if len(section) > 1 and _false_columns(section):
                        cols = list(section)
                        first = cols[0]
                        blocks = [blk for c in cols for blk in c.blocks]
                        blocks.sort(key=lambda b: (round(b.bbox[1]), -b.bbox[2]))
                        first.blocks.reset(blocks)
                        x0 = min(c.bbox[0] for c in cols); x1 = max(c.bbox[2] for c in cols)
                        first.update_bbox((x0, first.bbox[1], x1, max(c.bbox[3] for c in cols)))
                        section.reset([first])
                except Exception as e:
                    logger.debug(f"column merge skipped: {e}")

    def walk_blocks(blocks, col=None):
        nonlocal changed
        for blk in blocks:
            if isinstance(blk, TextBlock):
                lines = list(blk.lines)
                for row in _rows(lines):
                    try:
                        changed += _fix_row(row, doc_base, col)
                    except Exception as e:  # never fail the conversion over one row
                        logger.debug(f"row left as is: {e}")
                blk.lines.reset([l for l in lines if len(l.spans)])
            elif getattr(blk, "is_table_block", False):  # tables: cells hold their own blocks
                for row in blk:
                    for cell in row:
                        inner = getattr(cell, "blocks", None)
                        if inner is not None:
                            walk_blocks(inner)

    for page in converter.pages:
        if not getattr(page, "finalized", True):
            continue
        for section in page.sections:
            for column in section:
                try:
                    joined_blocks = _join_same_row(list(column.blocks))
                    if len(joined_blocks) != len(column.blocks):
                        column.blocks.reset(joined_blocks)
                except Exception as e:
                    logger.debug(f"row join skipped: {e}")
                tb = [b for b in column.blocks if isinstance(b, TextBlock) and b.lines]
                col = (min(b.bbox[0] for b in tb), max(b.bbox[2] for b in tb)) if len(tb) > 1 else None
                walk_blocks(column.blocks, col)
                blocks = list(column.blocks)
                text_blocks = [b for b in blocks if isinstance(b, TextBlock) and b.lines]
                if len(text_blocks) > 1:
                    left = min(b.bbox[0] for b in text_blocks)
                    right = max(b.bbox[2] for b in text_blocks)
                    try:
                        merged = _merge_wrapped(blocks, left, right)
                        if len(merged) != len(blocks):
                            column.blocks.reset(merged)
                    except Exception as e:
                        logger.debug(f"paragraph merge skipped: {e}")
    return changed


# ── pass 2: the .docx ─────────────────────────────────────────────────────────

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def fix_rtl_docx(path: Path) -> int:
    """Mark RTL paragraphs and runs in a .docx written by pdf2docx. Returns paragraphs changed."""
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    doc = Document(str(path))
    changed = 0

    def paragraphs(container):
        for p in container.paragraphs:
            yield p
        for table in container.tables:
            for row in table.rows:
                for cell in row.cells:
                    yield from paragraphs(cell)

    def first_child(parent, tag):
        return parent.find(qn(tag))

    for p in paragraphs(doc):
        if not _RTL.search(p.text or ""):
            continue
        pPr = p._p.get_or_add_pPr()
        rtl_para = rtl_majority(p.text or "")  # an English paragraph quoting Hebrew stays LTR
        if rtl_para and first_child(pPr, "w:bidi") is None:
            bidi = OxmlElement("w:bidi")
            # w:bidi must come before w:spacing / w:ind / w:jc in pPr
            anchor = None
            for tag in ("w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind",
                        "w:contextualSpacing", "w:mirrorIndents", "w:suppressOverlap", "w:jc",
                        "w:textDirection", "w:textAlignment", "w:outlineLvl", "w:rPr", "w:sectPr"):
                anchor = first_child(pPr, tag)
                if anchor is not None:
                    break
            if anchor is not None:
                anchor.addprevious(bidi)
            else:
                pPr.append(bidi)
        # In a right-to-left paragraph Word reads jc left/right as start/end, so
        # pdf2docx's "right" (the page's right edge) is "left" here, and vice versa.
        jc = first_child(pPr, "w:jc")
        if rtl_para and jc is not None and jc.get(qn("w:val")) in ("left", "right"):
            jc.set(qn("w:val"), "left" if jc.get(qn("w:val")) == "right" else "right")
        ind = first_child(pPr, "w:ind")
        if rtl_para and ind is not None:  # indents are start/end too
            left, right = ind.get(qn("w:left")), ind.get(qn("w:right"))
            if left is not None or right is not None:
                ind.set(qn("w:left"), right or "0")
                ind.set(qn("w:right"), left or "0")

        for r in p.runs:
            rPr = r._r.get_or_add_rPr()
            if _RTL.search(r.text or "") and first_child(rPr, "w:rtl") is None:
                rtl = OxmlElement("w:rtl")  # schema order: before cs/em/lang/...
                after = next((first_child(rPr, t) for t in ("w:cs", "w:em", "w:lang", "w:eastAsianLayout",
                                                           "w:specVanish", "w:oMath")
                              if first_child(rPr, t) is not None), None)
                if after is not None:
                    after.addprevious(rtl)
                else:
                    rPr.append(rtl)
            fonts = first_child(rPr, "w:rFonts")
            if fonts is not None and fonts.get(qn("w:cs")) is None and fonts.get(qn("w:ascii")):
                fonts.set(qn("w:cs"), fonts.get(qn("w:ascii")))
            for latin, cs in (("w:sz", "w:szCs"), ("w:b", "w:bCs"), ("w:i", "w:iCs")):
                src = first_child(rPr, latin)
                if src is not None and first_child(rPr, cs) is None:
                    el = OxmlElement(cs)
                    if src.get(qn("w:val")) is not None:
                        el.set(qn("w:val"), src.get(qn("w:val")))
                    src.addnext(el)
        changed += 1

    if changed:
        doc.save(str(path))
    return changed
