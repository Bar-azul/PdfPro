"""
TranscribeService
=================
Speech to text for audio and video files (voice messages, recorded calls,
lectures, interviews) with Whisper large-v3-turbo on Cloudflare Workers AI,
the same account the translator uses. Output: plain text, a Word document
(right-to-left paragraphs for Hebrew/Arabic) or SRT subtitles.

ffmpeg turns any input into small mono 16 kHz MP3 pieces of about four
minutes, cut at pauses where possible so words aren't split, and the pieces
are transcribed a few at a time. Timestamps are shifted back onto the full
recording.
"""

import base64
import logging
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from ..config import settings
from . import progress
from ..services.pdf_service import _temp_file, _ms
from ..utils.errors import ApiError

logger = logging.getLogger(__name__)

MAX_MINUTES = 60
CHUNK_SECONDS = 240          # target piece length
CUT_WINDOW = 30              # look this far either side of a target for a pause
LANGS = {"auto": None, "he": "he", "en": "en", "ar": "ar", "ru": "ru", "fr": "fr", "es": "es"}
_RTL = re.compile(r"[֐-׿؀-ۿ]")


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _duration(path: Path) -> float:
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "default=noprint_wrappers=1:nokey=1", str(path)], timeout=60)
    try:
        return float(r.stdout.strip())
    except ValueError:
        raise ApiError(400, "audio_unreadable",
                       "This file has no audio we can read. Try MP3, M4A, WAV, OGG or a video file.")


def _pauses(path: Path) -> list[float]:
    """Midpoints of silences (pauses between sentences) in seconds."""
    r = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn",
              "-af", "silencedetect=noise=-32dB:d=0.35", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return [(s + e) / 2 for s, e in zip(starts, ends)]


def _cut_points(duration: float, pauses: list[float]) -> list[float]:
    """Piece boundaries: every ~4 minutes, moved to the nearest pause when one is close."""
    cuts, t = [0.0], 0.0
    while duration - t > CHUNK_SECONDS + CUT_WINDOW:
        target = t + CHUNK_SECONDS
        near = [p for p in pauses if abs(p - target) <= CUT_WINDOW and p > t + 60]
        t = min(near, key=lambda p: abs(p - target)) if near else target
        cuts.append(t)
    cuts.append(duration)
    return cuts


def _piece(src: Path, start: float, end: float) -> bytes:
    out = _temp_file("piece", ".mp3")
    try:
        r = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                  "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(src),
                  "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(out)])
        if r.returncode != 0 or not out.exists():
            raise ApiError(400, "audio_unreadable",
                           "This file has no audio we can read. Try MP3, M4A, WAV, OGG or a video file.")
        return out.read_bytes()
    finally:
        out.unlink(missing_ok=True)


def _fmt(t: float, sep: str = ",") -> str:
    t = max(0.0, t)
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


class TranscribeService:

    @staticmethod
    def transcribe(path: Path, language: str = "auto", output_format: str = "txt",
                   title: str = "Transcript") -> Path:
        t0 = time.time()
        if not (settings.CLOUDFLARE_ACCOUNT_ID and settings.CLOUDFLARE_API_TOKEN):
            raise ApiError(503, "transcribe_failed", "Transcription is not available right now.")
        lang = LANGS.get(language, None)

        progress.stage("converting")
        duration = _duration(path)
        if duration > MAX_MINUTES * 60 + 5:
            raise ApiError(400, "audio_too_long",
                           f"The recording is {duration / 60:.0f} minutes long. "
                           f"The limit is {MAX_MINUTES} minutes per file.")
        if duration < 0.5:
            raise ApiError(400, "transcribe_no_speech", "No speech was found in this recording.")
        cuts = _cut_points(duration, _pauses(path) if duration > CHUNK_SECONDS + CUT_WINDOW else [])
        spans = list(zip(cuts, cuts[1:]))

        url = (f"https://api.cloudflare.com/client/v4/accounts/{settings.CLOUDFLARE_ACCOUNT_ID}"
               f"/ai/run/{settings.CLOUDFLARE_WHISPER_MODEL}")
        headers = {"Authorization": f"Bearer {settings.CLOUDFLARE_API_TOKEN}"}
        done = 0
        progress.update(0, len(spans), "transcribing")

        def run(span):
            start, end = span
            body = {"audio": base64.b64encode(_piece(path, start, end)).decode(),
                    "task": "transcribe", "vad_filter": True,
                    "condition_on_previous_text": False}
            if lang:
                body["language"] = lang
            res, last = None, None
            for attempt in range(3):
                try:
                    r = httpx.post(url, headers=headers, json=body, timeout=180)
                except httpx.HTTPError as e:
                    last = str(e)
                    time.sleep(2 * (attempt + 1))
                    continue
                if r.status_code == 200:
                    res = r.json().get("result") or {}
                    break
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code < 500 and r.status_code != 429:
                    break
                time.sleep(2 * (attempt + 1))
            if res is None:
                logger.warning(f"Transcription failed for {start:.0f}-{end:.0f}s: {last}")
                raise ApiError(502, "transcribe_failed",
                               "The transcription service is not responding right now. "
                               "Please try again in a few minutes.")
            segs = []
            for s in res.get("segments") or []:
                text = (s.get("text") or "").strip()
                if text:
                    segs.append((start + float(s.get("start", 0)), start + float(s.get("end", 0)), text))
            if not segs and (res.get("text") or "").strip():
                segs.append((start, end, res["text"].strip()))
            return segs

        # progress is per request (a context variable), so report from this thread
        segments = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            for part in pool.map(run, spans):
                segments.extend(part)
                done += 1
                progress.update(done, len(spans), "transcribing")
        segments = _drop_repeats(segments)
        if not segments:
            raise ApiError(400, "transcribe_no_speech", "No speech was found in this recording.")

        progress.stage("saving")
        out = {"txt": _to_txt, "docx": _to_docx, "srt": _to_srt}.get(output_format, _to_txt)(segments, title)
        minutes = duration / 60
        logger.info(f"Transcribed {minutes:.1f} min ({len(spans)} pieces, lang={language}) in {_ms(t0)}ms")
        return out


def _drop_repeats(segments: list[tuple]) -> list[tuple]:
    """Whisper sometimes repeats a line over silence or noise; keep the first."""
    out = []
    for seg in segments:
        if len(out) >= 2 and seg[2] == out[-1][2] == out[-2][2]:
            continue
        out.append(seg)
    return out


def _paragraphs(segments: list[tuple], gap: float = 2.5, max_chars: int = 600) -> list[tuple]:
    """Join segments into paragraphs at pauses (or when a paragraph gets long)."""
    paras, cur = [], None
    for start, end, text in segments:
        if cur and (start - cur[1] > gap or len(cur[2]) > max_chars):
            paras.append(cur)
            cur = None
        cur = (cur[0], end, cur[2] + " " + text) if cur else (start, end, text)
    if cur:
        paras.append(cur)
    return paras


def _to_txt(segments, title) -> Path:
    out = _temp_file("transcript", ".txt")
    lines = [f"[{_fmt(s, '.')[:8]}] {t}" for s, _e, t in _paragraphs(segments)]
    out.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return out


def _to_srt(segments, title) -> Path:
    out = _temp_file("transcript", ".srt")
    blocks = [f"{i}\n{_fmt(s)} --> {_fmt(max(e, s + 0.5))}\n{t}"
              for i, (s, e, t) in enumerate(segments, 1)]
    out.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return out


def _to_docx(segments, title) -> Path:
    from docx import Document
    from docx.shared import Pt, RGBColor
    from .rtl_docx import fix_rtl_docx
    out = _temp_file("transcript", ".docx")
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(12)
    doc.add_heading(title, level=1)
    for start, _end, text in _paragraphs(segments):
        p = doc.add_paragraph()
        stamp = p.add_run(f"[{_fmt(start, '.')[:8]}]  ")
        stamp.font.size = Pt(9)
        stamp.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
        p.add_run(text)
    doc.save(str(out))
    if any(_RTL.search(t) for _s, _e, t in segments):
        fix_rtl_docx(out)
    return out
