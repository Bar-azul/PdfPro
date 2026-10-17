"""
Transcribe router — /api/transcribe/*
Audio or video recording → text, Word document or SRT subtitles.
"""

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from ..middleware.rate_limit import limiter
from ..models.schemas import FileResult
from .auth import get_optional_user
from ..services.storage_service import StorageService
from ..services.transcribe_service import LANGS, TranscribeService
from ..utils.errors import ApiError
from ..utils.file_utils import make_file_response

router = APIRouter()

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".oga", ".aac", ".flac", ".amr",
                    ".wma", ".webm", ".mp4", ".mov", ".mkv", ".3gp", ".m4v"}
MAX_MB = 100
OUTPUT_EXT = {"txt": ".txt", "docx": ".docx", "srt": ".srt"}


@router.post("/audio", response_model=FileResult, summary="Transcribe an audio or video recording")
@limiter.limit("5/hour")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form(default="auto"),
    output_format: str = Form(default="docx"),
    user: dict | None = Depends(get_optional_user),
):
    """Speech to text (Whisper). Up to 60 minutes per file."""
    t0 = time.time()
    if language not in LANGS:
        raise ApiError(400, "invalid_input", f"Unsupported language: {language}")
    if output_format not in OUTPUT_EXT:
        raise ApiError(400, "invalid_input", f"Unsupported output format: {output_format}")

    name = file.filename or "recording"
    ext = Path(name).suffix.lower()
    mime = (file.content_type or "").lower()
    # browsers label voice notes inconsistently (audio/ogg, audio/x-m4a, video/mp4,
    # application/octet-stream...), so the extension decides as much as the type
    if ext not in AUDIO_EXTENSIONS and not mime.startswith(("audio/", "video/")):
        raise ApiError(415, "audio_unsupported",
                       "Upload an audio or video file (MP3, M4A, WAV, OGG, OPUS, MP4...).")
    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise ApiError(413, "file_too_large", f"The file is larger than {MAX_MB}MB.")
    if not data:
        raise ApiError(400, "audio_unreadable", "The file is empty.")

    upload_meta = await StorageService.save_upload(data, name if ext else name + ".m4a")
    result_path = await asyncio.to_thread(
        TranscribeService.transcribe, upload_meta["path"], language, output_format, Path(name).stem,
    )
    out_name = f"{Path(name).stem}_transcript{OUTPUT_EXT[output_format]}"
    output_meta = await StorageService.save_output(result_path, out_name)
    return make_file_response(output_meta["file_id"], out_name, output_meta["download_url"],
                              output_meta["size"], t0)
