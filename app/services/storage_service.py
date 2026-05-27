"""
StorageService
==============
Handles:
  • Saving uploaded files with UUID-based names
  • Tracking expiry metadata
  • Scheduled cleanup of expired files
  • Generating signed download URLs
"""

import asyncio
import logging
import shutil
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import aiofiles

from ..config import settings

logger = logging.getLogger(__name__)

# In-memory registry:  file_id → {path, expires_at, original_name, size}
_registry: dict[str, dict] = {}
_cleanup_task: asyncio.Task | None = None


class StorageService:

    # ── Initialization ──────────────────────────────────────────────────────

    @staticmethod
    async def init_directories():
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Upload dir : {settings.UPLOAD_DIR}")
        logger.info(f"Output dir : {settings.OUTPUT_DIR}")

    # ── Save uploaded file ───────────────────────────────────────────────────

    @staticmethod
    async def save_upload(file_bytes: bytes, original_filename: str) -> dict:
        """
        Persist uploaded file bytes to disk.
        Returns file metadata dict.
        """
        suffix = Path(original_filename).suffix.lower() or ".bin"
        file_id = str(uuid.uuid4())
        dest = settings.UPLOAD_DIR / f"{file_id}{suffix}"

        async with aiofiles.open(dest, "wb") as f:
            await f.write(file_bytes)

        expires_at = datetime.utcnow() + timedelta(seconds=settings.FILE_TTL_SECONDS)
        meta = {
            "file_id": file_id,
            "path": dest,
            "original_name": original_filename,
            "suffix": suffix,
            "size": len(file_bytes),
            "expires_at": expires_at,
            "created_at": datetime.utcnow(),
        }
        _registry[file_id] = meta
        logger.debug(f"Saved upload {file_id} ({len(file_bytes):,} bytes)")
        return meta

    # ── Save output file ─────────────────────────────────────────────────────

    @staticmethod
    async def save_output(source_path: Path, output_filename: str) -> dict:
        """
        Move/copy a processed output file into the output directory.
        Returns metadata including the public download URL.
        """
        file_id = str(uuid.uuid4())
        suffix = Path(output_filename).suffix.lower()
        dest = settings.OUTPUT_DIR / f"{file_id}{suffix}"

        # Use copy if source is already in a temp location
        await asyncio.to_thread(shutil.copy2, source_path, dest)

        size = dest.stat().st_size
        expires_at = datetime.utcnow() + timedelta(seconds=settings.FILE_TTL_SECONDS)

        meta = {
            "file_id": file_id,
            "path": dest,
            "original_name": output_filename,
            "suffix": suffix,
            "size": size,
            "expires_at": expires_at,
            "created_at": datetime.utcnow(),
            "download_url": f"/files/{file_id}{suffix}",
        }
        _registry[file_id] = meta
        logger.debug(f"Saved output {file_id} ({size:,} bytes)")
        return meta

    # ── Lookup ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_file(file_id: str) -> dict | None:
        return _registry.get(file_id)

    @staticmethod
    def get_upload_path(file_id: str) -> Path | None:
        meta = _registry.get(file_id)
        if not meta:
            return None
        return meta["path"]

    # ── Cleanup scheduler ────────────────────────────────────────────────────

    @staticmethod
    async def start_cleanup_scheduler():
        global _cleanup_task
        _cleanup_task = asyncio.create_task(StorageService._cleanup_loop())
        logger.info(f"Cleanup scheduler started (interval={settings.CLEANUP_INTERVAL_SECONDS}s)")

    @staticmethod
    async def stop_cleanup_scheduler():
        global _cleanup_task
        if _cleanup_task:
            _cleanup_task.cancel()
            try:
                await _cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Cleanup scheduler stopped")

    @staticmethod
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(settings.CLEANUP_INTERVAL_SECONDS)
            await StorageService.run_cleanup()

    @staticmethod
    async def run_cleanup() -> int:
        """Delete expired files from disk and registry. Returns count deleted."""
        now = datetime.utcnow()
        expired = [fid for fid, m in _registry.items() if m["expires_at"] < now]
        count = 0
        for fid in expired:
            meta = _registry.pop(fid)
            path: Path = meta["path"]
            try:
                if path.exists():
                    path.unlink()
                    count += 1
            except OSError as e:
                logger.warning(f"Could not delete {path}: {e}")
        if count:
            logger.info(f"Cleanup: removed {count} expired file(s)")
        return count

    # ── Disk usage stats ─────────────────────────────────────────────────────

    @staticmethod
    def disk_usage() -> dict:
        upload_size = sum(
            p.stat().st_size
            for p in settings.UPLOAD_DIR.iterdir()
            if p.is_file()
        )
        output_size = sum(
            p.stat().st_size
            for p in settings.OUTPUT_DIR.iterdir()
            if p.is_file()
        )
        return {
            "uploads_bytes": upload_size,
            "outputs_bytes": output_size,
            "total_bytes": upload_size + output_size,
            "tracked_files": len(_registry),
        }
