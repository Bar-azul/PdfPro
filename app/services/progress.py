"""
Real processing progress for the website's progress bar.

The site sends a random ?job_id= with each tool request and polls
GET /api/progress/{job_id} while it waits. Services report progress with
`progress.update(done, total, stage)` from inside their loops; the current
job comes from a context variable, so service signatures don't change
(asyncio.to_thread copies the context into the worker thread).

State lives in memory: the server runs a single worker process.
"""

import re
import threading
import time
from contextvars import ContextVar

_current: ContextVar[str | None] = ContextVar("progress_job", default=None)
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_TTL = 15 * 60
_VALID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def begin(job_id: str | None):
    """Start tracking a job for the current request; returns a reset token."""
    if not job_id or not _VALID.match(job_id):
        return None
    now = time.time()
    with _lock:
        for k in [k for k, v in _jobs.items() if now - v["at"] > _TTL]:
            del _jobs[k]
        _jobs[job_id] = {"stage": "processing", "done": 0, "total": 0, "state": "running", "at": now}
    return _current.set(job_id)


def end(token, ok: bool = True):
    job_id = _current.get()
    if job_id:
        with _lock:
            if job_id in _jobs:
                _jobs[job_id].update(state="done" if ok else "error", at=time.time())
    if token is not None:
        _current.reset(token)


def update(done: float, total: float | None = None, stage: str | None = None):
    """Report progress: `done` of `total` units (pages, images, files...)."""
    job_id = _current.get()
    if not job_id:
        return
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        if total is not None:
            job["total"] = total
        job["done"] = min(done, job["total"]) if job["total"] else done
        if stage:
            job["stage"] = stage
        job["at"] = time.time()


def stage(name: str):
    """A step without a count (e.g. 'saving'): the bar shows activity, not a percentage."""
    update(0, 0, name)


def get(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        pct = round(100 * job["done"] / job["total"]) if job["total"] else None
        return {"stage": job["stage"], "done": job["done"], "total": job["total"],
                "percent": pct, "state": job["state"]}
