"""
PDFPro Backend — FastAPI Application
=====================================
Entry point for the PDF processing API.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .routers import convert, organize, edit, ocr, translate, auth
from .services.storage_service import StorageService
from .middleware.rate_limit import setup_rate_limiter
from .config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pdfpro")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("🚀 PDFPro API starting up...")
    await StorageService.init_directories()
    await StorageService.start_cleanup_scheduler()
    logger.info("✅ Storage service ready")
    yield
    logger.info("🛑 PDFPro API shutting down...")
    await StorageService.stop_cleanup_scheduler()


app = FastAPI(
    title="PDFPro API",
    description="Professional PDF processing API — convert, edit, merge, split, OCR, translate.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Progress (?job_id= on tool requests, polled by the site) ───────────────────
from .services import progress as _progress


@app.middleware("http")
async def progress_middleware(request, call_next):
    token = _progress.begin(request.query_params.get("job_id")) if request.method == "POST" else None
    ok = False
    try:
        response = await call_next(request)
        ok = response.status_code < 400
        return response
    finally:
        if token is not None:
            _progress.end(token, ok)


@app.get("/api/client-ip", tags=["Health"], include_in_schema=False)
async def get_client_ip(request: Request):
    """The address hourly limits are counted against, and where it came from (for checking the setup)."""
    from .middleware.rate_limit import client_ip_source
    ip, source = client_ip_source(request)
    return {"ip": ip, "source": source}


@app.get("/api/progress/{job_id}", tags=["Health"], include_in_schema=False)
async def get_progress(job_id: str):
    info = _progress.get(job_id)
    if info is None:
        return JSONResponse(status_code=404, content={"detail": "unknown job", "code": "unknown_job"})
    return info


# ── Rate limiter ───────────────────────────────────────────────────────────────
setup_rate_limiter(app)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router,      prefix="/api/auth",      tags=["Auth"])
app.include_router(convert.router,   prefix="/api/convert",   tags=["Convert"])
app.include_router(organize.router,  prefix="/api/organize",  tags=["Organize"])
app.include_router(edit.router,      prefix="/api/edit",      tags=["Edit"])
app.include_router(ocr.router,       prefix="/api/ocr",       tags=["OCR"])
app.include_router(translate.router, prefix="/api/translate", tags=["Translate"])

# ── Static output files ───────────────────────────────────────────────────────
app.mount("/files", StaticFiles(directory=settings.OUTPUT_DIR), name="files")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Error responses ───────────────────────────────────────────────────────────
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """{"detail": "...", "code": "..."} — code lets the site show a translated message."""
    code = getattr(exc, "code", None) or f"http_{exc.status_code}"
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": code},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Turn FastAPI's 422 list into one readable sentence instead of raw JSON."""
    problems = []
    for err in exc.errors():
        field = next((str(p) for p in reversed(err.get("loc", ())) if not isinstance(p, int)), "input")
        if field in ("body", "query", "form"):
            field = "input"
        problems.append(f"{field}: {err.get('msg', 'invalid value')}")
    detail = "Invalid request — " + "; ".join(problems) if problems else "Invalid request"
    return JSONResponse(status_code=422, content={"detail": detail, "code": "invalid_input"})


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later.", "code": "server_error"},
    )

from fastapi.responses import HTMLResponse
from pathlib import Path

FRONTEND_FILE = Path(__file__).parent.parent / "frontend.html"

def _read_html():
    return FRONTEND_FILE.read_text(encoding="utf-8") if FRONTEND_FILE.exists() else "<h1>Not found</h1>"

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_en():
    return HTMLResponse(content=_read_html())

@app.get("/he", response_class=HTMLResponse, include_in_schema=False)
async def serve_he():
    return HTMLResponse(content=_read_html())

from app.routers import payments
app.include_router(payments.router, prefix="/api/payments", tags=["Payments"])