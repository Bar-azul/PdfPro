"""
Rate limiting middleware using slowapi (Starlette/FastAPI wrapper for limits).
"""

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ..config import settings

# Global limiter instance — imported by routers
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_FREE],
    storage_uri="memory://",
    config_filename=None,
)


def setup_rate_limiter(app: FastAPI) -> None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
