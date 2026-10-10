"""
Rate limiting middleware using slowapi (Starlette/FastAPI wrapper for limits).
"""

from fastapi import FastAPI, Request
import time

from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ..config import settings


def client_ip(request: Request) -> str:
    """
    The visitor's IP, not the proxy's. On Render every request arrives from an
    internal 10.x proxy address, so keying limits on request.client.host made
    all visitors behind the same proxy share one hourly quota. Proxies append
    to X-Forwarded-For, so the right-most public address is the one Render's
    edge saw; anything a client writes into the header itself sits to its left
    and is ignored.
    """
    import ipaddress
    forwarded = request.headers.get("x-forwarded-for", "")
    for part in reversed([p.strip() for p in forwarded.split(",") if p.strip()]):
        try:
            ip = ipaddress.ip_address(part)
        except ValueError:
            continue
        if ip.is_global:
            return str(ip)
    return get_remote_address(request)

# Global limiter instance — imported by routers
limiter = Limiter(
    key_func=client_ip,
    default_limits=[settings.RATE_LIMIT_FREE],
    storage_uri="memory://",
    config_filename=None,
)


def setup_rate_limiter(app: FastAPI) -> None:
    """Attach the shared limiter to the app and answer 429s with the reset time."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limited)


def rate_limited(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    429 with the seconds until this visitor's hourly quota resets, so the site
    can show a countdown instead of a bare "try later". The value is in the
    body (retry_after) as well as the Retry-After header, because a cross-site
    page can read the body but not that header without extra CORS setup.
    """
    retry_after = 3600
    try:
        item, args = request.state.view_rate_limit
        reset_at, _ = limiter.limiter.get_window_stats(item, *args)
        retry_after = max(1, int(reset_at - time.time()) + 1)
    except Exception:
        pass
    return JSONResponse(
        status_code=429,
        content={"code": "rate_limited", "retry_after": retry_after,
                 "detail": "You've reached the hourly limit for this tool. Please try again later."},
        headers={"Retry-After": str(retry_after)},
    )
