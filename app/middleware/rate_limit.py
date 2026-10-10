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


def _global_ip(value: str | None):
    import ipaddress
    try:
        ip = ipaddress.ip_address((value or "").strip())
    except ValueError:
        return None
    return str(ip) if ip.is_global else None


def client_ip_source(request: Request) -> tuple[str, str]:
    """
    The visitor's IP and which header it came from.

    Render sits behind Cloudflare, which sets CF-Connecting-IP / True-Client-IP
    to the address that connected to it and overwrites any value the client
    sent, so those come first. X-Forwarded-For is only a fallback: Render
    appends to it, and with Cloudflare in front its right-most entry can be a
    Cloudflare edge address that changes between requests (so a quota keyed on
    it never fills up). The direct peer (an internal 10.x proxy on Render) is
    the last resort.
    """
    for header in ("cf-connecting-ip", "true-client-ip"):
        ip = _global_ip(request.headers.get(header))
        if ip:
            return ip, header
    forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    for part in reversed(forwarded):
        ip = _global_ip(part)
        if ip:
            return ip, "x-forwarded-for"
    return get_remote_address(request), "peer"


def client_ip(request: Request) -> str:
    """Rate-limit key: the visitor's IP (see client_ip_source)."""
    return client_ip_source(request)[0]

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
