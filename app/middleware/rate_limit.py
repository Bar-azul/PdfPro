"""
Rate limiting middleware using slowapi (Starlette/FastAPI wrapper for limits).
"""

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
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
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
