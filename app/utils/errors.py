"""
User-facing API errors.

Every error carries a stable machine ``code`` next to the English ``detail``,
so the website can show its own translated message (Hebrew/English) and fall
back to ``detail`` for codes it doesn't know yet.
"""

from fastapi import HTTPException


class ApiError(HTTPException):
    def __init__(self, status_code: int, code: str, detail: str):
        super().__init__(status_code=status_code, detail=detail)
        self.code = code


# Shared messages, so the same situation reads the same everywhere
INVALID_PDF = ("invalid_pdf", "This file is not a valid PDF or it is damaged.")
EMPTY_PDF = ("empty_pdf", "This PDF has no pages.")
ENCRYPTED_PDF = (
    "encrypted_pdf",
    "This PDF is password-protected. Remove the password first, then try again.",
)


def api_error(status_code: int, pair: tuple[str, str]) -> ApiError:
    code, detail = pair
    return ApiError(status_code, code, detail)


def is_daily_quota(status: int, body: str) -> bool:
    """Cloudflare Workers AI: the account's free daily neuron allowance is used up (code 3036)."""
    return status == 429 and ("3036" in body or "daily free allocation" in body.lower())


def daily_quota_error() -> ApiError:
    return ApiError(503, "daily_quota",
                    "Today's free allowance for this tool has been used up. Please try again tomorrow.")
