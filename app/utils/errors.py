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
