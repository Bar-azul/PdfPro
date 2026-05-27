"""
Auth router — /api/auth/*
"""

from datetime import datetime
from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from ..models.schemas import RegisterRequest, TokenResponse, UserResponse
from ..services.auth_service import AuthService
from ..config import settings

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Dependency — extracts and validates the JWT, returns user dict."""
    try:
        user_id = AuthService.verify_token(token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = AuthService.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="משתמש לא נמצא")
    return user


def get_optional_user(token: str | None = Depends(oauth2_scheme)) -> dict | None:
    """Dependency — returns user if authenticated, None otherwise."""
    if not token:
        return None
    try:
        user_id = AuthService.verify_token(token)
        return AuthService.get_user(user_id)
    except Exception:
        return None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest):
    """Create a new account."""
    try:
        user = AuthService.register(body.email, body.password, body.full_name)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    token = AuthService.create_token(user["id"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": settings.JWT_EXPIRE_MINUTES * 60,
        "plan": user["plan"],
    }


@router.post("/login", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """Login and receive a JWT access token."""
    try:
        user = AuthService.login(form.username, form.password)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = AuthService.create_token(user["id"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": settings.JWT_EXPIRE_MINUTES * 60,
        "plan": user["plan"],
    }


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "full_name": current_user["full_name"],
        "plan": current_user["plan"],
        "conversions_today": AuthService.get_usage_today(current_user["id"]),
        "created_at": current_user["created_at"],
    }


@router.post("/logout")
async def logout():
    """
    Client-side logout — simply discard the token.
    (Stateless JWT; add a token blocklist for server-side invalidation.)
    """
    return {"message": "התנתקת בהצלחה"}
