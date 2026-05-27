"""
Payments router — /api/payments/*
Tranzila integration (Israeli payment gateway).

Setup:
1. הירשם בטרנזילה: tranzila.com
2. קבל את שם הספק (supplier) שלך
3. הוסף ל-.env את המשתנים
"""

import hashlib
import hmac
import logging
import uuid
from datetime import datetime, timedelta
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from ..config import settings
from .auth import get_current_user, get_optional_user
from ..services.auth_service import AuthService

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Tranzila URLs ─────────────────────────────────────────────────────────────
TRANZILA_IFRAME_URL = "https://direct.tranzila.com/{supplier}/iframenew.php"
TRANZILA_CHARGE_URL = "https://secure5.tranzila.com/cgi-bin/tranzila71u.cgi"

# ── Plans ─────────────────────────────────────────────────────────────────────
PLANS = {
    "pro": {
        "name":   "PDFPro — תכנית מקצועית",
        "amount": 39,
        "currency": 1,  # 1 = ILS
    },
    "enterprise": {
        "name":   "PDFPro — תכנית עסקית",
        "amount": 129,
        "currency": 1,
    },
}

# ── In-memory order + subscription store (replace with DB) ────────────────────
_orders: dict[str, dict] = {}         # order_id → {user_id, plan, status, token}
_subscriptions: dict[str, dict] = {}  # user_id  → {plan, token, next_charge, order_id}


# ── Request schemas ───────────────────────────────────────────────────────────

class CheckoutRequest(BaseModel):
    plan: str   # "pro" or "enterprise"


class CancelRequest(BaseModel):
    pass


# ── Public: get plans ─────────────────────────────────────────────────────────

@router.get("/plans", summary="תוכניות ומחירים")
async def get_plans():
    return {
        "plans": [
            {
                "id": "free",
                "name": "חינמי",
                "price_ils": 0,
                "features": ["5 המרות ביום", "קבצים עד 100MB", "כלים בסיסיים"],
            },
            {
                "id": "pro",
                "name": "מקצועי",
                "price_ils": 39,
                "features": [
                    "המרות ללא הגבלה", "קבצים עד 500MB",
                    "כל הכלים", "OCR ותרגום", "תמיכה מועדפת",
                ],
                "popular": True,
            },
            {
                "id": "enterprise",
                "name": "עסקי",
                "price_ils": 129,
                "features": [
                    "הכל ב-Pro", "עד 10 משתמשים",
                    "גישת API", "לוח ניהול", "SLA 99.9%",
                ],
            },
        ]
    }


# ── Create checkout URL ───────────────────────────────────────────────────────

@router.post("/create-checkout", summary="יצירת עמוד תשלום")
async def create_checkout(
    body: CheckoutRequest,
    user: dict | None = Depends(get_optional_user),
):
    """
    מחזיר URL לדף התשלום של טרנזילה.
    המשתמש מועבר לטרנזילה, משלם, ומוחזר לאתר.
    """
    plan = PLANS.get(body.plan)
    if not plan:
        raise HTTPException(400, detail=f"תוכנית לא חוקית: {body.plan}")

    if not settings.TRANZILA_SUPPLIER or settings.TRANZILA_SUPPLIER == "placeholder":
        raise HTTPException(503, detail="טרנזילה לא מוגדרת — הוסף TRANZILA_SUPPLIER ל-.env")

    order_id = str(uuid.uuid4())[:12].upper()

    # Save order
    _orders[order_id] = {
        "order_id":  order_id,
        "user_id":   user["id"] if user else None,
        "user_email": user["email"] if user else None,
        "plan":      body.plan,
        "amount":    plan["amount"],
        "status":    "pending",
        "created_at": datetime.utcnow().isoformat(),
    }

    # Build Tranzila payment URL
    base_url = TRANZILA_IFRAME_URL.format(supplier=settings.TRANZILA_SUPPLIER)

    params = {
        "sum":         plan["amount"],
        "currency":    plan["currency"],
        "cred_type":   1,                    # 1 = regular charge
        "description": plan["name"],
        "tranmode":    "AK",                 # Auth + Capture
        "txId":        order_id,
        "pdesc":       plan["name"],
        # Redirect URLs
        "success_url": f"{settings.BASE_URL}/api/payments/success?order_id={order_id}",
        "fail_url":    f"{settings.BASE_URL}/api/payments/fail?order_id={order_id}",
        "notify_url":  f"{settings.BASE_URL}/api/payments/notify",
        # Token for recurring (subscription)
        "store_card":  "1",                  # Save card token for monthly billing
    }

    checkout_url = f"{base_url}?{urlencode(params)}"
    logger.info(f"Tranzila checkout created: order={order_id} plan={body.plan}")

    return {
        "checkout_url": checkout_url,
        "order_id":     order_id,
    }


# ── Success redirect ──────────────────────────────────────────────────────────

@router.get("/success", include_in_schema=False)
async def payment_success(order_id: str, token: str = "", request: Request = None):
    """
    טרנזילה מפנה לכאן אחרי תשלום מוצלח.
    מפעילה את התוכנית ומפנה לדף הבית.
    """
    order = _orders.get(order_id)
    if not order:
        return RedirectResponse(f"{settings.FRONTEND_URL}/pricing?error=order_not_found")

    # Get token from query params (Tranzila sends it)
    params = dict(request.query_params) if request else {}
    card_token = params.get("token", params.get("TranzilaTK", token))
    auth_number = params.get("Num", params.get("authNr", ""))
    transaction_id = params.get("index", "")

    # Activate plan
    if order.get("user_id"):
        _activate_subscription(
            user_id=order["user_id"],
            plan=order["plan"],
            token=card_token,
            order_id=order_id,
            amount=order["amount"],
        )

    _orders[order_id]["status"]         = "completed"
    _orders[order_id]["token"]          = card_token
    _orders[order_id]["auth_number"]    = auth_number
    _orders[order_id]["transaction_id"] = transaction_id

    logger.info(f"Payment success: order={order_id} plan={order['plan']} auth={auth_number}")
    return RedirectResponse(f"{settings.FRONTEND_URL}/pricing?success=true&plan={order['plan']}")


# ── Fail redirect ─────────────────────────────────────────────────────────────

@router.get("/fail", include_in_schema=False)
async def payment_fail(order_id: str):
    """טרנזילה מפנה לכאן אחרי כישלון תשלום."""
    if order_id in _orders:
        _orders[order_id]["status"] = "failed"
    logger.warning(f"Payment failed: order={order_id}")
    return RedirectResponse(f"{settings.FRONTEND_URL}/pricing?cancelled=true")


# ── Server notification (webhook) ─────────────────────────────────────────────

@router.post("/notify", include_in_schema=False)
async def payment_notify(request: Request):
    """
    קבלת הודעה server-to-server מטרנזילה (אמינה יותר מה-redirect).
    """
    try:
        form = await request.form()
        params = dict(form)
    except Exception:
        body = await request.body()
        from urllib.parse import parse_qs
        params = {k: v[0] for k, v in parse_qs(body.decode()).items()}

    order_id    = params.get("txId", "")
    status_code = params.get("Response", "")      # "000" = success
    card_token  = params.get("TranzilaTK", "")
    auth_number = params.get("Num", "")

    logger.info(f"Tranzila notify: order={order_id} response={status_code}")

    if status_code == "000" and order_id in _orders:
        order = _orders[order_id]
        if order["status"] != "completed":  # Avoid double-activation
            if order.get("user_id"):
                _activate_subscription(
                    user_id=order["user_id"],
                    plan=order["plan"],
                    token=card_token,
                    order_id=order_id,
                    amount=order["amount"],
                )
            _orders[order_id]["status"]      = "completed"
            _orders[order_id]["token"]       = card_token
            _orders[order_id]["auth_number"] = auth_number

    return {"received": True}


# ── Monthly charge (subscription renewal) ─────────────────────────────────────

@router.post("/charge-renewals", include_in_schema=False)
async def charge_renewals():
    """
    Charge all subscriptions due today.
    Call this daily via a cron job or scheduler.

    Example cron (runs daily at 8:00 AM):
    0 8 * * * curl -X POST http://localhost:8000/api/payments/charge-renewals
    """
    today = datetime.utcnow().date()
    charged = []
    failed  = []

    for user_id, sub in list(_subscriptions.items()):
        next_charge = datetime.fromisoformat(sub["next_charge"]).date()
        if next_charge > today:
            continue

        plan   = PLANS.get(sub["plan"])
        token  = sub.get("token")

        if not plan or not token:
            continue

        # Charge using saved token
        success, auth = _charge_token(
            token=token,
            amount=plan["amount"],
            description=f"{plan['name']} — חידוש חודשי",
            order_id=f"REN-{user_id[:8]}-{today}",
        )

        if success:
            # Update next charge date (+1 month)
            _subscriptions[user_id]["next_charge"] = (
                datetime.utcnow() + timedelta(days=30)
            ).isoformat()
            charged.append(user_id)
            logger.info(f"Renewal charged: user={user_id} plan={sub['plan']} auth={auth}")
        else:
            failed.append(user_id)
            logger.warning(f"Renewal failed: user={user_id}")
            # In production: send email, retry tomorrow, downgrade after 3 failures

    return {"charged": len(charged), "failed": len(failed)}


# ── Cancel subscription ───────────────────────────────────────────────────────

@router.post("/cancel", summary="ביטול מנוי")
async def cancel_subscription(current_user: dict = Depends(get_current_user)):
    """ביטול מנוי — התוכנית תישאר פעילה עד סוף החודש."""
    user_id = current_user["id"]
    sub = _subscriptions.get(user_id)

    if not sub:
        raise HTTPException(400, detail="לא נמצא מנוי פעיל")

    _subscriptions[user_id]["cancelled"] = True
    _subscriptions[user_id]["cancel_date"] = datetime.utcnow().isoformat()

    # Downgrade at next billing date
    logger.info(f"Subscription cancelled: user={user_id}, active until {sub['next_charge']}")

    return {
        "message": "המנוי בוטל בהצלחה",
        "active_until": sub["next_charge"],
    }


# ── Subscription status ───────────────────────────────────────────────────────

@router.get("/status", summary="סטטוס מנוי")
async def subscription_status(current_user: dict = Depends(get_current_user)):
    sub = _subscriptions.get(current_user["id"])
    if not sub:
        return {"plan": "free", "active": False}
    return {
        "plan":        sub["plan"],
        "active":      not sub.get("cancelled", False),
        "next_charge": sub.get("next_charge"),
        "cancelled":   sub.get("cancelled", False),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _activate_subscription(
    user_id: str,
    plan: str,
    token: str,
    order_id: str,
    amount: int,
):
    """Upgrade user plan and register subscription for monthly billing."""
    AuthService.upgrade_plan(user_id, plan)
    _subscriptions[user_id] = {
        "plan":        plan,
        "token":       token,
        "order_id":    order_id,
        "amount":      amount,
        "start_date":  datetime.utcnow().isoformat(),
        "next_charge": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        "cancelled":   False,
    }
    logger.info(f"Subscription activated: user={user_id} plan={plan}")


def _charge_token(
    token: str,
    amount: int,
    description: str,
    order_id: str,
) -> tuple[bool, str]:
    """
    Charge a saved card token via Tranzila API.
    Returns (success: bool, auth_number: str)
    """
    params = {
        "supplier":    settings.TRANZILA_SUPPLIER,
        "TranzilaTK": token,
        "sum":         amount,
        "currency":    1,
        "cred_type":   1,
        "tranmode":    "AK",
        "description": description,
        "txId":        order_id,
        "notify_url":  f"{settings.BASE_URL}/api/payments/notify",
    }

    # Add terminal password if configured
    if settings.TRANZILA_TERMINAL_PASSWORD:
        params["TranzilaPW"] = settings.TRANZILA_TERMINAL_PASSWORD

    try:
        response = httpx.post(
            TRANZILA_CHARGE_URL,
            data=params,
            timeout=30,
        )
        result = dict(
            pair.split("=", 1)
            for pair in response.text.split("&")
            if "=" in pair
        )
        success = result.get("Response", "") == "000"
        auth    = result.get("Num", "")
        return success, auth
    except Exception as e:
        logger.error(f"Tranzila charge error: {e}")
        return False, ""