"""
Payments router — /api/payments/*
Stripe integration for Pro & Enterprise plan subscriptions.
"""

import logging
import stripe
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..config import settings
from .auth import get_current_user, get_optional_user
from ..services.auth_service import AuthService

logger = logging.getLogger(__name__)
router = APIRouter()

stripe.api_key = settings.STRIPE_SECRET_KEY

# ── Price IDs — set these in your Stripe dashboard ───────────────────────────
# After creating products in Stripe, paste the Price IDs here:
PLANS = {
    "pro": {
        "price_id": settings.STRIPE_PRICE_PRO,
        "name": "PDFPro — Pro Plan",
        "amount": 3900,  # ₪39 in agorot
    },
    "enterprise": {
        "price_id": settings.STRIPE_PRICE_ENTERPRISE,
        "name": "PDFPro — Enterprise Plan",
        "amount": 12900,  # ₪129 in agorot
    },
}


# ── Request / Response schemas ────────────────────────────────────────────────

class CheckoutRequest(BaseModel):
    plan: str  # "pro" or "enterprise"
    success_url: str = "http://localhost:8000/pricing?success=true"
    cancel_url: str  = "http://localhost:8000/pricing?cancelled=true"


# ── Create Checkout Session ───────────────────────────────────────────────────

@router.post("/create-checkout-session", summary="Create Stripe Checkout session")
async def create_checkout_session(
    body: CheckoutRequest,
    user: dict | None = Depends(get_optional_user),
):
    """
    Creates a Stripe Checkout session for the selected plan.
    Returns a URL to redirect the user to Stripe's hosted checkout page.
    """
    plan_info = PLANS.get(body.plan)
    if not plan_info:
        raise HTTPException(400, detail=f"תוכנית לא חוקית: {body.plan}")

    if not settings.STRIPE_SECRET_KEY or settings.STRIPE_SECRET_KEY == "sk_test_placeholder":
        raise HTTPException(503, detail="Stripe לא מוגדר — הוסף STRIPE_SECRET_KEY ל-.env")

    try:
        # Build metadata to identify the user after payment
        metadata = {"plan": body.plan}
        if user:
            metadata["user_id"] = user["id"]
            metadata["user_email"] = user["email"]

        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": plan_info["price_id"], "quantity": 1}],
            success_url=body.success_url + "&session_id={CHECKOUT_SESSION_ID}",
            cancel_url=body.cancel_url,
            metadata=metadata,
            customer_email=user["email"] if user else None,
            billing_address_collection="auto",
            allow_promotion_codes=True,
        )

        logger.info(f"Checkout session created: {session.id} for plan={body.plan}")
        return {"checkout_url": session.url, "session_id": session.id}

    except stripe.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(400, detail=f"שגיאת Stripe: {str(e)}")


# ── Webhook ───────────────────────────────────────────────────────────────────

@router.post("/webhook", summary="Stripe webhook handler", include_in_schema=False)
async def stripe_webhook(request: Request):
    """
    Receives events from Stripe and updates user plans accordingly.
    Configure this URL in your Stripe dashboard → Webhooks.
    Events handled: checkout.session.completed, customer.subscription.deleted
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    if not settings.STRIPE_WEBHOOK_SECRET or settings.STRIPE_WEBHOOK_SECRET == "whsec_placeholder":
        logger.warning("Webhook secret not set — skipping signature verification")
        try:
            event = stripe.Event.construct_from(
                stripe.util.convert_to_stripe_object(
                    __import__("json").loads(payload)
                ), stripe.api_key
            )
        except Exception as e:
            raise HTTPException(400, detail=str(e))
    else:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
            )
        except stripe.SignatureVerificationError:
            logger.warning("Invalid webhook signature")
            raise HTTPException(400, detail="Invalid signature")

    # ── Handle events ─────────────────────────────────────────────────────────

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        metadata = session.get("metadata", {})
        user_id = metadata.get("user_id")
        plan = metadata.get("plan", "pro")

        if user_id:
            try:
                AuthService.upgrade_plan(user_id, plan)
                # Save Stripe customer/subscription IDs for future use
                _save_stripe_ids(
                    user_id,
                    customer_id=session.get("customer"),
                    subscription_id=session.get("subscription"),
                )
                logger.info(f"User {user_id} upgraded to {plan}")
            except Exception as e:
                logger.error(f"Failed to upgrade user {user_id}: {e}")
        else:
            # Guest checkout — store for when they register
            logger.info(f"Guest checkout completed for plan={plan}, email={session.get('customer_email')}")

    elif event["type"] == "customer.subscription.deleted":
        # Subscription cancelled — downgrade to free
        subscription = event["data"]["object"]
        customer_id = subscription.get("customer")
        user_id = _find_user_by_customer_id(customer_id)
        if user_id:
            AuthService.upgrade_plan(user_id, "free")
            logger.info(f"User {user_id} downgraded to free (subscription cancelled)")

    elif event["type"] == "invoice.payment_failed":
        # Payment failed — could notify user
        invoice = event["data"]["object"]
        logger.warning(f"Payment failed for customer {invoice.get('customer')}")

    return {"received": True}


# ── Customer Portal ───────────────────────────────────────────────────────────

@router.post("/portal", summary="Open Stripe Customer Portal")
async def create_portal_session(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """
    Creates a Stripe Customer Portal session so the user can manage
    their subscription, update payment method, or cancel.
    """
    customer_id = _get_customer_id(current_user["id"])
    if not customer_id:
        raise HTTPException(400, detail="לא נמצא מנוי פעיל עבור משתמש זה")

    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url="http://localhost:8000/pricing",
        )
        return {"portal_url": session.url}
    except stripe.StripeError as e:
        raise HTTPException(400, detail=str(e))


# ── Verify session (called after redirect from Stripe) ────────────────────────

@router.get("/verify-session", summary="Verify completed checkout session")
async def verify_session(session_id: str):
    """
    Called after Stripe redirects back with ?session_id=xxx
    Returns the plan that was purchased.
    """
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        if session.payment_status == "paid":
            return {
                "status": "paid",
                "plan": session.metadata.get("plan", "pro"),
                "email": session.customer_email,
            }
        return {"status": session.payment_status}
    except stripe.StripeError as e:
        raise HTTPException(400, detail=str(e))


# ── Plan info (public) ────────────────────────────────────────────────────────

@router.get("/plans", summary="Get available plans and pricing")
async def get_plans():
    return {
        "plans": [
            {
                "id": "free",
                "name": "Free",
                "price_ils": 0,
                "features": ["5 conversions/day", "Files up to 25MB", "Basic tools"],
            },
            {
                "id": "pro",
                "name": "Pro",
                "price_ils": 39,
                "price_id": settings.STRIPE_PRICE_PRO,
                "features": ["Unlimited conversions", "Files up to 500MB", "All tools", "OCR & Translation", "Priority support"],
                "popular": True,
            },
            {
                "id": "enterprise",
                "name": "Enterprise",
                "price_ils": 129,
                "price_id": settings.STRIPE_PRICE_ENTERPRISE,
                "features": ["Everything in Pro", "Up to 10 users", "API access", "Admin dashboard", "SLA 99.9%"],
            },
        ]
    }


# ── In-memory Stripe ID store (replace with DB in production) ─────────────────
_stripe_ids: dict[str, dict] = {}  # user_id → {customer_id, subscription_id}

def _save_stripe_ids(user_id: str, customer_id: str, subscription_id: str):
    _stripe_ids[user_id] = {"customer_id": customer_id, "subscription_id": subscription_id}

def _get_customer_id(user_id: str) -> str | None:
    return _stripe_ids.get(user_id, {}).get("customer_id")

def _find_user_by_customer_id(customer_id: str) -> str | None:
    for uid, ids in _stripe_ids.items():
        if ids.get("customer_id") == customer_id:
            return uid
    return None
