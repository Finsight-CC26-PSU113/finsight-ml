"""
OCR FinSight - Security Utilities
Password hashing, JWT token creation/verification.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Optional

from passlib.context import CryptContext

# ── Password hashing ──────────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# ── Simple HMAC token (placeholder for future JWT) ────────────────────────

_SECRET: str = secrets.token_hex(32)


def create_token(data: str, ttl: int = 3600) -> str:
    """Create a time-limited HMAC token."""
    expires = int(time.time()) + ttl
    payload = f"{data}.{expires}"
    sig = hmac.new(_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_token(token: str) -> Optional[str]:
    """Verify an HMAC token. Returns data if valid, None if expired/invalid."""
    try:
        parts = token.rsplit(".", 2)
        if len(parts) != 3:
            return None
        data, expires, sig = parts
        payload = f"{data}.{expires}"
        expected = hmac.new(_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        if int(expires) < time.time():
            return None
        return data
    except (ValueError, IndexError):
        return None
