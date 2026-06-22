"""
OCR FinSight — API v1 Router
Aggregates all v1 endpoint routers.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import health, scan

api_router = APIRouter()
api_router.include_router(health.router, prefix="", tags=["health"])
api_router.include_router(scan.router, prefix="", tags=["scan"])
