"""
OCR FinSight — Health Check Endpoint
GET /api/v1/health
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import get_classifier, get_extractor, get_ocr

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Return model load status and version info."""
    clf = get_classifier()
    version = getattr(clf, "_version", None) if clf else None
    is_context = getattr(clf, "_is_context_model", False) if clf else False
    ctx_window = getattr(clf, "_context_window", None) if clf else None

    return {
        "status": "ok",
        "models": {
            "ocr": get_ocr() is not None,
            "ocr_engine": "PaddleOCR",
            "classifier": clf is not None,
            "classifier_version": version,
            "classifier_context": f"±{ctx_window}" if is_context else None,
            "extractor": get_extractor() is not None,
        },
    }
