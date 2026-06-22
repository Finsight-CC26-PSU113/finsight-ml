"""
OCR FinSight — FastAPI Dependencies
Singleton model instances loaded once at startup, injected into endpoints.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from app.core.config import settings
from app.services.ocr import OCREngine
from app.services.classifier import ContextAwareClassifier, load_v5_classifier
from app.services.extractor import ReceiptExtractor

# ── Module-level singletons ────────────────────────────────────────────────

_ocr_engine: Optional[OCREngine] = None
_classifier: Optional[ContextAwareClassifier] = None
_extractor: Optional[ReceiptExtractor] = None


def startup_models() -> None:
    """Load all ML models at application startup."""
    global _ocr_engine, _classifier, _extractor

    print("=" * 55)
    print("  OCR FinSight — Loading models")
    print("=" * 55)

    # 1. OCR
    print("[startup] Loading PaddleOCR...")
    _ocr_engine = OCREngine()
    print("[startup] PaddleOCR ready")

    # 2. Classifier V5
    print("[startup] Loading Classifier V5 Indonesia...")
    try:
        _classifier = load_v5_classifier()
        print("[startup] Classifier V5 ready (86.26% acc)")
    except Exception as e:
        print(f"[startup] Classifier failed: {e}")
        _classifier = None

    # 3. Extractor
    _extractor = ReceiptExtractor()
    print("[startup] Extractor ready")
    print("=" * 55)


def shutdown_models() -> None:
    """Release model resources on shutdown."""
    global _ocr_engine, _classifier, _extractor
    _ocr_engine = None
    _classifier = None
    _extractor = None


# ── Dependency injection helpers ────────────────────────────────────────────


def get_ocr() -> OCREngine:
    if _ocr_engine is None:
        raise RuntimeError("OCR engine not loaded. Check startup event.")
    return _ocr_engine


def get_classifier() -> Optional[ContextAwareClassifier]:
    return _classifier


def get_extractor() -> ReceiptExtractor:
    if _extractor is None:
        raise RuntimeError("Extractor not loaded. Check startup event.")
    return _extractor
