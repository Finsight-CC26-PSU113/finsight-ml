"""
OCR FinSight - Dependency Injection
Singleton model instances loaded once at startup.
"""

from typing import Optional
from app.config import ROOT_DIR, MAX_TEXT_LENGTH, FINETUNED_EASYOCR_PATH

_ocr_engine = None
_classifier = None
_extractor = None


async def startup_models():
    """Load all models at application startup."""
    global _ocr_engine, _classifier, _extractor

    from app.services.ocr import OCREngine
    from app.services.extractor import ReceiptExtractor
    from app.services.classifier import ReceiptLineClassifier, load_v2_weights

    print("=" * 50)
    print("OCR FinSight API — Starting up")
    print("=" * 50)

    print("[1/3] Loading EasyOCR...")
    if FINETUNED_EASYOCR_PATH.exists():
        _ocr_engine = OCREngine(model_path=str(FINETUNED_EASYOCR_PATH))
    else:
        _ocr_engine = OCREngine()
    if getattr(_ocr_engine, 'finetuned_loaded', False):
        print("      EasyOCR ready (fine-tuned: 62.89% exact match, 13.12% CER)")
    else:
        print("      EasyOCR ready (default model)")

    print("[2/3] Loading Classifier V2...")
    try:
        import tensorflow as tf
        v2_weights = ROOT_DIR / "models" / "classifier_v2" / "best_weights.weights.h5"
        model = ReceiptLineClassifier()
        dummy = (
            tf.zeros((1, MAX_TEXT_LENGTH), dtype=tf.int32),
            tf.zeros((1, 10), dtype=tf.float32),
            tf.zeros((1, 5), dtype=tf.float32),
        )
        model(dummy, training=False)
        load_v2_weights(model, str(v2_weights))
        model._is_v2 = True
        _classifier = model
        print("      Classifier V2 ready (12 categories, 82.31% accuracy)")
    except Exception as e:
        print(f"      V2 failed ({e}), trying V1 fallback...")
        try:
            from training.online_learning import OnlineLearningModel
            _classifier = OnlineLearningModel()
            _classifier._is_v2 = False
            print("      Classifier V1 (online learning) ready")
        except Exception as e2:
            print(f"      No classifier available: {e2}")
            _classifier = None

    print("[3/3] Loading Extractor...")
    _extractor = ReceiptExtractor()
    print("      Extractor ready")

    print("=" * 50)
    print("API ready at http://0.0.0.0:8000")
    print("Docs at   http://localhost:8000/api/docs")
    print("=" * 50)


def get_ocr():
    return _ocr_engine


def get_classifier():
    return _classifier


def get_extractor():
    return _extractor
