"""
OCR FinSight - FastAPI Backend (Production-ready)
==================================================
Single endpoint pipeline: OCR -> Classifier -> Extractor.
Mengembalikan JSON yang sudah ter-label per baris (bukan per kata) +
data terstruktur hasil extractor.

Endpoints:
    GET  /                  -> Halaman web (index_v2.html)
    GET  /api/health        -> Health check (status semua model)
    POST /api/predict       -> Pipeline lengkap, return JSON terlabel
    POST /api/scan          -> Alias /api/predict (kompatibilitas)
    GET  /api/docs          -> Swagger UI
"""

import sys
import re
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ROOT_DIR, LINE_CLASSES, MAX_TEXT_LENGTH, CONTEXT_WINDOW, OCR_GPU
from src.preprocessing import preprocess_for_easyocr
from src.ocr_engine_paddle import PaddleOCREngine as OCREngine
OCR_ENGINE_TYPE = "PaddleOCR"
from src.extractor import ReceiptExtractor
from src.model_v4_context import ContextAwareClassifier

# ============================================================
# App setup
# ============================================================

app = FastAPI(
    title="OCR FinSight API",
    description="Pipeline OCR + Classifier + Extractor dalam satu endpoint.",
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates + static
WEB_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates_v2"))
if (WEB_DIR / "static_v2").exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static_v2")), name="static")

# ============================================================
# Global model instances (loaded once at startup)
# ============================================================

ocr_engine: Optional[OCREngine] = None
classifier = None
extractor: Optional[ReceiptExtractor] = None


@app.on_event("startup")
async def startup():
    """Load all models at startup (not per-request)."""
    global ocr_engine, classifier, extractor

    print("=" * 55)
    print("🧾 OCR FinSight API v2.1 — Starting up")
    print("=" * 55)

    # 1. OCR Engine (PaddleOCR only)
    print(f"📦 Loading PaddleOCR...")
    ocr_engine = OCREngine(lang='en', gpu=OCR_GPU)  # 'en' works for Indonesia receipts too
    print("✅ PaddleOCR ready")

    # 2. Classifier (try V4 context > V3 > V2, fallback to V1)
    print("📦 Loading Classifier...")
    try:
        import tensorflow as tf
        from src.model import ReceiptLineClassifier

        # Try models in priority: V5 (Indonesia, ±4 context) > V4 (±2 context) > V3 > V2
        model_versions = [
            (ROOT_DIR / "models" / "classifier_v5_indonesia" / "best_weights.weights.h5", "V5", "Indonesia-only ±4 context (86.26% acc)", True, 4),
            (ROOT_DIR / "models" / "classifier_v4_context" / "best_weights.weights.h5", "V4", "context-aware ±2", True, 2),
            (ROOT_DIR / "models" / "classifier_v3" / "best_weights.weights.h5", "V3", "improved training config", False, None),
            (ROOT_DIR / "models" / "classifier_v2" / "best_weights.weights.h5", "V2", "baseline 12 categories", False, None),
        ]
        
        loaded = False
        for weights_path, version, desc, is_context_model, context_window in model_versions:
            if weights_path.exists():
                try:
                    print(f"   Trying Classifier {version} ({desc})...")
                    
                    if is_context_model:
                        # V4/V5 use context-aware architecture (5 inputs)
                        model = ContextAwareClassifier(context_window=context_window)
                        
                        dummy_chars = tf.zeros((1, MAX_TEXT_LENGTH), dtype=tf.int32)
                        dummy_tf = tf.zeros((1, 10), dtype=tf.float32)
                        dummy_pf = tf.zeros((1, 5), dtype=tf.float32)
                        dummy_context_chars = tf.zeros((1, context_window * 2 * 20), dtype=tf.int32)
                        dummy_context_text = tf.zeros((1, context_window * 2 * 5), dtype=tf.float32)
                        _ = model((dummy_chars, dummy_tf, dummy_pf, dummy_context_chars, dummy_context_text), training=False)
                    else:
                        # V2/V3 use standard architecture (3 inputs)
                        model = ReceiptLineClassifier()
                        
                        dummy_chars = tf.zeros((1, MAX_TEXT_LENGTH), dtype=tf.int32)
                        dummy_tf = tf.zeros((1, 10), dtype=tf.float32)
                        dummy_pf = tf.zeros((1, 5), dtype=tf.float32)
                        _ = model((dummy_chars, dummy_tf, dummy_pf), training=False)

                    model.load_weights(str(weights_path))
                    model._is_v2 = True
                    model._version = version
                    model._is_context_model = is_context_model
                    model._context_window = context_window if is_context_model else None
                    classifier = model
                    print(f"✅ Classifier {version} loaded - {desc}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"   ⚠️ {version} failed: {e}")
                    continue
        
        if not loaded:
            raise Exception("No V2+ model available")
            
    except Exception as e:
        print(f"⚠️  All V2+ models failed: {e}")
        try:
            from src.online_learning import OnlineLearningModel
            classifier = OnlineLearningModel()
            classifier._is_v2 = False
            classifier._version = "V1"
            print("✅ Classifier V1 loaded (fallback)")
        except Exception as e2:
            print(f"❌ No classifier available: {e2}")
            classifier = None

    # 3. Extractor
    extractor = ReceiptExtractor()
    print("✅ Extractor ready")
    
    # Summary
    print("\n📊 Model Summary:")
    print(f"   OCR: PaddleOCR (2-3x faster, state-of-the-art accuracy)")
    
    # Classifier summary
    classifier_version = getattr(classifier, '_version', 'V1') if classifier else 'None'
    is_context = getattr(classifier, '_is_context_model', False)
    context_window = getattr(classifier, '_context_window', None)
    
    if classifier_version == "V5":
        print(f"   Classifier: V5 (Indonesia-only, ±{context_window} context, 86.26% acc)")
    elif classifier_version == "V4":
        print(f"   Classifier: V4 (±{context_window} context)")
    elif classifier and getattr(classifier, '_is_v2', False):
        print(f"   Classifier: {classifier_version} (12 categories)")
    else:
        print(f"   Classifier: {classifier_version} (fallback)")
    
    print(f"   Extractor: Enhanced with smart validation")
    
    print("\n🌐 Web UI    : http://localhost:8000/")
    print("📖 Swagger   : http://localhost:8000/api/docs")
    print("🩺 Health    : http://localhost:8000/api/health")
    print("=" * 55)


# ============================================================
# Feature extraction for classifier V2
# ============================================================

DATE_REGEX = re.compile(
    r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})|(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})|'
    r'(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})|'
    r'((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}[\s,]+\d{2,4})',
    re.IGNORECASE
)
TIME_REGEX = re.compile(r'\d{1,2}:\d{2}(:\d{2})?')
QTY_REGEX = re.compile(r'\d+\s*[xX\*@]\s*[\d.,]+|@\s*[\d.,]+')
CURRENCY_REGEX = re.compile(r'(rm|rp|idr|usd|sgd|sr|\$|€)\s*[\d.,]+', re.IGNORECASE)


def _text_feats(text: str) -> list[float]:
    text = str(text) if text else ""
    n = len(text)
    if n == 0:
        return [0.0] * 10
    digits = sum(c.isdigit() for c in text)
    alphas = sum(c.isalpha() for c in text)
    uppers = sum(c.isupper() for c in text)
    spaces = sum(c.isspace() for c in text)
    specials = n - digits - alphas - spaces
    return [
        digits / n, alphas / n, uppers / max(alphas, 1),
        spaces / n, specials / n, n / 100.0,
        len(text.split()) / 20.0,
        1.0 if CURRENCY_REGEX.search(text) else 0.0,
        1.0 if (DATE_REGEX.search(text) or TIME_REGEX.search(text)) else 0.0,
        1.0 if QTY_REGEX.search(text) else 0.0,
    ]


def _char_ids(text: str, max_len: int = MAX_TEXT_LENGTH) -> list[int]:
    text = str(text) if text else ""
    ids = [min(ord(c), 127) for c in text[:max_len]]
    return ids + [0] * (max_len - len(ids))


def _generate_context_features(lines: list[dict], context_window: int = CONTEXT_WINDOW) -> tuple[list, list]:
    """Generate context features from neighboring lines for V4 context model.
    
    Returns:
        (context_chars_list, context_text_list)
    """
    n_lines = len(lines)
    context_chars_list = []
    context_text_list = []
    
    for i in range(n_lines):
        context_chars = []
        context_text = []
        
        # Look at context_window lines before and after
        for offset in range(-context_window, context_window + 1):
            if offset == 0:
                continue  # Skip current line
            
            ctx_idx = i + offset
            if 0 <= ctx_idx < n_lines:
                ctx_text = lines[ctx_idx].get('text', '')
                context_chars.extend(_char_ids(ctx_text, max_len=20))  # First 20 chars
                context_text.extend(_text_feats(ctx_text)[:5])  # First 5 features
            else:
                # Padding for out-of-bounds
                context_chars.extend([0] * 20)
                context_text.extend([0.0] * 5)
        
        context_chars_list.append(context_chars)
        context_text_list.append(context_text)
    
    return context_chars_list, context_text_list


def _predict_with_v2(model, lines: list[dict]) -> tuple[list[str], list[float]]:
    """Predict labels using classifier V2/V3 (batch inference)."""
    n = len(lines)
    chars = np.array([_char_ids(l.get('text', '')) for l in lines], dtype=np.int32)
    tfeats = np.array([_text_feats(l.get('text', '')) for l in lines], dtype=np.float32)
    pfeats = np.array([
        [float(l.get('y_center', 0.5)), float(l.get('x_center', 0.5)),
         float(l.get('width', 0.1)), float(l.get('height', 0.05)),
         float(i) / max(n, 1)]
        for i, l in enumerate(lines)
    ], dtype=np.float32)

    preds = model((chars, tfeats, pfeats), training=False).numpy()
    pred_idx = np.argmax(preds, axis=1)
    confs = np.max(preds, axis=1)
    labels = [LINE_CLASSES[int(idx)] for idx in pred_idx]
    return labels, [float(c) for c in confs]


def _predict_with_v4_context(model, lines: list[dict]) -> tuple[list[str], list[float]]:
    """Predict labels using classifier V4/V5 context (batch inference with context)."""
    # Get context window from model (V4=2, V5=4)
    context_window = getattr(model, '_context_window', 2)
    
    n = len(lines)
    chars = np.array([_char_ids(l.get('text', '')) for l in lines], dtype=np.int32)
    tfeats = np.array([_text_feats(l.get('text', '')) for l in lines], dtype=np.float32)
    pfeats = np.array([
        [float(l.get('y_center', 0.5)), float(l.get('x_center', 0.5)),
         float(l.get('width', 0.1)), float(l.get('height', 0.05)),
         float(i) / max(n, 1)]
        for i, l in enumerate(lines)
    ], dtype=np.float32)
    
    # Generate context features with the correct window
    context_chars_list, context_text_list = _generate_context_features(lines, context_window)
    context_chars = np.array(context_chars_list, dtype=np.int32)
    context_text = np.array(context_text_list, dtype=np.float32)

    preds = model((chars, tfeats, pfeats, context_chars, context_text), training=False).numpy()
    pred_idx = np.argmax(preds, axis=1)
    confs = np.max(preds, axis=1)
    labels = [LINE_CLASSES[int(idx)] for idx in pred_idx]
    return labels, [float(c) for c in confs]


# ============================================================
# Pipeline core
# ============================================================

async def _run_pipeline(image_bytes: bytes) -> dict:
    """Single-process pipeline: decode -> deskew -> OCR -> classify -> extract.

    Returns dict shape (TANPA annotated image — JSON terstruktur saja):
        {
          "success": bool,
          "extracted": {store, date, items[], total, total_items, totals{}, address},
          "lines": [{idx, text, label, confidence, ocr_confidence, bbox}],
          "stats": {total_lines, avg_confidence, processing_time_ms, image_size}
        }
    """
    if not ocr_engine:
        raise HTTPException(status_code=503, detail="OCR engine not ready")

    import cv2

    # 1. Decode
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image. Use JPG/PNG/WEBP.")

    # 2. Preprocess (deskew + CLAHE + bilateral + sharpen + upscale 2x)
    img = preprocess_for_easyocr(img, upscale_factor=2.0)

    # 3. OCR -> per-line (bukan per-kata)
    lines, img_info = ocr_engine.read_receipt_with_image_info(img)
    if not lines:
        raise HTTPException(status_code=422, detail="No text detected in image.")

    # 4. Classify -> tempel label ke setiap baris
    classified_lines = []
    if classifier:
        try:
            is_v2 = getattr(classifier, '_is_v2', False)
            is_context_model = getattr(classifier, '_is_context_model', False)
            
            if is_context_model:
                labels, confs = _predict_with_v4_context(classifier, lines)
            elif is_v2:
                labels, confs = _predict_with_v2(classifier, lines)
            else:
                labels, confs = classifier.predict(lines)
                
            for line, label, conf in zip(lines, labels, confs):
                lc = dict(line)
                lc['predicted_class'] = label
                lc['class_confidence'] = conf
                classified_lines.append(lc)
        except Exception as e:
            print(f"[classifier] error: {e}, fallback to OTHER")
            for line in lines:
                lc = dict(line); lc['predicted_class'] = 'OTHER'; lc['class_confidence'] = 0.5
                classified_lines.append(lc)
    else:
        for line in lines:
            lc = dict(line); lc['predicted_class'] = 'OTHER'; lc['class_confidence'] = 0.5
            classified_lines.append(lc)

    # 5. Extract -> data terstruktur
    extracted = extractor.extract(classified_lines)

    # Items
    items_struct = extracted.get('items', [])
    
    # Totals breakdown
    totals = extracted.get('totals', {})
    
    # DEBUG: Show what was classified as GRAND_TOTAL
    grand_total_lines = [l for l in classified_lines if l.get('predicted_class') == 'GRAND_TOTAL']
    if grand_total_lines:
        print(f"[DEBUG] GRAND_TOTAL classified lines:")
        for l in grand_total_lines:
            print(f"  - '{l['text']}' (confidence: {l.get('class_confidence', 0):.2f})")
    
    # Simple format response
    return {
        "success": True,
        "store": extracted.get('store', ''),
        "date": extracted.get('date', ''),
        "items": [
            {
                "name": item['name'],
                "qty": item['qty'],
                "price": float(item['price'])
            }
            for item in items_struct
        ],
        "total": float(totals.get('grand_total', 0.0) or extracted.get('total', 0.0))
    }


# ============================================================
# Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve UI (index_v2.html)."""
    return templates.TemplateResponse("index_v2.html", {"request": request})


@app.get("/api/health")
async def health():
    """Health check — returns model status."""
    classifier_version = getattr(classifier, '_version', 'V1') if classifier else None
    is_v2_plus = classifier and getattr(classifier, '_is_v2', False)
    is_context = getattr(classifier, '_is_context_model', False)
    context_window = getattr(classifier, '_context_window', None)
    
    classifier_metrics = None
    if is_v2_plus:
        classifier_metrics = {
            "num_categories": 12,
            "architecture": f"context-aware (±{context_window} lines)" if is_context else "standard"
        }
        if classifier_version == "V5":
            classifier_metrics["accuracy"] = 86.26
            classifier_metrics["grand_total_accuracy"] = 82.65
            classifier_metrics["dataset"] = "Indonesia-only"
    
    return {
        "status": "ok",
        "models": {
            "ocr": ocr_engine is not None,
            "ocr_version": "PaddleOCR",
            "ocr_metrics": {
                "engine": "PaddleOCR",
                "speed": "2-3x faster than EasyOCR",
                "accuracy": "State-of-the-art for receipts",
                "languages": "80+ supported"
            },
            "classifier": classifier is not None,
            "classifier_version": classifier_version,
            "classifier_metrics": classifier_metrics,
            "extractor": extractor is not None,
        }
    }


@app.post("/api/predict")
async def predict(image: UploadFile = File(..., description="Receipt image (JPG, PNG, WEBP)")):
    """
    Complete OCR pipeline with simple structured output.
    
    **Pipeline**: Image → Preprocessing → OCR → Classification (V4 Context-Aware, 12 categories) → Extraction
    
    **Returns**:
    ```json
    {
      "success": true,
      "store": "Kopi Nako Summarecon Bekasi",
      "date": "Jun 18, 2023",
      "items": [
        {
          "name": "Iced Matcha Latte",
          "qty": 1,
          "price": 29000.0
        },
        {
          "name": "Kahlua Kopi",
          "qty": 1,
          "price": 27000.0
        }
      ],
      "total": 169400.0
    }
    ```
    
    **12 Categories** (used internally for better extraction):
    - STORE, ADDRESS_CONTACT, DATE, ITEM_DESC, ITEM_PRICE/QTY
    - SUBTOTAL, TAX, DISCOUNT, SERVICE_CHARGE, GRAND_TOTAL
    - CASH_PAYMENT, OTHER
    """
    content_type = image.content_type or ""
    if "image/" not in content_type:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {content_type}. Use JPG/PNG/WEBP.")

    image_bytes = await image.read()
    return await _run_pipeline(image_bytes)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web.api_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
