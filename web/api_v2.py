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

from src.config import ROOT_DIR, LINE_CLASSES, MAX_TEXT_LENGTH
from src.preprocessing import preprocess_for_easyocr
from src.ocr_engine import OCREngine
from src.extractor import ReceiptExtractor

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

    # 1. OCR Engine (with fine-tuned model if available)
    print("📦 Loading EasyOCR...")
    finetuned_path = ROOT_DIR / "models" / "finetuned_easyocr" / "best_model.pth"
    if finetuned_path.exists():
        print(f"   Found fine-tuned model: {finetuned_path}")
        ocr_engine = OCREngine(model_path=str(finetuned_path))
    else:
        print("   Using default EasyOCR model")
        ocr_engine = OCREngine()
    print("✅ EasyOCR ready")

    # 2. Classifier V2 (12 categories)
    print("📦 Loading Classifier V2...")
    try:
        import tensorflow as tf
        from src.model import ReceiptLineClassifier

        v2_weights = ROOT_DIR / "models" / "classifier_v2" / "best_weights.weights.h5"
        if not v2_weights.exists():
            raise FileNotFoundError(f"Classifier V2 weights not found: {v2_weights}")
            
        model = ReceiptLineClassifier()

        dummy_chars = tf.zeros((1, MAX_TEXT_LENGTH), dtype=tf.int32)
        dummy_tf = tf.zeros((1, 10), dtype=tf.float32)
        dummy_pf = tf.zeros((1, 5), dtype=tf.float32)
        _ = model((dummy_chars, dummy_tf, dummy_pf), training=False)

        model.load_weights(str(v2_weights))
        model._is_v2 = True
        classifier = model
        print("✅ Classifier V2 loaded (12 categories, 82.45% test accuracy)")
    except Exception as e:
        print(f"⚠️  Classifier V2 failed: {e}")
        try:
            from src.online_learning import OnlineLearningModel
            classifier = OnlineLearningModel()
            classifier._is_v2 = False
            print("✅ Classifier V1 loaded (fallback)")
        except Exception as e2:
            print(f"❌ No classifier available: {e2}")
            classifier = None

    # 3. Extractor
    extractor = ReceiptExtractor()
    print("✅ Extractor ready")
    
    print("\n📊 Model Summary:")
    print(f"   OCR: {'Fine-tuned EasyOCR (62.89% exact match, 13.12% CER)' if finetuned_path.exists() else 'Default EasyOCR'}")
    print(f"   Classifier: {'V2 (12 categories, 82.45% accuracy)' if (classifier and getattr(classifier, '_is_v2', False)) else 'V1 (fallback)'}")
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


def _predict_with_v2(model, lines: list[dict]) -> tuple[list[str], list[float]]:
    """Predict labels using classifier V2 (batch inference)."""
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
            if is_v2:
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

    # Calculate items total
    items_total = sum(it['price'] * it['qty'] for it in items_struct)
    
    return {
        "success": True,
        "receipt": {
            "store_name": extracted.get('store', ''),
            "date": extracted.get('date', ''),
            "address": extracted.get('address', ''),
        },
        "items": [
            {
                "name": it['name'],
                "quantity": it['qty'],
                "unit_price": it['price'],
                "total_price": round(it['price'] * it['qty'], 2)
            }
            for it in items_struct
        ],
        "financial_summary": {
            "items_total": round(items_total, 2),
            "subtotal": round(totals.get('subtotal', 0.0), 2),
            "tax": round(totals.get('tax', 0.0), 2),
            "discount": round(totals.get('discount', 0.0), 2),
            "service_charge": round(totals.get('service_charge', 0.0), 2),
            "grand_total": round(totals.get('grand_total', 0.0), 2),
            "payment": {
                "cash_paid": round(totals.get('cash', 0.0), 2),
                "change": round(totals.get('change', 0.0), 2),
            }
        }
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
    finetuned_path = ROOT_DIR / "models" / "finetuned_easyocr" / "best_model.pth"
    return {
        "status": "ok",
        "models": {
            "ocr": ocr_engine is not None,
            "ocr_version": "fine-tuned" if finetuned_path.exists() else "default",
            "ocr_metrics": {
                "exact_match_accuracy": 62.89,
                "character_error_rate": 13.12
            } if finetuned_path.exists() else None,
            "classifier": classifier is not None,
            "classifier_version": "v2" if (classifier and getattr(classifier, '_is_v2', False)) else "v1",
            "classifier_metrics": {
                "test_accuracy": 82.45,
                "num_categories": 12
            } if (classifier and getattr(classifier, '_is_v2', False)) else None,
            "extractor": extractor is not None,
        }
    }


@app.post("/api/predict")
async def predict(image: UploadFile = File(..., description="Receipt image (JPG, PNG, WEBP)")):
    """
    Complete OCR pipeline with clean structured output.
    
    **Pipeline**: Image → Preprocessing → OCR → Classification (12 categories) → Extraction
    
    **Returns**:
    ```json
    {
      "success": true,
      "receipt": {
        "store_name": "INDOMARET",
        "date": "2026-06-01 14:30",
        "address": "Jl. Sudirman No. 123, Jakarta"
      },
      "items": [
        {
          "name": "Indomie Goreng",
          "quantity": 2,
          "unit_price": 3500.0,
          "total_price": 7000.0
        },
        {
          "name": "Aqua 600ml",
          "quantity": 1,
          "unit_price": 3000.0,
          "total_price": 3000.0
        }
      ],
      "financial_summary": {
        "items_total": 10000.0,
        "subtotal": 10000.0,
        "tax": 1100.0,
        "discount": 500.0,
        "service_charge": 0.0,
        "grand_total": 10600.0,
        "payment": {
          "cash_paid": 15000.0,
          "change": 4400.0
        }
      },
      "lines_by_category": {
        "STORE": [
          {"text": "INDOMARET", "confidence": 0.95, "ocr_confidence": 0.98}
        ],
        "DATE": [
          {"text": "01/06/2026 14:30", "confidence": 0.99, "ocr_confidence": 0.97}
        ],
        "ITEM_DESC": [
          {"text": "Indomie Goreng", "confidence": 0.91, "ocr_confidence": 0.96},
          {"text": "Aqua 600ml", "confidence": 0.89, "ocr_confidence": 0.95}
        ],
        "TAX": [
          {"text": "PPN 11%", "confidence": 0.84, "ocr_confidence": 0.92},
          {"text": "Rp 1.100", "confidence": 0.82, "ocr_confidence": 0.99}
        ],
        "DISCOUNT": [
          {"text": "Member Discount", "confidence": 0.87, "ocr_confidence": 0.94},
          {"text": "Rp 500", "confidence": 0.85, "ocr_confidence": 0.98}
        ],
        "GRAND_TOTAL": [
          {"text": "Total Bayar", "confidence": 0.93, "ocr_confidence": 0.96},
          {"text": "Rp 10.600", "confidence": 0.91, "ocr_confidence": 0.99}
        ]
      },
      "metadata": {
        "total_lines": 45,
        "total_items": 2,
        "avg_classification_confidence": 85.2,
        "avg_ocr_confidence": 96.5,
        "classifier_version": "v2_12_categories",
        "categories_detected": ["STORE", "ADDRESS_CONTACT", "DATE", "ITEM_DESC", "ITEM_PRICE/QTY", "SUBTOTAL", "TAX", "DISCOUNT", "GRAND_TOTAL", "CASH_PAYMENT", "OTHER"]
      }
    }
    ```
    
    **12 Categories**:
    - **STORE**: Store/business name
    - **ADDRESS_CONTACT**: Address, phone, email, tax ID
    - **DATE**: Date and/or time
    - **ITEM_DESC**: Product/item names
    - **ITEM_PRICE/QTY**: Item prices or quantities
    - **SUBTOTAL**: Subtotal before tax/charges
    - **TAX**: Tax, GST, VAT, PPN, SST
    - **DISCOUNT**: Discounts, vouchers, promotions
    - **SERVICE_CHARGE**: Service charges, tips
    - **GRAND_TOTAL**: Final total amount
    - **CASH_PAYMENT**: Cash paid, change given
    - **OTHER**: Everything else (cashier, receipt number, footer, etc.)
    
    **Note**: Coordinate information (bbox) is not included in the response for cleaner output.
    """
    content_type = image.content_type or ""
    if "image/" not in content_type:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {content_type}. Use JPG/PNG/WEBP.")

    image_bytes = await image.read()
    return await _run_pipeline(image_bytes)


# Alias for backward compatibility
@app.post("/api/scan")
async def scan(image: UploadFile = File(..., description="Receipt image (alias of /api/predict)")):
    return await predict(image)


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
