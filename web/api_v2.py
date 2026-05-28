"""
OCR FinSight - FastAPI Backend (Production-ready)
==================================================
REST API untuk receipt scanning. Bisa dipanggil dari:
- Frontend web (React, Vue, dll)
- Mobile app (Flutter, React Native)
- Backend lain (microservice)

Endpoints:
    POST /api/scan          → Upload gambar, dapat JSON hasil ekstraksi
    GET  /api/health        → Health check
    GET  /api/docs          → Swagger UI (auto-generated)

Usage:
    # Development
    python web/api_v2.py

    # Production (dengan uvicorn)
    uvicorn web.api_v2:app --host 0.0.0.0 --port 8000 --workers 2

    # Docker
    docker run -p 8000:8000 ocr-finsight

Example call (curl):
    curl -X POST http://localhost:8000/api/scan \
         -F "image=@receipt.jpg" \
         -H "Accept: application/json"

Example call (Python requests):
    import requests
    with open("receipt.jpg", "rb") as f:
        r = requests.post("http://localhost:8000/api/scan", files={"image": f})
    print(r.json())

Example call (JavaScript fetch):
    const form = new FormData();
    form.append("image", fileInput.files[0]);
    const res = await fetch("http://localhost:8000/api/scan", {
        method: "POST",
        body: form
    });
    const data = await res.json();
"""

import sys
import re
import time
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ROOT_DIR, LINE_CLASSES, MAX_TEXT_LENGTH
from src.preprocessing import deskew
from src.ocr_engine import OCREngine
from src.extractor import ReceiptExtractor

# ============================================================
# App setup
# ============================================================

app = FastAPI(
    title="OCR FinSight API",
    description="Receipt scanning API — extracts store, date, items, and total from receipt images.",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — allow all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    print("🧾 OCR FinSight API v2.0 — Starting up")
    print("=" * 55)
    
    # 1. OCR Engine
    print("📦 Loading EasyOCR...")
    ocr_engine = OCREngine()
    print("✅ EasyOCR ready")
    
    # 2. Classifier V2
    print("📦 Loading Classifier V2...")
    try:
        import tensorflow as tf
        from src.model import ReceiptLineClassifier
        
        v2_weights = ROOT_DIR / "models" / "classifier_v2" / "best_weights.weights.h5"
        model = ReceiptLineClassifier()
        
        dummy_chars = tf.zeros((1, MAX_TEXT_LENGTH), dtype=tf.int32)
        dummy_tf = tf.zeros((1, 10), dtype=tf.float32)
        dummy_pf = tf.zeros((1, 5), dtype=tf.float32)
        _ = model((dummy_chars, dummy_tf, dummy_pf), training=False)
        
        model.load_weights(str(v2_weights))
        model._is_v2 = True
        classifier = model
        print("✅ Classifier V2 loaded (85% acc, DATE recall 96%)")
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
    print("\n🌐 API ready at http://0.0.0.0:8000")
    print("📖 Docs at http://localhost:8000/api/docs")
    print("=" * 55)


# ============================================================
# Feature extraction for classifier V2
# ============================================================

def _predict_with_v2(model, lines: list[dict]) -> tuple[list[str], list[float]]:
    """Predict labels using classifier V2."""
    import re
    import tensorflow as tf
    
    DATE_REGEX = re.compile(
        r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})|(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})|'
        r'(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})|'
        r'((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}[\s,]+\d{2,4})',
        re.IGNORECASE
    )
    TIME_REGEX = re.compile(r'\d{1,2}:\d{2}(:\d{2})?')
    QTY_REGEX = re.compile(r'\d+\s*[xX\*@]\s*[\d.,]+|@\s*[\d.,]+')
    CURRENCY_REGEX = re.compile(r'(rm|rp|idr|usd|sgd|sr|\$|€)\s*[\d.,]+', re.IGNORECASE)
    
    def text_feats(text):
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
    
    def char_ids(text, max_len=MAX_TEXT_LENGTH):
        text = str(text) if text else ""
        ids = [min(ord(c), 127) for c in text[:max_len]]
        return ids + [0] * (max_len - len(ids))
    
    n = len(lines)
    chars = np.array([char_ids(l.get('text', '')) for l in lines], dtype=np.int32)
    tfeats = np.array([text_feats(l.get('text', '')) for l in lines], dtype=np.float32)
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
# Response models
# ============================================================

class ItemResult(BaseModel):
    name: str
    qty: int
    price: float


class TotalsResult(BaseModel):
    grand_total: float
    subtotal: float
    discount: float
    tax: float
    cash: float
    change: float


class ScanResult(BaseModel):
    success: bool
    store: str
    date: str
    items: list[ItemResult]
    total: float
    totals: TotalsResult
    address: str
    raw_lines: list[dict]
    stats: dict
    processing_time_ms: int


# ============================================================
# Endpoints
# ============================================================

@app.get("/api/health")
async def health():
    """Health check — returns model status."""
    return {
        "status": "ok",
        "models": {
            "ocr": ocr_engine is not None,
            "classifier": classifier is not None,
            "classifier_version": "v2" if (classifier and getattr(classifier, '_is_v2', False)) else "v1",
            "extractor": extractor is not None,
        }
    }


@app.post("/api/scan", response_model=ScanResult)
async def scan_receipt(image: UploadFile = File(..., description="Receipt image (JPG, PNG, WEBP)")):
    """
    Scan a receipt image and extract structured data.
    
    Returns:
    - **store**: Store/business name
    - **date**: Transaction date and time
    - **items**: List of purchased items with qty and price
    - **total**: Grand total amount
    - **totals**: Breakdown (subtotal, tax, discount, cash, change)
    - **address**: Store address/contact info
    - **raw_lines**: All detected text lines with labels and confidence
    - **stats**: Processing stats (lines detected, avg confidence, time)
    """
    if not ocr_engine:
        raise HTTPException(status_code=503, detail="OCR engine not ready")
    
    # Validate file type
    content_type = image.content_type or ""
    if not any(t in content_type for t in ["image/jpeg", "image/png", "image/webp", "image/"]):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {content_type}. Use JPG, PNG, or WEBP.")
    
    t_start = time.time()
    
    try:
        import cv2
        
        # Read image
        file_bytes = np.frombuffer(await image.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Cannot decode image. Make sure it's a valid JPG/PNG.")
        
        # Preprocess
        img = deskew(img)
        
        # OCR
        lines, img_info = ocr_engine.read_receipt_with_image_info(img)
        
        if not lines:
            raise HTTPException(status_code=422, detail="No text detected in image.")
        
        # Classify
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
                # Fallback: all OTHER
                for line in lines:
                    lc = dict(line)
                    lc['predicted_class'] = 'OTHER'
                    lc['class_confidence'] = 0.5
                    classified_lines.append(lc)
        else:
            for line in lines:
                lc = dict(line)
                lc['predicted_class'] = 'OTHER'
                lc['class_confidence'] = 0.5
                classified_lines.append(lc)
        
        # Extract structured data
        result = extractor.extract(classified_lines)
        
        t_ms = int((time.time() - t_start) * 1000)
        
        # Build response
        return ScanResult(
            success=True,
            store=result['store'],
            date=result['date'],
            items=[
                ItemResult(name=item['name'], qty=item['qty'], price=item['price'])
                for item in result['items']
            ],
            total=result['total'],
            totals=TotalsResult(**result['totals']),
            address=result.get('address', ''),
            raw_lines=[
                {
                    'text': l['text'],
                    'label': l.get('predicted_class', 'OTHER'),
                    'confidence': round(float(l.get('class_confidence', 0.0)), 3),
                    'ocr_confidence': round(float(l.get('confidence', 0.0)), 3),
                    'bbox': {
                        'x_min': round(float(l.get('x_min', 0)), 4),
                        'y_min': round(float(l.get('y_min', 0)), 4),
                        'x_max': round(float(l.get('x_max', 0)), 4),
                        'y_max': round(float(l.get('y_max', 0)), 4),
                    }
                }
                for l in classified_lines
            ],
            stats={
                'total_lines': len(classified_lines),
                'avg_confidence': round(
                    np.mean([l.get('class_confidence', 0) for l in classified_lines]) * 100, 1
                ),
                'processing_time_ms': t_ms,
            },
            processing_time_ms=t_ms,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print("🧾 OCR FinSight FastAPI Backend")
    print("=" * 55)
    uvicorn.run(
        "web.api_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
