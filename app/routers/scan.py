"""
OCR FinSight - Scan Router

POST /api/predict  — main endpoint: upload receipt, get structured JSON
POST /api/scan     — alias for /api/predict
GET  /api/health   — model status
"""

import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.receipt import ScanResult, ItemResult
from app.dependencies import get_ocr, get_classifier, get_extractor
from app.services.preprocessor import preprocess_for_easyocr
from app.services.classifier import predict_lines

router = APIRouter()


async def _run_pipeline(image_bytes: bytes) -> dict:
    """Decode → preprocess → OCR → classify → extract. Returns minimal JSON."""
    ocr = get_ocr()
    if not ocr:
        raise HTTPException(status_code=503, detail="OCR engine not ready")

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image. Use JPG/PNG/WEBP.")

    img = preprocess_for_easyocr(img, upscale_factor=2.0)

    lines, _ = ocr.read_receipt_with_image_info(img)
    if not lines:
        raise HTTPException(status_code=422, detail="No text detected in image.")

    clf = get_classifier()
    classified_lines = []
    if clf:
        try:
            labels, confs = predict_lines(clf, lines)
            for line, label, conf in zip(lines, labels, confs):
                lc = dict(line)
                lc['predicted_class'] = label
                lc['class_confidence'] = conf
                classified_lines.append(lc)
        except Exception as e:
            print(f"[classifier] error: {e}, fallback to OTHER")
            classified_lines = [{**l, 'predicted_class': 'OTHER', 'class_confidence': 0.5} for l in lines]
    else:
        classified_lines = [{**l, 'predicted_class': 'OTHER', 'class_confidence': 0.5} for l in lines]

    extracted = get_extractor().extract(classified_lines)

    return {
        "success": True,
        "store": extracted.get('store', ''),
        "date": extracted.get('date', ''),
        "items": [
            {"name": it['name'], "qty": it['qty'], "price": it['price']}
            for it in extracted.get('items', [])
        ],
        "total": extracted.get('total', 0.0),
    }


@router.get("/health")
async def health():
    """Health check — returns model load status."""
    clf = get_classifier()
    return {
        "status": "ok",
        "models": {
            "ocr": get_ocr() is not None,
            "classifier": clf is not None,
            "classifier_version": "v2" if (clf and getattr(clf, '_is_v2', False)) else "v1",
            "extractor": get_extractor() is not None,
        },
    }


@router.post("/predict", response_model=ScanResult)
async def predict(image: UploadFile = File(..., description="Receipt image (JPG, PNG, WEBP)")):
    """
    Scan receipt image and return structured data.

    Returns: store name, date, items list with qty and price, grand total.
    """
    if "image/" not in (image.content_type or ""):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {image.content_type}")
    return await _run_pipeline(await image.read())


@router.post("/scan", response_model=ScanResult)
async def scan(image: UploadFile = File(..., description="Alias for /api/predict")):
    """Alias for /api/predict — backward compatibility."""
    if "image/" not in (image.content_type or ""):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {image.content_type}")
    return await _run_pipeline(await image.read())
