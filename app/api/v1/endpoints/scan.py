"""
OCR FinSight — Scan Endpoint
POST /api/v1/predict  — upload receipt image, get structured JSON
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.api.deps import get_classifier, get_extractor, get_ocr
from app.schemas.receipt import ScanResult
from app.services.classifier import predict_lines
from app.services.preprocessor import preprocess_for_easyocr

router = APIRouter(tags=["scan"])


async def _run_pipeline(image_bytes: bytes) -> dict:
    """Full pipeline: decode -> preprocess -> OCR -> classify -> extract."""
    ocr = get_ocr()

    # 1. Decode
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400, detail="Cannot decode image. Use JPG/PNG/WEBP."
        )

    # 2. Preprocess
    img = preprocess_for_easyocr(img, upscale_factor=2.0)

    # 3. OCR
    lines, _ = ocr.read_receipt_with_image_info(img)
    if not lines:
        raise HTTPException(status_code=422, detail="No text detected in image.")

    # 4. Classify
    clf = get_classifier()
    classified_lines = []
    if clf:
        try:
            labels, confs = predict_lines(clf, lines)
            for line, label, conf in zip(lines, labels, confs):
                lc = dict(line)
                lc["predicted_class"] = label
                lc["class_confidence"] = conf
                classified_lines.append(lc)
        except Exception as e:
            print(f"[classifier] error: {e}, fallback to OTHER")
            classified_lines = [
                {**l, "predicted_class": "OTHER", "class_confidence": 0.5}
                for l in lines
            ]
    else:
        classified_lines = [
            {**l, "predicted_class": "OTHER", "class_confidence": 0.5}
            for l in lines
        ]

    # 5. Extract
    extracted = get_extractor().extract(classified_lines)

    return {
        "success": True,
        "store": extracted.get("store", ""),
        "date": extracted.get("date", ""),
        "items": [
            {"name": it["name"], "qty": it["qty"], "price": it["price"]}
            for it in extracted.get("items", [])
        ],
        "total": float(extracted.get("total", 0.0)),
    }


@router.post("/predict", response_model=ScanResult)
async def predict(
    image: UploadFile = File(..., description="Receipt image (JPG, PNG, WEBP)"),
):
    """Scan a receipt image and return structured data.

    Returns: store name, date, items list with qty and price, grand total.
    """
    content_type = image.content_type or ""
    if "image/" not in content_type:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Use JPG/PNG/WEBP.",
        )
    return await _run_pipeline(await image.read())
