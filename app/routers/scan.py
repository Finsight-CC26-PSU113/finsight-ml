"""
OCR FinSight - Scan Router
POST /api/scan  — upload receipt image, get structured JSON
GET  /api/health — model status
"""

import time
import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.receipt import ScanResult, ItemResult, TotalsResult
from app.dependencies import get_ocr, get_classifier, get_extractor
from app.services.preprocessor import preprocess_for_easyocr
from app.services.classifier import predict_lines

router = APIRouter()


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


@router.post("/scan", response_model=ScanResult)
async def scan_receipt(
    image: UploadFile = File(..., description="Receipt image (JPG, PNG, WEBP)"),
):
    """
    Scan a receipt image and return structured data.

    Returns store name, date, items list, totals breakdown, and address.
    """
    ocr = get_ocr()
    if not ocr:
        raise HTTPException(status_code=503, detail="OCR engine not ready")

    content_type = image.content_type or ""
    if not any(t in content_type for t in ["image/jpeg", "image/png", "image/webp", "image/"]):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

    t_start = time.time()

    try:
        file_bytes = np.frombuffer(await image.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Cannot decode image.")

        img = preprocess_for_easyocr(img)
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
            except Exception:
                classified_lines = [
                    {**l, 'predicted_class': 'OTHER', 'class_confidence': 0.5}
                    for l in lines
                ]
        else:
            classified_lines = [
                {**l, 'predicted_class': 'OTHER', 'class_confidence': 0.5}
                for l in lines
            ]

        result = get_extractor().extract(classified_lines)
        t_ms = int((time.time() - t_start) * 1000)

        return ScanResult(
            success=True,
            store=result['store'],
            date=result['date'],
            items=[ItemResult(name=i['name'], qty=i['qty'], price=i['price'])
                   for i in result['items']],
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
                    },
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
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
