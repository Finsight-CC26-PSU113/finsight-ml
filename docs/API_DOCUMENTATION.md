# OCR FinSight — API Documentation

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Health Check

```
GET /api/health
```

**Response:**

```json
{
  "status": "ok",
  "models": {
    "ocr": true,
    "classifier": true,
    "classifier_version": "v2",
    "extractor": true
  }
}
```

---

### 2. Scan Receipt (Main Endpoint)

```
POST /api/predict
POST /api/scan  (alias)
```

**Request:**

- Content-Type: `multipart/form-data`
- Body: `image` (file) — JPG, PNG, atau WEBP

**Response:**

```json
{
  "success": true,
  "store": "Pradana Swalayan",
  "date": "03-Feb-2023",
  "items": [
    { "name": "SAJIKU TEPUNG BUMBU PEDAS 70GR", "qty": 1, "price": 18600.0 },
    { "name": "DoDo COTTEn BUDS 131 (150PCS)", "qty": 1, "price": 6000.0 },
    { "name": "MAM LIME CHARCOAL 800ML", "qty": 1, "price": 8000.0 },
    { "name": "PANTENE CONDITIONER BLACK 75ML", "qty": 1, "price": 13000.0 },
    { "name": "REXONA HIJAB NAT PEACH 45ML", "qty": 1, "price": 21500.0 }
  ],
  "total": 82000.0
}
```

**Error Responses:**

- `400` — Invalid file type
- `422` — No text detected in image
- `503` — OCR engine not ready

---

### 3. Swagger UI (Interactive Docs)

```
GET /api/docs
```

---

## Pipeline Architecture

```
Image Upload
    ↓
[1] Image Preprocessing (preprocessing.py)
    - Deskew (Hough transform)
    - Bilateral filter (noise reduction)
    - CLAHE on L-channel (contrast enhancement)
    - Light sharpening (unsharp mask)
    - Upscale 2× Lanczos
    ↓
[2] OCR — EasyOCR (Fine-tuned)
    - Languages: English + Indonesian
    - Post-OCR horizontal row merging
    - Output: list of text lines with bbox coordinates
    ↓
[3] Line Classification — BiLSTM + Attention (TensorFlow)
    - Model: ReceiptLineClassifier (Model Subclassing)
    - Custom components: TextFeatureLayer, PositionFeatureLayer, FocalLoss
    - Classes: STORE, ADDRESS_CONTACT, DATE, ITEM_DESC, ITEM_PRICE/QTY, TOTAL_PAYMENT, OTHER
    - Accuracy: 85% (validation set)
    ↓
[4] Data Extraction (extractor.py)
    - Store: top-most text lines with alphabetic content
    - Date: regex patterns (numeric DD/MM/YYYY, word-month, compact)
    - Items: price-driven pairing (column-aware, y-overlap matching)
    - Total: keyword-based extraction with cash/change exclusion
    ↓
JSON Response {store, date, items[], total}
```

---

## Technology Stack

| Component        | Technology                          |
| ---------------- | ----------------------------------- |
| Web Framework    | FastAPI 0.121 + Uvicorn             |
| OCR Engine       | EasyOCR (fine-tuned, 20 epochs)     |
| Classifier       | TensorFlow 2.x, BiLSTM + Attention  |
| Image Processing | OpenCV (CLAHE, bilateral, deskew)   |
| Training         | Custom GradientTape loop, FocalLoss |
| Ground Truth     | Gemini Vision API (labeling)        |
| Logging          | TensorBoard                         |

---

## Example Usage

### cURL

```bash
curl -X POST http://localhost:8000/api/scan \
  -F "image=@receipt.jpg"
```

### Python

```python
import requests

with open("receipt.jpg", "rb") as f:
    r = requests.post("http://localhost:8000/api/scan", files={"image": f})
print(r.json())
```

### JavaScript

```javascript
const form = new FormData();
form.append("image", fileInput.files[0]);
const res = await fetch("/api/scan", { method: "POST", body: form });
const data = await res.json();
console.log(data.store, data.items, data.total);
```

---

## Running

```bash
# Development
uvicorn app.main:app --reload --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Docker
docker run -p 8000:8000 ocr-finsight
```
