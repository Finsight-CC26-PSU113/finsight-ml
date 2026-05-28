# OCR FinSight — API Integration Guide

## Base URL

| Environment    | URL                          |
| -------------- | ---------------------------- |
| Local dev      | `http://localhost:8000`      |
| Docker         | `http://localhost:8000`      |
| VPS/Production | `http://YOUR_SERVER_IP:8000` |

---

## Run the API

```bash
# Development (Windows native)
python web/api_v2.py

# Development (WSL/Linux)
wsl python3 web/api_v2.py

# Production (uvicorn, 2 workers)
uvicorn web.api_v2:app --host 0.0.0.0 --port 8000 --workers 2

# Docker
docker-compose up
```

Interactive docs: `http://localhost:8000/api/docs`

---

## Endpoints

### `POST /api/scan`

Upload gambar struk, dapat JSON hasil ekstraksi.

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Field: `image` (file JPG/PNG/WEBP)

**Response JSON:**

```json
{
  "success": true,
  "store": "Ichiban Sushi",
  "date": "Aug 19, 2024 6:32:54 PM",
  "items": [
    { "name": "Beef Teriyaki Ramen", "qty": 1, "price": 42000.0 },
    { "name": "Katsu Roll", "qty": 1, "price": 35000.0 }
  ],
  "total": 257565.0,
  "totals": {
    "grand_total": 257565.0,
    "subtotal": 223000.0,
    "discount": 0.0,
    "tax": 23415.0,
    "cash": 0.0,
    "change": 0.0
  },
  "address": "AEON Sentul City, Bogor",
  "raw_lines": [
    {
      "text": "ICHIBAN SUSHI",
      "label": "STORE",
      "confidence": 0.95,
      "ocr_confidence": 0.99,
      "bbox": { "x_min": 0.2, "y_min": 0.03, "x_max": 0.8, "y_max": 0.07 }
    }
  ],
  "stats": {
    "total_lines": 42,
    "avg_confidence": 87.3,
    "processing_time_ms": 2150
  },
  "processing_time_ms": 2150
}
```

### `GET /api/health`

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

## Integration Examples

### JavaScript / TypeScript (Frontend)

```javascript
// Vanilla JS
async function scanReceipt(file) {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch("http://localhost:8000/api/scan", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail);
  }

  return await response.json();
}

// Usage
const input = document.getElementById("receipt-input");
input.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  const result = await scanReceipt(file);
  console.log("Store:", result.store);
  console.log("Total:", result.total);
  console.log("Items:", result.items);
});
```

```typescript
// TypeScript with types
interface ScanResult {
  success: boolean;
  store: string;
  date: string;
  items: Array<{ name: string; qty: number; price: number }>;
  total: number;
  totals: {
    grand_total: number;
    subtotal: number;
    discount: number;
    tax: number;
    cash: number;
    change: number;
  };
  address: string;
  raw_lines: Array<{
    text: string;
    label: string;
    confidence: number;
    ocr_confidence: number;
    bbox: { x_min: number; y_min: number; x_max: number; y_max: number };
  }>;
  stats: {
    total_lines: number;
    avg_confidence: number;
    processing_time_ms: number;
  };
  processing_time_ms: number;
}

async function scanReceipt(file: File): Promise<ScanResult> {
  const form = new FormData();
  form.append("image", file);
  const res = await fetch("/api/scan", { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}
```

### React

```jsx
import { useState } from "react";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export function ReceiptScanner() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const form = new FormData();
      form.append("image", file);

      const res = await fetch(`${API_URL}/api/scan`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail);
      }

      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleUpload} />
      {loading && <p>Scanning...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}
      {result && (
        <div>
          <h3>{result.store}</h3>
          <p>Date: {result.date}</p>
          <p>Total: Rp {result.total.toLocaleString("id-ID")}</p>
          <ul>
            {result.items.map((item, i) => (
              <li key={i}>
                {item.name} — Rp {item.price.toLocaleString("id-ID")}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

### Flutter / Dart (Mobile)

```dart
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

const String apiUrl = 'http://YOUR_SERVER_IP:8000';

Future<Map<String, dynamic>> scanReceipt(File imageFile) async {
  final uri = Uri.parse('$apiUrl/api/scan');
  final request = http.MultipartRequest('POST', uri);

  request.files.add(
    await http.MultipartFile.fromPath('image', imageFile.path),
  );

  final streamedResponse = await request.send();
  final response = await http.Response.fromStream(streamedResponse);

  if (response.statusCode != 200) {
    final error = jsonDecode(response.body);
    throw Exception(error['detail']);
  }

  return jsonDecode(response.body);
}

// Usage
void main() async {
  final file = File('/path/to/receipt.jpg');
  final result = await scanReceipt(file);

  print('Store: ${result['store']}');
  print('Date: ${result['date']}');
  print('Total: ${result['total']}');

  for (final item in result['items']) {
    print('  - ${item['name']}: ${item['price']}');
  }
}
```

### Python (Backend-to-Backend)

```python
import requests

API_URL = "http://localhost:8000"

def scan_receipt(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/api/scan",
            files={"image": (image_path, f, "image/jpeg")},
            timeout=30,
        )
    response.raise_for_status()
    return response.json()

# Usage
result = scan_receipt("receipt.jpg")
print(f"Store: {result['store']}")
print(f"Total: Rp {result['total']:,.0f}")
```

### cURL (Testing)

```bash
# Scan receipt
curl -X POST http://localhost:8000/api/scan \
     -F "image=@receipt.jpg" \
     -H "Accept: application/json" | python -m json.tool

# Health check
curl http://localhost:8000/api/health
```

---

## Error Responses

| Status | Meaning           | Example                                       |
| ------ | ----------------- | --------------------------------------------- |
| `400`  | Invalid file type | `{"detail": "Invalid file type: text/plain"}` |
| `422`  | No text detected  | `{"detail": "No text detected in image."}`    |
| `500`  | Processing error  | `{"detail": "Processing failed: ..."}`        |
| `503`  | Model not ready   | `{"detail": "OCR engine not ready"}`          |

---

## Production Deployment

### VPS (Ubuntu)

```bash
# 1. Clone repo
git clone https://github.com/Finsight-CC26-PSU113/finsight-ml.git
cd finsight-ml
git checkout recepit_detection

# 2. Install dependencies
python setup.py

# 3. Run with uvicorn (background)
nohup uvicorn web.api_v2:app --host 0.0.0.0 --port 8000 --workers 2 > api.log 2>&1 &

# 4. Or use systemd service (recommended)
# See: https://fastapi.tiangolo.com/deployment/server-workers/
```

### Docker

```bash
docker build -t ocr-finsight .
docker run -d -p 8000:8000 --name ocr-api ocr-finsight

# Or with docker-compose
docker-compose up -d
```

### Nginx reverse proxy (optional, for HTTPS)

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10M;
    }
}
```

---

## Fine-tuning EasyOCR (Optional)

Jika ingin meningkatkan akurasi OCR dari 81% ke ~85-88%:

```bash
# Kita sudah punya 31,039 OCR crops dari Gemini labeling
wsl python3 scripts/finetune_easyocr.py \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-4

# Output: models/finetuned_easyocr/best_model.pth
# Estimasi waktu: ~2 jam di GPU RTX 3050
```

Setelah fine-tune, update `OCREngine` untuk load model baru:

```python
# src/ocr_engine.py — sudah otomatis load finetuned model jika ada
# models/finetuned_easyocr/best_model.pth
```
