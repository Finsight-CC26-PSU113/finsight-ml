# OCR FinSight — API Documentation

**Version:** 2.0  
**Base URL:** `http://localhost:8000`  
**Interactive Docs:** `http://localhost:8000/api/docs` (Swagger UI)

---

## Overview

OCR FinSight API menerima gambar struk belanja dan mengembalikan data terstruktur hasil ekstraksi ML pipeline:

```
Image → EasyOCR (text detection) → BiLSTM Classifier (line labeling) → Extractor (structured JSON)
```

---

## Run the API

```bash
# Windows native (no WSL needed)
python web/api_v2.py

# Production (multi-worker)
uvicorn web.api_v2:app --host 0.0.0.0 --port 8000 --workers 2

# Docker
docker-compose up
```

---

## Endpoints

### `POST /api/scan`

Scan gambar struk dan ekstrak data terstruktur.

**Request**

| Field   | Type   | Required | Description                             |
| ------- | ------ | -------- | --------------------------------------- |
| `image` | `file` | ✅       | Gambar struk (JPG, PNG, WEBP, max 10MB) |

```
Content-Type: multipart/form-data
```

**Response `200 OK`**

```json
{
  "success": true,
  "store": "Ichiban Sushi",
  "date": "Aug 19, 2024 6:32:54 PM",
  "items": [
    { "name": "Beef Teriyaki Ramen", "qty": 1, "price": 42000.0 },
    { "name": "Katsu Roll", "qty": 1, "price": 35000.0 },
    { "name": "Salmon Sakura", "qty": 1, "price": 35000.0 }
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
      "confidence": 0.952,
      "ocr_confidence": 0.991,
      "bbox": { "x_min": 0.21, "y_min": 0.03, "x_max": 0.79, "y_max": 0.07 }
    },
    {
      "text": "Aug 19, 2024 6:32:54 PM",
      "label": "DATE",
      "confidence": 0.887,
      "ocr_confidence": 0.975,
      "bbox": { "x_min": 0.05, "y_min": 0.19, "x_max": 0.55, "y_max": 0.22 }
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

**Response Fields**

| Field                        | Type     | Description                            |
| ---------------------------- | -------- | -------------------------------------- |
| `success`                    | `bool`   | Always `true` on 200                   |
| `store`                      | `string` | Nama toko/restoran                     |
| `date`                       | `string` | Tanggal & waktu transaksi              |
| `items`                      | `array`  | Daftar item belanja                    |
| `items[].name`               | `string` | Nama item                              |
| `items[].qty`                | `int`    | Jumlah                                 |
| `items[].price`              | `float`  | Harga per item                         |
| `total`                      | `float`  | Grand total (yang harus dibayar)       |
| `totals.grand_total`         | `float`  | Grand total                            |
| `totals.subtotal`            | `float`  | Subtotal sebelum pajak                 |
| `totals.discount`            | `float`  | Total diskon                           |
| `totals.tax`                 | `float`  | Pajak (PPN/GST/Tax)                    |
| `totals.cash`                | `float`  | Uang tunai yang dibayar                |
| `totals.change`              | `float`  | Kembalian                              |
| `address`                    | `string` | Alamat/kontak toko                     |
| `raw_lines`                  | `array`  | Semua baris OCR dengan label ML        |
| `raw_lines[].text`           | `string` | Teks hasil OCR                         |
| `raw_lines[].label`          | `string` | Label ML (lihat tabel label di bawah)  |
| `raw_lines[].confidence`     | `float`  | Confidence classifier (0–1)            |
| `raw_lines[].ocr_confidence` | `float`  | Confidence EasyOCR (0–1)               |
| `raw_lines[].bbox`           | `object` | Bounding box (normalized 0–1)          |
| `stats.total_lines`          | `int`    | Jumlah baris terdeteksi                |
| `stats.avg_confidence`       | `float`  | Rata-rata confidence (%)               |
| `stats.processing_time_ms`   | `int`    | Waktu proses (ms)                      |
| `processing_time_ms`         | `int`    | Sama dengan `stats.processing_time_ms` |

**ML Label Classes**

| Label             | Deskripsi                      | Contoh                                    |
| ----------------- | ------------------------------ | ----------------------------------------- |
| `STORE`           | Nama toko/bisnis               | `"INDOMARET"`, `"Ichiban Sushi"`          |
| `ADDRESS_CONTACT` | Alamat, telp, email, NPWP      | `"Jl. Sudirman No. 10"`, `"Tel: 021-xxx"` |
| `DATE`            | Tanggal & waktu transaksi      | `"Aug 19, 2024"`, `"24/05/2026 10:30"`    |
| `ITEM_DESC`       | Nama barang/menu               | `"Nasi Goreng"`, `"Beef Teriyaki Ramen"`  |
| `ITEM_PRICE/QTY`  | Harga & jumlah barang          | `"Rp 25.000"`, `"2 x 5000"`               |
| `TOTAL_PAYMENT`   | Total, subtotal, pajak, diskon | `"Total 257,565"`, `"Subtotal 223,000"`   |
| `OTHER`           | Lainnya                        | Kasir, footer, nomor meja                 |

**Error Responses**

| Status | Kondisi                   | Response                                      |
| ------ | ------------------------- | --------------------------------------------- |
| `400`  | File bukan gambar         | `{"detail": "Invalid file type: text/plain"}` |
| `422`  | Tidak ada teks terdeteksi | `{"detail": "No text detected in image."}`    |
| `500`  | Error internal            | `{"detail": "Processing failed: ..."}`        |
| `503`  | Model belum siap          | `{"detail": "OCR engine not ready"}`          |

---

### `GET /api/health`

Cek status model yang ter-load.

**Response `200 OK`**

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

| Field                       | Description                             |
| --------------------------- | --------------------------------------- |
| `models.ocr`                | EasyOCR loaded                          |
| `models.classifier`         | BiLSTM classifier loaded                |
| `models.classifier_version` | `"v2"` (85% acc) atau `"v1"` (fallback) |
| `models.extractor`          | Extractor loaded                        |

---

## Integration Examples

### JavaScript / Fetch API

```javascript
const API_URL = "http://localhost:8000";

async function scanReceipt(file) {
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

  return await res.json();
}

// Contoh penggunaan
document.getElementById("upload").addEventListener("change", async (e) => {
  const result = await scanReceipt(e.target.files[0]);
  console.log("Toko:", result.store);
  console.log("Tanggal:", result.date);
  console.log("Total:", result.total);
  console.log("Items:", result.items);
});
```

### React

```jsx
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

async function scanReceipt(file) {
  const form = new FormData();
  form.append("image", file);
  const res = await fetch(`${API_URL}/api/scan`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}

export function ReceiptScanner() {
  const [result, setResult] = React.useState(null);
  const [loading, setLoading] = React.useState(false);

  const handleFile = async (e) => {
    setLoading(true);
    try {
      setResult(await scanReceipt(e.target.files[0]));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleFile} />
      {loading && <p>Memproses...</p>}
      {result && (
        <div>
          <h2>{result.store}</h2>
          <p>Tanggal: {result.date}</p>
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

### Flutter / Dart

```dart
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

const String apiUrl = 'http://YOUR_SERVER_IP:8000';

Future<Map<String, dynamic>> scanReceipt(File imageFile) async {
  final request = http.MultipartRequest(
    'POST',
    Uri.parse('$apiUrl/api/scan'),
  );
  request.files.add(
    await http.MultipartFile.fromPath('image', imageFile.path),
  );

  final response = await http.Response.fromStream(await request.send());

  if (response.statusCode != 200) {
    throw Exception(jsonDecode(response.body)['detail']);
  }
  return jsonDecode(response.body);
}

// Contoh penggunaan
void main() async {
  final result = await scanReceipt(File('/path/to/receipt.jpg'));
  print('Toko: ${result['store']}');
  print('Total: ${result['total']}');
  for (final item in result['items']) {
    print('  - ${item['name']}: ${item['price']}');
  }
}
```

### Python (Backend-to-Backend)

```python
import requests

def scan_receipt(image_path: str, api_url: str = "http://localhost:8000") -> dict:
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{api_url}/api/scan",
            files={"image": (image_path, f, "image/jpeg")},
            timeout=30,
        )
    response.raise_for_status()
    return response.json()

result = scan_receipt("receipt.jpg")
print(f"Toko  : {result['store']}")
print(f"Tanggal: {result['date']}")
print(f"Total : Rp {result['total']:,.0f}")
for item in result['items']:
    print(f"  - {item['name']} x{item['qty']} = Rp {item['price']:,.0f}")
```

### cURL

```bash
# Scan struk
curl -X POST http://localhost:8000/api/scan \
     -F "image=@receipt.jpg" | python -m json.tool

# Health check
curl http://localhost:8000/api/health
```

---

## ML Pipeline Detail

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐    ┌──────────────┐
│  Image      │───▶│  EasyOCR         │───▶│  BiLSTM Classifier  │───▶│  Extractor   │
│  (JPG/PNG)  │    │  (text + bbox)   │    │  (label per line)   │    │  (JSON out)  │
└─────────────┘    └──────────────────┘    └─────────────────────┘    └──────────────┘
                   • Deteksi teks          • 7 label classes           • store
                   • Bounding box          • 85% accuracy              • date
                   • OCR confidence        • DATE recall 96%           • items
                   • GPU auto-detect       • Trained: 44K lines        • total
                                           • Dataset: MY + ID          • address
```

**Models:**

| Model                | File                                           | Accuracy              | Dataset               |
| -------------------- | ---------------------------------------------- | --------------------- | --------------------- |
| BiLSTM Classifier V2 | `models/classifier_v2/best_weights.weights.h5` | 85% (DATE recall 96%) | 44K lines, 783 images |
| EasyOCR (fine-tuned) | `models/finetuned_easyocr/best_model.pth`      | 81.1% CER             | 33K crops             |
| EasyOCR (base)       | Auto-download via EasyOCR                      | —                     | Fallback              |

---

## Deployment

### VPS (Ubuntu 22.04)

```bash
# Clone & setup
git clone https://github.com/Finsight-CC26-PSU113/finsight-ml.git
cd finsight-ml && git checkout recepit_detection
python setup.py

# Run (background)
nohup uvicorn web.api_v2:app --host 0.0.0.0 --port 8000 --workers 2 > api.log 2>&1 &
```

### Docker

```bash
docker build -t ocr-finsight .
docker run -d -p 8000:8000 ocr-finsight
# atau
docker-compose up -d
```

### Minimum Spec

| Resource | Minimum     | Recommended |
| -------- | ----------- | ----------- |
| RAM      | 4 GB        | 8 GB        |
| CPU      | 2 vCPU      | 4 vCPU      |
| Storage  | 10 GB       | 20 GB       |
| GPU      | Tidak wajib | NVIDIA T4   |

Inference time: ~2–3 detik/gambar (CPU)
