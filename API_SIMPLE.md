# 🧾 OCR FinSight - Simple API Documentation

API sederhana untuk ekstraksi data dari foto struk belanja.

---

## 🚀 Quick Start

### 1. Start Server

```bash
python web/simple_app.py
```

Server akan berjalan di `http://localhost:5000`

### 2. Test API

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@receipt.jpg"
```

---

## 📡 Endpoints

### `POST /api/predict`

**Main endpoint untuk ekstraksi data dari struk**

#### Request

- **Method:** POST
- **Content-Type:** multipart/form-data
- **Field:** `image` (file) - Gambar struk (JPG, PNG)

#### Response Format

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
    },
    {
      "name": "Rosberi",
      "qty": 1,
      "price": 29000.0
    },
    {
      "name": "Manggo Lassie",
      "qty": 1,
      "price": 29000.0
    }
  ],
  "total": 169400.0
}
```

#### Response Fields

- `success` (boolean) - Status keberhasilan
- `store` (string) - Nama toko/merchant
- `date` (string) - Tanggal transaksi
- `items` (array) - Daftar item belanja
  - `name` (string) - Nama item
  - `qty` (int) - Jumlah/quantity
  - `price` (float) - Harga satuan
- `total` (float) - Total pembayaran (grand total)

#### Error Response

```json
{
  "success": false,
  "error": "Error message here",
  "store": "",
  "date": "",
  "items": [],
  "total": 0.0
}
```

---

### `GET /api/health`

**Health check endpoint**

#### Response

```json
{
  "status": "ok",
  "ocr": true,
  "classifier": true,
  "extractor": true
}
```

---

### `GET /`

**API documentation page** (HTML)

Tampilan dokumentasi API di browser.

---

## 💻 Usage Examples

### Python

```python
import requests

# Upload image
files = {'image': open('receipt.jpg', 'rb')}
response = requests.post('http://localhost:5000/api/predict', files=files)

# Parse result
result = response.json()
if result['success']:
    print(f"Store: {result['store']}")
    print(f"Date: {result['date']}")
    print(f"Total: Rp {result['total']:,.0f}")

    print("\nItems:")
    for item in result['items']:
        print(f"  - {item['name']} x{item['qty']} @ Rp {item['price']:,.0f}")
else:
    print(f"Error: {result['error']}")
```

### JavaScript (Node.js)

```javascript
const FormData = require("form-data");
const fs = require("fs");
const fetch = require("node-fetch");

const form = new FormData();
form.append("image", fs.createReadStream("receipt.jpg"));

fetch("http://localhost:5000/api/predict", {
  method: "POST",
  body: form,
})
  .then((res) => res.json())
  .then((data) => {
    if (data.success) {
      console.log("Store:", data.store);
      console.log("Date:", data.date);
      console.log("Total:", data.total);
      console.log("Items:", data.items);
    } else {
      console.error("Error:", data.error);
    }
  });
```

### cURL

```bash
# Basic request
curl -X POST http://localhost:5000/api/predict \
  -F "image=@receipt.jpg"

# With pretty print (jq)
curl -X POST http://localhost:5000/api/predict \
  -F "image=@receipt.jpg" | jq .

# Save to file
curl -X POST http://localhost:5000/api/predict \
  -F "image=@receipt.jpg" -o result.json
```

---

## ⚙️ Configuration

### Model Selection

API secara otomatis menggunakan model terbaik yang tersedia:

1. **Classifier V4** (context-aware) - jika ada di `models/classifier_v4/`
2. **Classifier V3** - jika ada di `models/classifier_v3/`
3. **Classifier V2** - jika ada di `models/classifier_v2/`
4. **Online Learning Model** - fallback

### Port Configuration

Edit di `simple_app.py`:

```python
app.run(debug=False, host='0.0.0.0', port=5000)  # Change port here
```

---

## 🧪 Testing

### Test dengan gambar struk

```bash
# Test single image
curl -X POST http://localhost:5000/api/predict \
  -F "image=@clean_dataset/test/fullDataset_870.jpg"

# Test multiple images
for img in clean_dataset/test/*.jpg; do
  echo "Testing: $img"
  curl -X POST http://localhost:5000/api/predict \
    -F "image=@$img" | jq '.store, .total'
done
```

### Expected Processing Time

- **OCR:** ~1-3 seconds
- **Classification:** ~0.1-0.5 seconds
- **Extraction:** ~0.1 seconds
- **Total:** ~2-4 seconds per receipt

---

## 🐛 Troubleshooting

### "No text detected"

- Gambar terlalu blur atau gelap
- Resolusi terlalu rendah
- Coba preprocess dulu (brightness, contrast)

### Low accuracy

- Model belum trained dengan data yang mirip
- Struk format baru/unik
- Perlu fine-tuning model

### Slow response

- GPU not available (using CPU)
- Image resolution terlalu tinggi
- Optimize image size before upload

---

## 📝 Notes

- API ini **stateless** - tidak menyimpan history
- Setiap request independent
- Model loaded sekali saat startup
- Thread-safe untuk concurrent requests
- Max image size: 10MB (bisa diatur di Flask)

---

## 🔄 API Changes from Previous Version

### Removed Endpoints

- ❌ `/api/process` - merged into `/api/predict`
- ❌ `/api/extract/simple` - merged into `/api/predict`
- ❌ `/api/extract/full` - removed (too complex)
- ❌ All debug/verbose endpoints

### Simplified Response

- ✅ Only essential fields: `success`, `store`, `date`, `items`, `total`
- ✅ Clean item format: `name`, `qty`, `price`
- ✅ No bbox, confidence, atau metadata lainnya
- ✅ Error response dengan format konsisten

---

Made with ❤️ by OCR FinSight Team
