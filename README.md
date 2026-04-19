<div align="center">

# 🤖 Finsight — ML Service

**OCR & Transaction Classification Microservice**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv)](https://opencv.org)

> Bagian dari Capstone Project **Coding Camp 2026 powered by DBS Foundation**
> Team ID: **CC26-PSU113**

</div>

---

## 📖 Tentang

Repo ini berisi **ML Service** Finsight — microservice FastAPI yang menangani seluruh pipeline AI:

```
Foto Struk (JPEG/PNG)
       │
       ▼
┌─────────────────────┐
│  Image Preprocessing │  ← OpenCV (grayscale, threshold, crop ROI, deskew)
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  OCR (CRNN + CTC)   │  ← Text detection & recognition
│  Tesseract fallback │  ← Jika akurasi CRNN rendah
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  NER Parser         │  ← Ekstrak: tanggal, merchant, total, item
│  (Rule-based)       │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  TF Transaction     │  ← TensorFlow classifier
│  Classifier         │  ← Kategori: makanan, transportasi, hiburan, dst
└─────────────────────┘
       │
       ▼
  Structured JSON → Backend
```

---

## 🗂️ Struktur Folder

```
finsight-ml-service/
├── app/
│   ├── main.py                # FastAPI entry point
│   ├── routers/
│   │   ├── ocr.py             # Endpoint /ocr/process
│   │   ├── classify.py        # Endpoint /classify/transaction
│   │   └── health.py          # Endpoint /model/health & /model/metrics
│   ├── services/
│   │   ├── preprocessing.py   # OpenCV pipeline
│   │   ├── ocr_service.py     # CRNN + Tesseract
│   │   ├── ner_parser.py      # Rule-based NER
│   │   └── classifier.py      # TF Transaction Classifier
│   ├── models/
│   │   └── saved_model/       # TF SavedModel (gitignored)
│   └── utils/
│       └── helpers.py
├── notebooks/                 # Eksperimen & training
├── tests/
├── requirements.txt
├── .env.example
└── Dockerfile
```

---

## ⚙️ Tech Stack

| Teknologi | Kegunaan |
|---|---|
| FastAPI | REST API framework |
| TensorFlow 2.x | Transaction Classifier (CRNN + Custom Training Loop) |
| OpenCV | Image preprocessing |
| Tesseract OCR | Fallback text recognition |
| Pillow | Image handling |
| tf.GradientTape | Custom training loop (gradient clipping + TensorBoard) |

---

## 🚀 Cara Menjalankan

### Prasyarat
- Python >= 3.10
- Tesseract OCR terinstall di sistem

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-ind

# macOS
brew install tesseract
```

### Langkah-langkah

```bash
# 1. Clone repo
git clone https://github.com/finsight-cc26/finsight-ml-service.git
cd finsight-ml-service

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env

# 5. Jalankan service
uvicorn app.main:app --reload --port 8000
```

Service akan berjalan di **http://localhost:8000**

Swagger docs: **http://localhost:8000/docs**

---

## 🌍 Environment Variables

```env
MODEL_PATH=app/models/saved_model
TESSERACT_PATH=/usr/bin/tesseract
OCR_CONFIDENCE_THRESHOLD=0.75
LOG_LEVEL=INFO
```

---

## 📋 API Endpoints

| Method | Endpoint | Deskripsi |
|---|---|---|
| POST | `/ocr/process` | Upload gambar → ekstrak & klasifikasi transaksi |
| POST | `/classify/transaction` | Klasifikasi teks transaksi (tanpa OCR) |
| GET | `/model/health` | Status model & service |
| GET | `/model/metrics` | Akurasi & performa model |

### Contoh Request — `/ocr/process`

```bash
curl -X POST http://localhost:8000/ocr/process \
  -F "file=@struk.jpg"
```

### Contoh Response

```json
{
  "success": true,
  "merchant": "Indomaret Cipagalo",
  "date": "2024-03-08",
  "total": 63000,
  "category": "makanan",
  "items": [
    { "name": "PIATTOS SAPI PNG 68G", "qty": 2, "price": 22400 },
    { "name": "MR BREAD TAWAR KUPAS", "qty": 1, "price": 16500 }
  ],
  "ocr_confidence": 0.91
}
```

---

## 🧠 Model

### TF Transaction Classifier
- Input: teks hasil parsing NER (merchant + items)
- Output: kategori transaksi (makanan, transportasi, hiburan, kesehatan, dll)
- Training: `tf.GradientTape` custom loop dengan gradient clipping
- Target akurasi: **≥ 85%**
- Monitoring: TensorBoard

### CRNN OCR
- Arsitektur: CNN + BiLSTM + CTC Decode
- Tesseract sebagai fallback jika confidence < threshold

---

## 📜 Scripts

| Command | Deskripsi |
|---|---|
| `uvicorn app.main:app --reload` | Dev server |
| `uvicorn app.main:app --port 8000` | Production |
| `python -m pytest tests/` | Jalankan tests |

---

## 🚢 Deployment

ML Service di-deploy ke **VPS terpisah** dari backend.

```bash
# Production dengan Gunicorn
pip install gunicorn
gunicorn app.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Atau via **Docker**:

```bash
docker build -t finsight-ml .
docker run -p 8000:8000 finsight-ml
```

---

## 👥 Maintainer

**Aidil Baihaqi** & **Muhammad Thesar** — AI Engineer
> Coding Camp 2026 | CC26-PSU113

