# 🧾 OCR FinSight

**AI-Powered Receipt OCR & Data Extraction System**

Sistem OCR canggih untuk mengekstrak data terstruktur dari foto struk/receipt Indonesia. Menggunakan PaddleOCR, TensorFlow classifier, dan rule-based extraction untuk akurasi tinggi.

---

## 📋 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Usage](#-api-usage)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

### Core Capabilities

- **📸 Receipt OCR**: Extract text from receipt photos with high accuracy
- **🏷️ Smart Classification**: 12-category line classifier (STORE, DATE, ITEMS, TOTAL, etc.)
- **💰 Structured Extraction**: Auto-extract store name, date, items, prices, totals
- **🇮🇩 Indonesia-Optimized**: Trained specifically on Indonesian receipts
- **⚡ Fast Processing**: ~1-2 seconds per receipt (CPU mode)
- **🎯 High Accuracy**: 86.26% overall, 82.65% on GRAND_TOTAL detection

### Technical Features

- **Context-Aware Classifier**: Uses ±4 lines context for better classification
- **Advanced Preprocessing**: Deskew, CLAHE, bilateral smoothing, 2x upscaling
- **Keyword-First Extraction**: Robust total detection with keyword priority
- **Web Interface**: Simple drag-and-drop UI for testing
- **REST API**: Easy integration with FastAPI
- **Production Ready**: Optimized for deployment

---

## 🔬 How It Works

### Pipeline Overview

```
📸 Image Input
    ↓
🎨 Preprocessing (deskew, CLAHE, upscale 2x)
    ↓
🔍 OCR (PaddleOCR) → Text + Bounding Boxes
    ↓
🏷️ Classification (TensorFlow V5 Context Model)
    ↓
💎 Extraction (Keyword + Rule-based)
    ↓
📊 Structured JSON Output
```

### Detailed Process

#### 1. **Image Preprocessing**

Transforms raw photo into OCR-optimized image:

```python
# Pipeline steps:
1. Deskew           → Fix tilted receipts (Hough line detection)
2. Bilateral Filter → Reduce noise, preserve edges
3. CLAHE            → Enhance contrast (LAB color space)
4. Light Sharpen    → Improve text clarity
5. Upscale 2x       → Make small text larger (critical!)
```

**Why this matters**: Receipt photos often have small text (<20px). Upscaling 2x improves OCR accuracy by 15-25%.

#### 2. **OCR Detection (PaddleOCR)**

Extracts text with bounding boxes:

```json
[
  {
    "text": "INDOMARET",
    "bbox": [
      [10, 20],
      [200, 20],
      [200, 45],
      [10, 45]
    ],
    "confidence": 0.98,
    "y_center": 0.05,
    "x_center": 0.3
  }
]
```

**Features**:

- Line-based merging (not word-by-word)
- Smart horizontal grouping
- Sorted top-to-bottom, left-to-right
- Post-processing (fix O→0, I→1, etc.)

#### 3. **Line Classification (TensorFlow V5)**

Classifies each line into 12 categories:

**Categories**:

1. `STORE` - Store name
2. `ADDRESS_CONTACT` - Address/phone
3. `DATE` - Transaction date
4. `ITEM_DESC` - Item name
5. `ITEM_PRICE/QTY` - Item price/quantity
6. `SUBTOTAL` - Subtotal before tax
7. `TAX` - Tax/PPN
8. `DISCOUNT` - Discounts
9. `SERVICE_CHARGE` - Service fees
10. `GRAND_TOTAL` - Final total to pay
11. `CASH_PAYMENT` - Cash received
12. `OTHER` - Unclassified

**Architecture**:

- **Context-Aware**: Uses ±4 lines context (8 neighboring lines)
- **Multi-Input**: Character embeddings + text features + position features + context
- **Training**: 11,495 Indonesia-only samples
- **Accuracy**: 86.26% overall, 82.65% on GRAND_TOTAL

**Why context matters**: "37,000" alone is ambiguous. But with context:

```
Subtotal 3 produk    [SUBTOTAL]
Biaya Tambahan       [SERVICE_CHARGE]
Total Tagihan        [GRAND_TOTAL] ← classifier knows this is grand total
Total Bayar          [CASH_PAYMENT]
```

#### 4. **Data Extraction (Rule-Based + Keywords)**

Extracts structured data using keyword-first strategy:

**Keyword Priority**:

```python
# GRAND_TOTAL (Total Tagihan)
"Total Tagihan"  → grand_total ✅  (HIGH priority)
"Grand Total"    → grand_total ✅
"Total"          → grand_total ✅  (only if no "bayar/tunai")

# CASH_PAYMENT (Uang yang Dibayar)
"Total Bayar"    → cash ✅  (NOT grand_total!)
"Tunai"          → cash ✅

# CHANGE (Kembalian)
"Kembalian"      → change ✅
```

**Why keyword-first?**: More robust than classifier alone. "Total Tagihan" always means grand_total, even if classifier misclassifies.

**Validation**:

- Grand total must be ≥ subtotal × 0.5
- Grand total must be ≤ subtotal × 2.5
- Tax must be ≤ items sum
- Reject suspicious patterns (NPWP, phone numbers)

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                       │
│                   (web/api_v2.py)                       │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Preprocessing│   │   OCR Engine │   │  Classifier  │
│    Module    │   │  (PaddleOCR) │   │  (TF V5)     │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │  Extractor   │
                   │ (Rule-based) │
                   └──────────────┘
                            │
                            ▼
                   📊 Structured JSON
```

### Model Files

```
models/
├── classifier_v5_indonesia/          ← Current production model
│   ├── best_weights.weights.h5       (TensorFlow weights)
│   ├── training_history.csv          (metrics)
│   └── config.json
│
├── finetuned_easyocr/                ← EasyOCR backup (optional)
│   └── best_model_compatible.pth
│
└── (V4, V3, V2 for fallback)
```

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **OS**: Windows, Linux, macOS
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/OCR-FinSight.git
cd OCR-FinSight
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This installs PaddleOCR (CPU version). For GPU support:

```bash
pip install paddlepaddle-gpu
```

### Step 4: Verify Installation

```bash
python -c "import paddle, paddleocr, tensorflow; print('✅ All modules loaded')"
```

---

## 🎯 Quick Start

### Option 1: Web Interface (Easiest)

1. **Start the server**:

```bash
python -m uvicorn web.api_v2:app --host 0.0.0.0 --port 8000 --reload
```

2. **Open browser**:

```
http://localhost:8000
```

3. **Upload receipt photo** and see results instantly!

### Option 2: API Endpoint

```python
import requests

# Upload image
with open("receipt.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/predict",
        files={"image": f}
    )

result = response.json()
print(result)
```

**Response**:

```json
{
  "success": true,
  "store": "Indomaret",
  "date": "14 Jan 23",
  "items": [
    {
      "name": "Ice Matcha",
      "qty": 1,
      "price": 10000.0
    }
  ],
  "total": 37000.0
}
```

### Option 3: Python Script

```python
from web.api_v2 import _run_pipeline
import asyncio

# Load image bytes
with open("receipt.jpg", "rb") as f:
    image_bytes = f.read()

# Run pipeline
result = asyncio.run(_run_pipeline(image_bytes))
print(result)
```

---

## 🌐 API Usage

### Endpoints

#### `GET /`

Web UI for testing

#### `GET /api/health`

Health check

**Response**:

```json
{
  "status": "ok",
  "models": {
    "ocr": true,
    "ocr_version": "PaddleOCR",
    "classifier": true,
    "classifier_version": "V5",
    "classifier_metrics": {
      "accuracy": 86.26,
      "grand_total_accuracy": 82.65,
      "dataset": "Indonesia-only"
    }
  }
}
```

#### `POST /api/predict`

Main OCR endpoint

**Request**:

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@receipt.jpg"
```

**Response**:

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
    }
  ],
  "total": 169400.0
}
```

### Error Handling

**400 Bad Request**:

```json
{
  "detail": "Invalid file type. Use JPG/PNG/WEBP."
}
```

**422 Unprocessable Entity**:

```json
{
  "detail": "No text detected in image."
}
```

**503 Service Unavailable**:

```json
{
  "detail": "OCR engine not ready"
}
```

---

## 🧠 Model Details

### Classifier V5 (Current Production)

**Training Dataset**:

- **Source**: Indonesia-only receipts
- **Size**: 11,495 samples
- **Split**: 80% train, 10% validation, 10% test
- **Categories**: 12 classes

**Architecture**:

```
Input: (text, position, context_lines)
  ↓
Character Embedding (dim=64)
  ↓
BiLSTM (128 units) + Dropout(0.3)
  ↓
Text Features (10 dims) + Position Features (5 dims)
  ↓
Context Features (±4 lines, 200 dims)
  ↓
Dense(64) + Dropout(0.3)
  ↓
Output: Softmax(12 classes)
```

**Training Config**:

- Optimizer: Adam (lr=0.001)
- Loss: Categorical crossentropy
- Class weights: GRAND_TOTAL=5x, SUBTOTAL=3.5x, TAX=3x
- Early stopping: patience=5
- Epochs: 50 (actual: 15-20 with early stop)

**Performance**:
| Metric | Value |
|--------|-------|
| Overall Accuracy | 86.26% |
| GRAND_TOTAL Accuracy | 82.65% |
| SUBTOTAL Accuracy | 88.5% |
| ITEM_DESC Accuracy | 89.2% |
| Processing Time | ~50ms |

**Why V5 is Better**:

- ✅ ±4 context window (V4 had ±2)
- ✅ Indonesia-only training (no Malaysia noise)
- ✅ Heavy class weight boosting
- ✅ Better handling of edge cases

---

## 📁 Project Structure

```
OCR-FinSight/
├── web/
│   ├── api_v2.py                    # FastAPI server (main entry)
│   └── templates_v2/
│       └── index_v2.html            # Web UI
│
├── src/
│   ├── ocr_engine_paddle.py         # PaddleOCR wrapper
│   ├── preprocessing.py             # Image preprocessing
│   ├── model_v4_context.py          # Classifier architecture
│   ├── extractor.py                 # Data extraction logic
│   ├── text_cleaner.py              # Text cleaning utilities
│   ├── classifier_corrector.py      # Rule-based corrections
│   └── config.py                    # Global config
│
├── models/
│   ├── classifier_v5_indonesia/     # Current production model
│   ├── classifier_v4_context/       # Backup model
│   └── finetuned_easyocr/          # EasyOCR model (optional)
│
├── scripts/
│   ├── train_classifier_v5_indonesia.py   # Training script
│   └── ...
│
├── data/
│   └── combined_groundtruth/
│       └── struk_indonesia_semua.csv      # Training data
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── API_SIMPLE.md                    # API documentation
```

---

## 🔧 Development

### Training New Classifier

```bash
python scripts/train_classifier_v5_indonesia.py
```

**Modify hyperparameters** in script:

```python
CONTEXT_WINDOW = 4         # ±4 lines context
EMBEDDING_DIM = 64
LSTM_UNITS = 128
LEARNING_RATE = 0.001
BATCH_SIZE = 32
```

### Testing Changes

```bash
# Start server with auto-reload
python -m uvicorn web.api_v2:app --reload

# Test with curl
curl -X POST "http://localhost:8000/api/predict" \
  -F "image=@test_receipt.jpg"
```

### Debugging

Enable debug output:

```python
# In web/api_v2.py or src/extractor.py
print(f"[DEBUG] Classified lines: {classified_lines}")
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'paddleocr'"

**Solution**:

```bash
pip install paddlepaddle paddleocr
```

### Problem: "Cannot import name 'inference' from 'paddle'"

**Solution**: Version mismatch. Install compatible versions:

```bash
pip install paddlepaddle==2.6.2 paddleocr==2.6.1.3
```

### Problem: OCR too slow (>5 seconds)

**Solutions**:

1. Reduce upscale factor in `src/preprocessing.py`:

```python
img = preprocess_for_easyocr(img, upscale_factor=1.5)  # Instead of 2.0
```

2. Use GPU (if available):

```python
# In src/config.py
OCR_GPU = True
```

3. Install GPU version:

```bash
pip install paddlepaddle-gpu
```

### Problem: Total detection wrong

**Debug steps**:

1. Check OCR output for "Total Tagihan" vs "Total Bayar"
2. Verify keywords in `src/extractor.py` line ~645
3. Check classifier predictions (should output debug info)

**Common fixes**:

- OCR typo: "Tota1" instead of "Total" → Post-processor should fix
- Wrong classification → Retrain with more examples
- Keyword not matched → Add to keyword list

### Problem: Items extraction includes "Kembalian" or "Total Bayar"

**Solution**: Already fixed! Check blacklist in `src/extractor.py` line ~180:

```python
blacklist_keywords = [
    'tagihan', 'bayar', 'kembalian', 'kembali',
    'total', 'subtotal', ...
]
```

---

## 🌐 Public Access (Tunneling)

### Using LocalTunnel

1. **Install localtunnel** (if not installed):

```bash
npm install -g localtunnel
```

2. **Start server** (in terminal 1):

```bash
python -m uvicorn web.api_v2:app --host 0.0.0.0 --port 8000
```

3. **Start tunnel** (in terminal 2):

```bash
lt --port 8000 --subdomain ocr-finsight
```

**Output**:

```
your url is: https://ocr-finsight.loca.lt
```

4. **Share the URL** - Anyone can access your OCR API!

**Note**: First visitor needs to enter tunnel password (shown in terminal).

### Alternative: ngrok

```bash
ngrok http 8000
```

---

## 📊 Performance Benchmarks

### OCR Speed (CPU)

| Image Size | Preprocessing | OCR  | Classification | Total |
| ---------- | ------------- | ---- | -------------- | ----- |
| 800x1200   | 80ms          | 1.2s | 50ms           | 1.33s |
| 1920x2880  | 120ms         | 2.5s | 80ms           | 2.7s  |

### Accuracy (Test Set, 1,149 receipts)

| Metric              | Value  |
| ------------------- | ------ |
| Store Name          | 92.3%  |
| Date                | 89.7%  |
| Items (≥1 detected) | 87.5%  |
| Grand Total         | 82.65% |
| Perfect Match       | 68.4%  |

### Resource Usage

- **RAM**: ~2GB (models loaded)
- **CPU**: 1 core @ 100% during inference
- **Storage**: ~500MB (models only)

---

## 📝 License

MIT License - see LICENSE file for details.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

## 🙏 Acknowledgments

- **PaddleOCR** by Baidu - State-of-the-art OCR engine
- **TensorFlow** - Deep learning framework
- **FastAPI** - Modern web framework
- **OpenCV** - Image processing

---

**Built with ❤️ for Indonesian fintech & retail**
