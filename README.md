# 🧾 OCR FinSight

**AI-Powered Receipt OCR & Data Extraction System**

Sistem OCR berbasis Deep Learning untuk mengekstrak data terstruktur dari foto struk belanja Indonesia. Menggunakan **PaddleOCR** untuk text detection, **BiLSTM Context-Aware Classifier** untuk line classification, dan **rule-based extraction** untuk parsing data.

## 📌 Project Overview

**OCR FinSight** adalah end-to-end solution untuk digitalisasi struk belanja dengan akurasi tinggi. Project ini mengimplementasikan:

- 🧠 **Context-Aware BiLSTM Classifier** (86.26% accuracy) - Trained on 11K+ Indonesia receipts
- 🔍 **PaddleOCR** - State-of-the-art text detection & recognition
- 🎨 **Advanced Image Preprocessing** - Deskew, CLAHE, upscaling 2x
- 💎 **Smart Keyword-First Extraction** - Robust total detection
- ⚡ **FastAPI REST API** - Production-ready deployment
- 🌐 **Web Interface** - Drag-and-drop testing UI

**Use Cases:**

- 💼 Expense tracking & reimbursement systems
- 📊 Financial analytics & insights
- 🏪 Retail analytics & inventory management
- 📱 Mobile receipt scanning apps

**Key Metrics:**

- **Overall Accuracy**: 86.26% (12-class classification)
- **Processing Speed**: ~1-2 seconds per receipt (CPU)
- **Model Size**: 278K parameters (1.06 MB)
- **Dataset**: 11,495 Indonesia-only receipts

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

## 🚀 Installation & Setup

### System Requirements

| Component   | Minimum                     | Recommended                        |
| ----------- | --------------------------- | ---------------------------------- |
| **Python**  | 3.10+                       | 3.11 or 3.12                       |
| **RAM**     | 4GB                         | 8GB                                |
| **Storage** | 2GB                         | 5GB (with datasets)                |
| **OS**      | Windows 10+ / Linux / macOS | Any                                |
| **GPU**     | Optional                    | NVIDIA CUDA (for faster inference) |

---

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/OCR-FinSight.git
cd OCR-FinSight
```

---

### Step 2: Setup Python Environment

#### **Option A: Using venv (Recommended)**

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows CMD:
venv\Scripts\activate.bat

# Windows PowerShell:
venv\Scripts\Activate.ps1

# Linux/macOS:
source venv/bin/activate
```

#### **Option B: Using conda**

```bash
conda create -n ocr-finsight python=3.11
conda activate ocr-finsight
```

---

### Step 3: Install Dependencies

#### **Basic Installation (CPU only)**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**What gets installed:**

- `tensorflow>=2.15.0` - Deep learning framework
- `paddlepaddle>=2.6.0` - OCR engine (CPU)
- `paddleocr>=2.6.1` - OCR toolkit
- `fastapi>=0.104.0` - Web framework
- `opencv-python>=4.8.0` - Image processing
- `numpy`, `pandas`, `scikit-learn` - Data processing

#### **GPU Installation (Optional, for faster inference)**

If you have NVIDIA GPU with CUDA:

```bash
# Install GPU version of PaddlePaddle
pip install paddlepaddle-gpu

# Verify GPU is detected
python -c "import paddle; print('GPU:', paddle.device.is_compiled_with_cuda())"
```

---

### Step 4: Download Pre-trained Models

The classifier models are **NOT included** in the repository due to file size. Download them separately:

#### **Option A: Automatic Download (Recommended)**

```bash
python scripts/download_models.py
```

This will download:

- ✅ Classifier V5 Indonesia (`best_weights.weights.h5`) - 1.06 MB
- ✅ Classifier V4 Context (backup) - 1.02 MB

#### **Option B: Manual Download**

**Download links:**

| Model                          | Size    | Link                                                                                                  | Accuracy |
| ------------------------------ | ------- | ----------------------------------------------------------------------------------------------------- | -------- |
| **Classifier V5 Indonesia** ⭐ | 1.06 MB | [📥 Google Drive](https://drive.google.com/file/d/1ubTfq-frSFCrQtbbyhsF6e_Ehrzf1JAF/view?usp=sharing) | 86.26%   |

**After download, place model here:**

```
models/
└── classifier_v5_indonesia/
    └── best_weights.weights.h5    ← Place downloaded file here
```

**Alternative direct download:**

```
https://drive.google.com/uc?export=download&id=1ubTfq-frSFCrQtbbyhsF6e_Ehrzf1JAF
```

**Note**: PaddleOCR models are auto-downloaded on first run (~10MB).

---

### Step 5: Verify Installation

```bash
# Test all dependencies
python -c "import paddle, paddleocr, tensorflow, fastapi; print('✅ All modules loaded successfully!')"

# Verify model files exist
python -c "from pathlib import Path; print('V5:', Path('models/classifier_v5_indonesia/best_weights.weights.h5').exists())"
```

**Expected output:**

```
✅ All modules loaded successfully!
V5: True
```

---

### Step 6: Run Quick Test

```bash
# Test OCR + Classifier on sample image
python scripts/test_pipeline.py --image test_receipts/sample1.jpg
```

If successful, you'll see:

```json
{
  "success": true,
  "store": "Indomaret",
  "date": "14 Jan 2023",
  "items": [...],
  "total": 63000.0
}
```

✅ **Installation complete!** Proceed to [Quick Start](#-quick-start) to run the app.

---

## 🎯 Quick Start

### Method 1: Web Interface (Easiest ⭐)

**Perfect for testing and demos!**

1. **Start the FastAPI server:**

```bash
# Method A: Using uvicorn directly
python -m uvicorn web.api_v2:app --host 0.0.0.0 --port 8000 --reload

# Method B: Using start script (Windows)
start_server.bat

# Method C: Using Python script
python web/api_v2.py
```

2. **Open browser:**

```
http://localhost:8000
```

3. **Upload receipt & see magic happen!** ✨

**Screenshot:**

```
┌─────────────────────────────────────────────────┐
│  OCR FinSight - Receipt Scanner                │
├─────────────────────────────────────────────────┤
│                                                 │
│  📷 [Drag and Drop Image Here]                 │
│      or click to upload                         │
│                                                 │
│  Supported: JPG, PNG, WEBP                     │
│  Max size: 10MB                                │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### Method 2: API Request (Python)

**For integration with other apps:**

```python
import requests

# Upload receipt image
with open("receipt.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/predict",
        files={"image": ("receipt.jpg", f, "image/jpeg")}
    )

# Parse response
result = response.json()

print(f"Store: {result['store']}")
print(f"Date: {result['date']}")
print(f"Total: Rp {result['total']:,.0f}")

for item in result['items']:
    print(f"  - {item['name']}: Rp {item['price']:,.0f}")
```

**Example Response:**

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
  "total": 116000.0
}
```

---

### Method 3: cURL (Command Line)

**For testing from terminal:**

```bash
# Basic request
curl -X POST "http://localhost:8000/api/predict" \
  -F "image=@receipt.jpg"

# Save response to file
curl -X POST "http://localhost:8000/api/predict" \
  -F "image=@receipt.jpg" \
  -o result.json

# Pretty print JSON (requires jq)
curl -X POST "http://localhost:8000/api/predict" \
  -F "image=@receipt.jpg" | jq .
```

---

### Method 4: Direct Python Script

**For batch processing or custom workflows:**

```python
import asyncio
from pathlib import Path
from web.api_v2 import _run_pipeline

async def process_receipt(image_path):
    # Load image
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Run OCR pipeline
    result = await _run_pipeline(image_bytes)

    return result

# Process single receipt
result = asyncio.run(process_receipt("receipt.jpg"))
print(result)

# Batch processing
receipts = Path("receipts/").glob("*.jpg")
for receipt in receipts:
    result = asyncio.run(process_receipt(receipt))
    print(f"{receipt.name}: Total Rp {result['total']:,.0f}")
```

---

### Troubleshooting First Run

#### Issue: "Address already in use"

**Solution:** Port 8000 is occupied. Use different port:

```bash
python -m uvicorn web.api_v2:app --port 8001
```

#### Issue: "Model file not found"

**Solution:** Download models first:

```bash
python scripts/download_models.py
```

Or check `models/classifier_v5_indonesia/best_weights.weights.h5` exists.

#### Issue: Server starts but no response

**Solution:** Check firewall settings:

```bash
# Allow port 8000
# Windows:
netsh advfirewall firewall add rule name="OCR FinSight" dir=in action=allow protocol=TCP localport=8000

# Linux:
sudo ufw allow 8000
```

#### Issue: "No text detected"

**Possible causes:**

- Image too blurry/dark
- Receipt text too small (<10px height)
- Wrong image format

**Solution:** Try preprocessing manually:

```bash
python scripts/preprocess_image.py --input blurry.jpg --output clean.jpg
```

---

### Next Steps

✅ Server running? Great! Now try:

1. **📖 Read [API Documentation](#-api-usage)** for detailed endpoint specs
2. **🧪 Test with sample receipts** in `test_receipts/` folder
3. **🔧 Customize extraction rules** in `src/extractor.py`
4. **📊 Train custom classifier** with your own data (see [Development](#-development))
5. **🌐 Deploy publicly** using ngrok/localtunnel (see [Public Access](#-public-access-tunneling))

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

### 📊 Monitoring Training with TensorBoard

The training script automatically logs all metrics to TensorBoard for visualization.

**View training progress in real-time:**

```bash
# Option 1: Use the convenience script
python scripts/view_tensorboard.py

# Option 2: Direct command
tensorboard --logdir=models/classifier_v5_indonesia/logs --port=6006
```

Then open browser to: **http://localhost:6006**

**Available Visualizations:**

1. **Scalars** - Track metrics over time:
   - Training loss & accuracy
   - Validation loss & accuracy
   - Learning rate changes
2. **Histograms** - Monitor weight distributions:
   - Layer activations
   - Weight updates per epoch
   - Gradient flow
3. **Graphs** - Visualize model architecture:
   - Complete computation graph
   - Layer connections
   - Input/output shapes

**TensorBoard Files:**

```
models/classifier_v5_indonesia/
├── logs/
│   └── train/
│       ├── events.out.tfevents.*   # TensorBoard event files
│       └── ...
├── training_history.csv             # CSV backup of metrics
└── best_weights.weights.h5          # Best model weights
```

**Tips:**

- Training logs persist across runs - old runs remain visible
- Compare multiple runs by changing log_dir in training script
- Export plots as SVG/PNG from TensorBoard UI
- Use "smoothing" slider to reduce noise in loss curves

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
