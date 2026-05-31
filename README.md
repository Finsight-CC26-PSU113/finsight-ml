# OCR FinSight 2.0 — Receipt OCR & Information Extraction

ML pipeline untuk membaca struk belanja (Malaysia & Indonesia) menggunakan fine-tuned EasyOCR + BiLSTM classifier.

📖 **[API Documentation → docs/API.md](docs/API.md)**

---

## Architecture

```
Image → EasyOCR (text + bbox) → BiLSTM Classifier (label per line) → Extractor → JSON
```

**ML Pipeline:**

- **EasyOCR** — deteksi teks + bounding box, fine-tuned pada 33K crops struk
- **BiLSTM Classifier V2** — klasifikasi 7 label per baris (85% acc, DATE recall 96%)
- **Extractor** — regex + rule-based untuk ekstrak store, date, items, total

---

## Quick Start

### Install

```bash
# Python 3.10–3.12, Windows/Linux/macOS (no WSL required)
python setup.py            # CPU mode (works everywhere)
python setup.py --gpu      # GPU mode (NVIDIA + CUDA 12.1)
```

### Run API (FastAPI)

```bash
python web/api_v2.py
# → http://localhost:8000/api/docs  (Swagger UI)
# → POST /api/scan                  (scan receipt)
```

### Run Demo UI (Flask)

```bash
python web/simple_app.py
# → http://localhost:5000
```

### Docker

```bash
docker-compose up
# → http://localhost:8000
```

---

## Project Structure

```
src/
├── config.py           # Paths & hyperparameters
├── ocr_engine.py       # EasyOCR wrapper (auto GPU/CPU)
├── model.py            # BiLSTM classifier architecture
├── extractor.py        # Structured data extraction
├── preprocessing.py    # Image preprocessing (deskew, denoise)
├── text_cleaner.py     # OCR text normalization
├── text_post_processor.py  # Post-processing rules
├── online_learning.py  # Post-processing label rules
└── auto_evaluation.py  # Performance monitoring

web/
├── api_v2.py           # FastAPI production backend  ← MAIN API
└── simple_app.py       # Flask demo UI

scripts/
├── generate_combined_groundtruth.py  # Gemini labeling (cls + OCR)
├── finetune_easyocr.py               # Fine-tune EasyOCR CRNN
├── train_classifier_v2.py            # Train BiLSTM classifier
└── train_transaction_classifier.py   # Transaction category classifier

models/
├── classifier_v2/best_weights.weights.h5   # BiLSTM V2 (2.7MB)
├── best_weights.weights.h5                 # BiLSTM V1 (online learning)
├── online_model.h5                         # Online learning weights
└── finetuned_easyocr/best_model.pth        # Fine-tuned EasyOCR

docs/
└── API.md              # Full API documentation
```

---

## Models

| Model                | File                                           | Accuracy           | Notes                       |
| -------------------- | ---------------------------------------------- | ------------------ | --------------------------- |
| BiLSTM Classifier V2 | `models/classifier_v2/best_weights.weights.h5` | **85%** (DATE 96%) | Trained on 44K lines, MY+ID |
| Fine-tuned EasyOCR   | `models/finetuned_easyocr/best_model.pth`      | **81.1% CER**      | 33K crops                   |
| BiLSTM V1 (fallback) | `models/best_weights.weights.h5`               | 92.95%             | Older dataset               |

### Label Classes

| Label             | Description                    |
| ----------------- | ------------------------------ |
| `STORE`           | Nama toko/bisnis               |
| `ADDRESS_CONTACT` | Alamat, telp, email, NPWP      |
| `DATE`            | Tanggal & waktu (semua format) |
| `ITEM_DESC`       | Nama barang/menu               |
| `ITEM_PRICE/QTY`  | Harga & jumlah                 |
| `TOTAL_PAYMENT`   | Total, subtotal, pajak, diskon |
| `OTHER`           | Lainnya                        |

---

## API

Full docs: **[docs/API.md](docs/API.md)**

```bash
# Scan receipt
curl -X POST http://localhost:8000/api/scan \
     -F "image=@receipt.jpg"

# Health check
curl http://localhost:8000/api/health
```

Response:

```json
{
  "store": "Ichiban Sushi",
  "date": "Aug 19, 2024 6:32:54 PM",
  "items": [{"name": "Beef Teriyaki Ramen", "qty": 1, "price": 42000}],
  "total": 257565.0,
  "totals": {"grand_total": 257565.0, "subtotal": 223000.0, "tax": 23415.0, ...}
}
```

---

## Training New Data

```bash
# 1. Label dataset dengan Gemini Vision (1 API call = classifier + OCR data)
python scripts/generate_combined_groundtruth.py \
    --api-keys KEY1 KEY2 KEY3 \
    --model gemini-3.1-flash-lite \
    --max-images 500

# 2. Train BiLSTM classifier
wsl python3 scripts/train_classifier_v2.py --epochs 50

# 3. Fine-tune EasyOCR
wsl python3 scripts/finetune_easyocr.py --epochs 20
```

---

## Performance

| Component            | Metric        | Value        |
| -------------------- | ------------- | ------------ |
| BiLSTM Classifier V2 | Test Accuracy | **85%**      |
| BiLSTM Classifier V2 | DATE Recall   | **96.4%**    |
| EasyOCR (fine-tuned) | Char Accuracy | **81.1%**    |
| Inference (CPU)      | Per image     | ~2–3 seconds |
| RAM (inference)      | Total         | ~2 GB        |

---

## Deployment

### Minimum VPS Spec

| Resource | Minimum       | Recommended   |
| -------- | ------------- | ------------- |
| RAM      | 4 GB          | 8 GB          |
| CPU      | 2 vCPU        | 4 vCPU        |
| Storage  | 10 GB SSD     | 20 GB SSD     |
| OS       | Ubuntu 22.04+ | Ubuntu 22.04+ |

### Providers (4 GB RAM)

| Provider     | Price/month  |
| ------------ | ------------ |
| IDCloudHost  | ~Rp 200–250k |
| Vultr        | ~$20         |
| DigitalOcean | ~$24         |
| Railway      | ~$10–15      |

---

## Troubleshooting

**`numpy._ARRAY_API not found` (Windows)**

```bash
pip install "numpy<2.0" --force-reinstall
# atau
python setup.py
```

**GPU not detected**

```bash
# App otomatis fallback ke CPU — tidak perlu action
# Untuk aktifkan GPU:
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

---

_OCR FinSight 2.0 — Capstone Project CC26-PSU113_
