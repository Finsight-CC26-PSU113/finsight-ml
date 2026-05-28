# OCR FinSight 2.0 — Receipt OCR & Information Extraction

ML pipeline untuk membaca struk belanja (Malaysia & Indonesia) menggunakan fine-tuned EasyOCR + BiLSTM classifier.

---

## Architecture

```
Receipt Image
    ↓
[EasyOCR Fine-tuned]  → Extract text + bounding boxes
    ↓
[BiLSTM Classifier]   → Label each line (STORE/DATE/ITEM/TOTAL/etc)
    ↓
[Rule-based Extractor]→ Structured JSON output
    ↓
{store, date, items, total}
```

---

## Models

| Model              | File                                      | Accuracy      | Description                      |
| ------------------ | ----------------------------------------- | ------------- | -------------------------------- |
| BiLSTM Classifier  | `models/best_weights.weights.h5`          | **92.95%**    | Label classification (7 classes) |
| Online Learning    | `models/online_model.h5`                  | ~50-70%       | Updated via user corrections     |
| Fine-tuned EasyOCR | `models/finetuned_easyocr/best_model.pth` | **81.1% CER** | Receipt-specific OCR             |

### Label Classes

`STORE` · `ADDRESS_CONTACT` · `DATE` · `ITEM_DESC` · `ITEM_PRICE/QTY` · `TOTAL_PAYMENT` · `OTHER`

---

## Quick Start (Cross-Platform)

Bekerja di **Windows native, Linux, macOS, Docker** — tidak perlu WSL.

### Option 1: Automated Setup (Recommended)

```bash
# Python 3.10–3.12 required
python setup.py            # CPU mode (works everywhere)
python setup.py --gpu      # GPU mode (NVIDIA + CUDA 12.1)
python setup.py --check    # Verify existing installation
```

### Option 2: Manual Install

```bash
# Critical: install numpy<2.0 FIRST (compatibility)
pip install "numpy>=1.26,<2.0"

# Install rest
pip install -r requirements.txt
```

### Run the App

```bash
python web/simple_app.py
# → http://localhost:5000
```

---

## Docker Deployment (Production)

```bash
# Build
docker build -t ocr-finsight .

# Run
docker run -p 5000:5000 ocr-finsight

# Or with docker-compose
docker-compose up
```

The container uses CPU-only PyTorch and works on any Linux server (no GPU/CUDA needed).

---

## Project Structure

```
src/
├── config.py              # Centralized config
├── ocr_engine.py          # EasyOCR wrapper (auto GPU/CPU detection)
├── model.py               # BiLSTM classifier architecture
├── extractor.py           # Regex-based value extraction
├── preprocessing.py       # Image preprocessing
├── text_cleaner.py        # OCR text normalization
├── online_learning.py     # Inference + post-processing rules
└── auto_evaluation.py     # Performance monitoring

web/
├── simple_app.py          # Flask app (main entry — recommended)
├── app_v2.py              # Alternative Model V2 app
└── templates/             # HTML templates

scripts/
├── generate_combined_groundtruth.py   # Gemini-based labeling (1 call → cls + ocr)
├── generate_classification_groundtruth.py  # Classification-only labeling
├── generate_ocr_groundtruth.py        # OCR-only labeling
├── train_classifier_v2.py             # Train BiLSTM from new data
├── finetune_easyocr.py                # Fine-tune EasyOCR
└── test_real_case.py                  # End-to-end test

models/
├── best_weights.weights.h5            # BiLSTM base weights
├── online_model.h5                    # Online learning weights
└── finetuned_easyocr/best_model.pth   # Fine-tuned EasyOCR
```

---

## Generating Training Data (Optional)

Jika ingin label ulang dataset dengan kategori yang lebih akurat menggunakan Gemini Vision:

```bash
# Combined: classification + OCR ground truth in 1 API call (saves quota)
python scripts/generate_combined_groundtruth.py \
    --api-key YOUR_GEMINI_KEY \
    --max-images 500

# Train BiLSTM with new labels
python scripts/train_classifier_v2.py --epochs 50

# Fine-tune EasyOCR with new OCR pairs
python scripts/finetune_easyocr.py --epochs 20
```

Free tier Gemini quota: ~500 requests/day per API key.

---

## Performance

| Component        | Metric               | Value        |
| ---------------- | -------------------- | ------------ |
| OCR (fine-tuned) | Character Accuracy   | **81.1%**    |
| OCR (fine-tuned) | Exact Match per line | **58.8%**    |
| Classifier       | Test Accuracy        | **92.95%**   |
| Inference (CPU)  | Per image            | ~2–3 seconds |
| RAM (inference)  | Total                | ~2 GB        |

---

## Deployment Specifications

### Minimum VPS Spec (Production)

| Component   | Minimum       | Recommended                      |
| ----------- | ------------- | -------------------------------- |
| **RAM**     | **4 GB**      | 8 GB                             |
| **CPU**     | 2 vCPU        | 4 vCPU                           |
| **Storage** | 10 GB SSD     | 20 GB SSD                        |
| **GPU**     | Not required  | NVIDIA T4 (for faster inference) |
| **OS**      | Ubuntu 22.04+ | Ubuntu 22.04+ / Debian 12        |

> GPU is **not required**. CPU-only mode handles low-to-medium traffic fine.

### VPS Provider Pricing (4 GB RAM)

| Provider     | Spec                        | Price/month           |
| ------------ | --------------------------- | --------------------- |
| Vultr        | 4 GB RAM, 2 vCPU, 80 GB SSD | ~$20                  |
| DigitalOcean | 4 GB RAM, 2 vCPU, 80 GB SSD | ~$24                  |
| IDCloudHost  | 4 GB RAM, 2 vCPU            | ~Rp 200–250k          |
| Biznet Gio   | 4 GB RAM, 2 vCPU            | ~Rp 250–300k          |
| Railway      | 4 GB RAM, shared CPU        | ~$10–15 (pay per use) |

> For demo/capstone: **Railway** atau **Render** (gratis tier ada). Production: **IDCloudHost** atau **Vultr**.

---

## Troubleshooting

### `numpy._ARRAY_API not found` error

NumPy 2.x tidak compatible dengan opencv-python pre-built. Fix:

```bash
pip install "numpy<2.0" --force-reinstall
```

Atau jalankan `python setup.py` (otomatis pasang versi yang benar).

### `Could not find CUDA` / GPU not detected

Aplikasi otomatis fallback ke CPU. Untuk GPU support: install PyTorch CUDA build:

```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

### Port 5000 already in use

```bash
# Edit web/simple_app.py last line:
app.run(host='0.0.0.0', port=8080)
```

---

## Dataset (not in repo)

- **Classifier**: 44,101 labeled lines dari 967 struk Malaysia
- **OCR Fine-tuning**: 33,513 crops dari 620 struk (Malaysia + Indonesia)

Dataset disimpan terpisah oleh data scientist team.

---

_OCR FinSight 2.0 — Capstone Project CC26-PSU113_
