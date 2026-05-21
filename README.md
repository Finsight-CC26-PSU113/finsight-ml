# OCR FinSight 2.0 — Receipt OCR & Information Extraction

ML pipeline untuk membaca struk belanja (Malaysia & Indonesia) menggunakan fine-tuned EasyOCR + BiLSTM classifier dengan online learning.

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

## Project Structure

```
src/
├── config.py              # Centralized config
├── ocr_engine.py          # EasyOCR wrapper
├── model.py               # BiLSTM classifier architecture
├── extractor.py           # Regex-based value extraction
├── preprocessing.py       # Image preprocessing (deskew, denoise)
├── text_cleaner.py        # OCR text normalization
├── text_post_processor.py # Post-processing rules (O/0, I/1 fixes)
├── online_learning.py     # Online learning + auto-evaluation
└── auto_evaluation.py     # Automatic performance monitoring

web/
├── ocr_correction_ui.py   # Flask web server (main entry point)
└── templates/             # HTML templates

scripts/
├── train_model.py         # Train BiLSTM classifier
├── generate_ocr_groundtruth.py  # Generate OCR training data via Gemini
├── finetune_easyocr.py    # Fine-tune EasyOCR recognition model
└── test_real_case.py      # End-to-end pipeline test

models/
├── best_weights.weights.h5        # BiLSTM base weights
├── online_model.h5                # Online learning weights
└── finetuned_easyocr/
    ├── best_model.pth             # Fine-tuned EasyOCR (epoch 9)
    └── training_log.json          # Training history
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start web server
wsl python3 web/ocr_correction_ui.py

# Open browser
http://localhost:5000
```

---

## Fine-tuning EasyOCR (Optional)

```bash
# Step 1: Generate ground truth via Gemini Vision API
python3 scripts/generate_ocr_groundtruth.py \
    --api-key YOUR_GEMINI_KEY \
    --model gemini-3.1-flash-lite \
    --max-images 500

# Step 2: Fine-tune
python3 scripts/finetune_easyocr.py \
    --epochs 20 --batch-size 64 --lr 1e-4
```

---

## Performance

| Component           | Metric               | Value                               |
| ------------------- | -------------------- | ----------------------------------- |
| OCR (fine-tuned)    | Character Accuracy   | **81.1%**                           |
| OCR (fine-tuned)    | Exact Match per line | **58.8%**                           |
| Classifier          | Test Accuracy        | **92.95%**                          |
| Online Learning     | Training Variance    | **4.17%** (stable)                  |
| Fine-tuning Dataset | Samples              | 33,513 crops (Malaysia + Indonesia) |

---

## API Endpoints

| Endpoint                | Method | Description                             |
| ----------------------- | ------ | --------------------------------------- |
| `/`                     | GET    | Web UI                                  |
| `/api/init`             | POST   | Initialize with image                   |
| `/api/save_corrections` | POST   | Save user corrections + trigger retrain |
| `/api/stats`            | GET    | Model performance stats                 |

---

## Requirements

- Python 3.10+
- CUDA GPU (recommended, RTX 3050+ tested)
- EasyOCR 1.7.x
- TensorFlow 2.15+
- PyTorch 2.x

See `requirements.txt` for full list.

---

## Dataset

Training data tidak disertakan di repo ini (tersimpan di data scientist).

- **Classifier**: 44,101 labeled lines dari 967 struk Malaysia
- **OCR Fine-tuning**: 33,513 crops dari 620 struk (Malaysia + Indonesia)
- **Online Learning**: 1,363+ user corrections

---

_OCR FinSight 2.0 — Capstone Project CC26-PSU113_
