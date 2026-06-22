# Transaction Category Classifier

Model klasifikasi otomatis kategori pengeluaran berdasarkan deskripsi teks transaksi (item name / store name dari hasil OCR).

---

## Final ML Pipeline

```
┌──────────────────┐
│   Gambar Struk   │  JPG / PNG
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────┐
│         PREPROCESSING            │
│  Deskew · CLAHE · Sharpen · 2×   │
│  src/preprocessing.py            │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│           PaddleOCR              │
│  Ekstraksi teks per baris        │
│  src/ocr_engine_paddle.py        │
│                                  │
│  Output:                         │
│  [{text, bbox, confidence}, ...] │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│   ReceiptLineClassifier (V5)     │
│  Klasifikasi tiap baris OCR      │
│  src/model_v4_context.py         │
│                                  │
│  Input : text + position feats   │
│  Output: label per baris         │
│  12 kelas:                       │
│  STORE · DATE · ITEM_DESC        │
│  ITEM_PRICE/QTY · SUBTOTAL       │
│  TAX · DISCOUNT · SERVICE_CHARGE │
│  GRAND_TOTAL · CASH_PAYMENT      │
│  ADDRESS_CONTACT · OTHER         │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│        ReceiptExtractor          │
│  Ekstraksi data terstruktur      │
│  src/extractor.py                │
│                                  │
│  Output JSON:                    │
│  {                               │
│    "store": "Indomaret",         │
│    "date": "18/06/2023",         │
│    "items": [                    │
│      {"name":"Indomie",          │
│       "qty":1,"price":3500}      │
│    ],                            │
│    "total": 169400.0             │
│  }                               │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│ TransactionCategoryClassifier ◄──┼── MODEL INI
│  BiLSTM text classifier          │
│  models/transaction_classifier/  │
│                                  │
│  Input : deskripsi (string)      │
│  Output: category + confidence   │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│         Final Response           │
│  {                               │
│    "store": "Indomaret",         │
│    "date": "18/06/2023",         │
│    "items": [...],               │
│    "total": 169400.0,            │
│    "category": "makanan",        │
│    "confidence": 0.94            │
│  }                               │
└──────────────────────────────────┘
```

---

## Model: TransactionCategoryClassifier

### Input

Field `deskripsi` berupa string teks pendek. Diambil dari output ReceiptExtractor dengan strategi berikut:

| Sumber OCR | Contoh nilai | Dipakai sebagai deskripsi |
|---|---|---|
| `items[0].name` | `"Indomie Goreng"` | `"indomie goreng"` |
| `store` | `"Indomaret"` | `"indomaret"` |
| Gabungan | store + semua item | `"indomaret indomie goreng susu ultra"` |
| Manual | Input user langsung | `"makan siang warteg pak budi"` |

```python
# Contoh input ke model
deskripsi = "indomaret indomie goreng susu ultra"
# atau
deskripsi = "grab car ke kantor"
# atau
deskripsi = "bayar tagihan listrik pln"
```

### Output

```json
{
  "category": "makanan",
  "confidence": 0.94,
  "is_anomaly": false,
  "anomaly_score": 0.02
}
```

### 7 Kategori

| Label | Contoh deskripsi |
|---|---|
| `makanan` | warung nasi, kfc, grab food, kopi, indomie |
| `belanja` | shopee, indomaret, alfamart, baju, sepatu |
| `transportasi` | grab car, gojek, bensin, parkir, kereta |
| `tagihan` | listrik pln, tagihan air, cicilan, indihome |
| `kesehatan` | apotek, dokter, klinik, obat, rs |
| `hiburan` | netflix, bioskop, spotify, karaoke, game |
| `lainnya` | transfer, admin bank, top up, refund |

---

## Arsitektur Model

```
Input: deskripsi (string)
  ↓
Tokenizer (vocab 5000, max_len 30)
  ↓
Embedding(5000, 64)
  ↓
SpatialDropout1D(0.2)
  ↓
BiLSTM(128, return_sequences=True)
  ↓
BiLSTM(64)
  ↓
Dense(128, relu) → Dropout(0.3)
  ↓
Dense(64, relu)  → Dropout(0.3)
  ↓
Dense(7, softmax)
  ↓
Output: probabilities [7 kelas]
```

**Performa model yang sudah ditraining sebelumnya:**
- Test Accuracy: **93.56%**
- Val Accuracy: **94.46%** (peak)
- Training samples: 6,171 (+ 7,000 sintetik augmentasi)
- Epochs: 15 (early stopping)

---

## Dataset

| File | Baris | Kelas | Keterangan |
|---|---|---|---|
| `data/dataset_eda_final.csv` | 3,578 | 7 | Dataset utama — sudah di-clean & di-map ke 7 kategori final |
| `data/df_synthetic.csv` | 2,100 | 7 | Data sintetik Indonesia — perfectly balanced (300/kelas) |
| `data/df_real.csv` | 74 | 7 | Data real berlabel — untuk validasi |

Dataset di-merge dan di-augmentasi dengan data sintetik tambahan (merchant lokal, typo, singkatan) sebelum training.

---

## Artifacts yang Dihasilkan Setelah Training

```
models/transaction_classifier/
  best_model.keras       ← model weights (load untuk inference)
  tokenizer.json         ← Keras Tokenizer config
  label_mapping.json     ← {label2idx, idx2label}
  config.json            ← hyperparameter & metadata
  training_history.json  ← history loss/accuracy per epoch
```

---

## Cara Training Ulang (Google Colab)

1. Buka `notebook/train_category_classifier.ipynb` di Google Colab
2. Upload 3 file CSV dari folder `data/` ke Colab (atau mount Google Drive)
3. Jalankan semua cell secara berurutan
4. Download artifacts dari `models/transaction_classifier/` setelah training selesai
5. Letakkan artifacts di `finsight-ml/models/transaction_classifier/`

---

## Inference (setelah model di-deploy ke finsight-ml)

```python
# Di web/api_v2.py — dipanggil setelah ReceiptExtractor
from src.transaction_classifier import TransactionClassifier

classifier = TransactionClassifier()

# Dari output extractor:
store = extracted.get("store", "")
items = [item["name"] for item in extracted.get("items", [])]
deskripsi = " ".join([store] + items).strip().lower()

result = classifier.predict(deskripsi)
# → {"category": "makanan", "confidence": 0.94}
```
