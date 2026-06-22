"""
Transaction Category Classifier
================================
Klasifikasi transaksi keuangan Indonesia ke kategori:
makanan, belanja, hiburan, transportasi, tagihan, kesehatan, lainnya, dll.

Pipeline:
1. Load & merge dataset
2. Augmentasi data sintetik (merchant lokal, typo, singkatan)
3. Feature engineering (text preprocessing, encoding, normalisasi)
4. Train TF BiLSTM classifier
5. Evaluate & save model

Usage:
    wsl python3 scripts/train_transaction_classifier.py
    wsl python3 scripts/train_transaction_classifier.py --augment 5000 --epochs 30
"""

import os
import sys
import json
import random
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Config
# ============================================================
DATA_DIR = PROJECT_ROOT / "Dataset Transaction Classifier"
OUTPUT_DIR = PROJECT_ROOT / "models" / "transaction_classifier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ['makanan', 'belanja', 'hiburan', 'transportasi', 'tagihan', 'kesehatan', 'lainnya']
MAX_LEN = 30       # max token length
VOCAB_SIZE = 5000  # vocabulary size
EMBED_DIM = 64
LSTM_UNITS = 128
DROPOUT = 0.3

# ============================================================
# 1. Augmentation Data
# ============================================================

# Merchant lokal Indonesia
MERCHANT_PREFIXES = {
    'makanan': [
        'warung', 'rm', 'resto', 'kedai', 'mie', 'nasi', 'bakso', 'soto',
        'warteg', 'kantin', 'cafe', 'kopi', 'boba', 'pizza', 'burger',
        'ayam', 'bebek', 'seafood', 'padang', 'sunda', 'jawa', 'mcd',
        'kfc', 'hokben', 'indomaret', 'alfamart', 'grab food', 'gofood',
        'shopee food', 'starbucks', 'jco', 'dunkin', 'chatime', 'xing fu tang'
    ],
    'belanja': [
        'shopee', 'tokopedia', 'lazada', 'blibli', 'tiktok shop', 'bukalapak',
        'indomaret', 'alfamart', 'hypermart', 'carrefour', 'giant', 'lottemart',
        'ace hardware', 'ikea', 'h&m', 'zara', 'uniqlo', 'miniso', 'daiso',
        'guardian', 'watson', 'century', 'apotek', 'toko', 'pasar'
    ],
    'transportasi': [
        'grab', 'gojek', 'maxim', 'indriver', 'ojol', 'ojek', 'taxi',
        'blue bird', 'express', 'transjakarta', 'mrt', 'lrt', 'kereta',
        'kai', 'damri', 'bus', 'angkot', 'bensin', 'pertamina', 'shell',
        'spbu', 'parkir', 'tol', 'jasa marga'
    ],
    'tagihan': [
        'pln', 'listrik', 'pdam', 'air', 'telkom', 'indihome', 'firstmedia',
        'biznet', 'myrepublic', 'telkomsel', 'xl', 'indosat', 'tri', 'smartfren',
        'bpjs', 'cicilan', 'kpr', 'angsuran', 'iuran', 'sewa', 'kos', 'kontrakan'
    ],
    'kesehatan': [
        'rs', 'rumah sakit', 'klinik', 'puskesmas', 'dokter', 'apotek',
        'kimia farma', 'guardian', 'century', 'k24', 'halodoc', 'alodokter',
        'good doctor', 'lab', 'laboratorium', 'radiologi', 'optik', 'dental'
    ],
    'hiburan': [
        'netflix', 'spotify', 'youtube', 'disney+', 'vidio', 'viu', 'mola',
        'cgv', 'cinepolis', 'xxi', 'bioskop', 'karaoke', 'inul vizta',
        'timezone', 'amazone', 'wahana', 'taman', 'museum', 'konser',
        'steam', 'playstation', 'xbox', 'nintendo', 'mobile legend', 'game'
    ],
    'lainnya': [
        'atm', 'transfer', 'admin', 'biaya', 'fee', 'cashback', 'refund',
        'bca', 'bni', 'bri', 'mandiri', 'cimb', 'danamon', 'ovo', 'dana',
        'gopay', 'shopeepay', 'linkaja', 'qris', 'top up', 'pulsa'
    ]
}

# Typo patterns (common Indonesian typos)
TYPO_MAP = {
    'a': ['4', '@'],
    'e': ['3'],
    'i': ['1', 'y'],
    'o': ['0'],
    's': ['5', 'z'],
    'g': ['9'],
    'makan': ['mkn', 'mkan', 'makn'],
    'belanja': ['blnja', 'belnja', 'blj'],
    'bayar': ['byr', 'bayar', 'byr'],
    'transfer': ['trf', 'trnsfer', 'tf'],
    'tagihan': ['tghn', 'taghan'],
    'listrik': ['lstrik', 'litrik'],
    'bensin': ['bnsn', 'bnsin'],
}

# Singkatan umum
ABBREVIATIONS = {
    'rumah sakit': 'rs',
    'restoran': 'resto',
    'warung makan': 'warmak',
    'makanan': 'mkn',
    'minuman': 'minum',
    'transportasi': 'transport',
    'pembayaran': 'bayar',
    'pembelian': 'beli',
    'tagihan': 'tgh',
    'listrik': 'lstrik',
    'internet': 'inet',
    'telepon': 'telp',
}


def apply_typo(text: str, prob: float = 0.15) -> str:
    """Apply random typos to text."""
    words = text.split()
    result = []
    for word in words:
        if random.random() < prob and len(word) > 3:
            # Check word-level typo
            if word in TYPO_MAP:
                word = random.choice(TYPO_MAP[word])
            else:
                # Char-level typo
                chars = list(word)
                idx = random.randint(0, len(chars)-1)
                c = chars[idx].lower()
                if c in TYPO_MAP:
                    chars[idx] = random.choice(TYPO_MAP[c])
                word = ''.join(chars)
        result.append(word)
    return ' '.join(result)


def apply_abbreviation(text: str, prob: float = 0.2) -> str:
    """Apply abbreviations."""
    for full, abbr in ABBREVIATIONS.items():
        if full in text and random.random() < prob:
            text = text.replace(full, abbr)
    return text


def generate_synthetic(category: str, n: int) -> list:
    """Generate synthetic transaction descriptions for a category."""
    samples = []
    prefixes = MERCHANT_PREFIXES.get(category, MERCHANT_PREFIXES['lainnya'])

    templates = {
        'makanan': [
            '{prefix}', '{prefix} {loc}', 'beli {prefix}', 'makan {prefix}',
            'order {prefix}', '{prefix} delivery', 'jajan {prefix}',
            'sarapan {prefix}', 'makan siang {prefix}', 'makan malam {prefix}',
            'snack {prefix}', 'kopi {prefix}', 'minuman {prefix}'
        ],
        'belanja': [
            'beli di {prefix}', '{prefix}', 'belanja {prefix}',
            'pembelian {prefix}', 'order {prefix}', '{prefix} online',
            'beli {item}', 'belanja {item}', 'pembelian {item}'
        ],
        'transportasi': [
            '{prefix}', 'naik {prefix}', 'bayar {prefix}', '{prefix} online',
            'isi {prefix}', 'top up {prefix}', 'parkir {prefix}',
            'tol {prefix}', 'tiket {prefix}'
        ],
        'tagihan': [
            'bayar {prefix}', 'tagihan {prefix}', '{prefix}',
            'cicilan {prefix}', 'iuran {prefix}', 'sewa {prefix}',
            'pembayaran {prefix}', 'angsuran {prefix}'
        ],
        'kesehatan': [
            '{prefix}', 'berobat {prefix}', 'beli obat {prefix}',
            'konsultasi {prefix}', 'cek {prefix}', 'periksa {prefix}',
            'bayar {prefix}', 'kunjungan {prefix}'
        ],
        'hiburan': [
            '{prefix}', 'langganan {prefix}', 'subscribe {prefix}',
            'nonton {prefix}', 'main {prefix}', 'beli {prefix}',
            'top up {prefix}', 'bayar {prefix}'
        ],
        'lainnya': [
            '{prefix}', 'biaya {prefix}', 'admin {prefix}',
            'transfer {prefix}', 'top up {prefix}', 'bayar {prefix}',
            'fee {prefix}', 'cashback {prefix}'
        ]
    }

    items = ['baju', 'celana', 'sepatu', 'tas', 'elektronik', 'hp', 'laptop',
             'buku', 'alat tulis', 'perabot', 'mainan', 'kosmetik', 'skincare']
    locs = ['terdekat', 'mall', 'online', 'depan kantor', 'pinggir jalan']

    tmpl_list = templates.get(category, templates['lainnya'])

    for _ in range(n):
        prefix = random.choice(prefixes)
        tmpl = random.choice(tmpl_list)
        text = tmpl.format(
            prefix=prefix,
            item=random.choice(items),
            loc=random.choice(locs)
        )

        # Apply augmentation
        aug_type = random.random()
        if aug_type < 0.2:
            text = apply_typo(text)
        elif aug_type < 0.35:
            text = apply_abbreviation(text)
        elif aug_type < 0.45:
            text = text.upper()
        elif aug_type < 0.55:
            text = text.title()

        samples.append({'deskripsi': text, 'category': category, 'source': 'synthetic'})

    return samples


# ============================================================
# 2. Text Preprocessing
# ============================================================

def clean_text(text: str) -> str:
    """Clean and normalize transaction description."""
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    # Remove special chars except spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove numbers standalone (keep if part of word)
    text = re.sub(r'\b\d+\b', '', text).strip()
    return text


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering pipeline."""
    df = df.copy()

    # Clean text
    df['text_clean'] = df['deskripsi'].apply(clean_text)

    # Text length features
    df['char_len'] = df['text_clean'].str.len()
    df['word_count'] = df['text_clean'].str.split().str.len()

    # Keyword features
    food_kw = ['makan', 'minum', 'kopi', 'resto', 'warung', 'food', 'cafe', 'snack']
    transport_kw = ['grab', 'gojek', 'ojek', 'bensin', 'parkir', 'tol', 'bus', 'kereta']
    bill_kw = ['tagihan', 'listrik', 'air', 'internet', 'cicilan', 'bayar', 'iuran']
    health_kw = ['dokter', 'obat', 'klinik', 'rs', 'apotek', 'kesehatan']

    df['has_food_kw'] = df['text_clean'].apply(lambda x: int(any(k in x for k in food_kw)))
    df['has_transport_kw'] = df['text_clean'].apply(lambda x: int(any(k in x for k in transport_kw)))
    df['has_bill_kw'] = df['text_clean'].apply(lambda x: int(any(k in x for k in bill_kw)))
    df['has_health_kw'] = df['text_clean'].apply(lambda x: int(any(k in x for k in health_kw)))

    return df


# ============================================================
# 3. TF Model
# ============================================================

def build_model(vocab_size: int, num_classes: int, max_len: int) -> 'tf.keras.Model':
    """Build BiLSTM text classifier."""
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(max_len,), name='text_input')

    # Embedding
    x = tf.keras.layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True)(inputs)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)

    # BiLSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=DROPOUT)
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(LSTM_UNITS // 2, dropout=DROPOUT)
    )(x)

    # Dense
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# ============================================================
# 4. Main Training Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', type=int, default=5000, help='Total synthetic samples to add')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-split', type=float, default=0.15)
    args = parser.parse_args()

    print("=" * 65)
    print("🏦 Transaction Category Classifier - Training Pipeline")
    print("=" * 65)

    # ── Step 1: Load data ──────────────────────────────────────
    print("\n📦 Step 1: Loading datasets...")
    dfs = []
    for f in DATA_DIR.glob("*.csv"):
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded {f.name}: {len(df)} rows")

    all_df = pd.concat(dfs, ignore_index=True)
    # Normalize categories
    all_df['category'] = all_df['category'].str.lower().str.strip()
    # Merge rare categories
    all_df['category'] = all_df['category'].replace({'kebugaran': 'kesehatan', 'travel': 'hiburan'})
    # Keep only known categories
    all_df = all_df[all_df['category'].isin(CATEGORIES)]
    all_df = all_df.drop_duplicates(subset=['deskripsi'])

    print(f"\n  Total unique rows: {len(all_df)}")
    print(f"  Category distribution:")
    for cat, cnt in all_df['category'].value_counts().items():
        print(f"    {cat:15s}: {cnt:4d}")

    # ── Step 2: Augmentation ───────────────────────────────────
    print(f"\n🔧 Step 2: Generating {args.augment} synthetic samples...")
    per_category = args.augment // len(CATEGORIES)
    synthetic_rows = []
    for cat in CATEGORIES:
        samples = generate_synthetic(cat, per_category)
        synthetic_rows.extend(samples)
        print(f"  Generated {len(samples)} samples for '{cat}'")

    synth_df = pd.DataFrame(synthetic_rows)
    combined_df = pd.concat([all_df[['deskripsi', 'category']], synth_df[['deskripsi', 'category']]], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  Total after augmentation: {len(combined_df)}")
    print(f"  Category distribution:")
    for cat, cnt in combined_df['category'].value_counts().items():
        print(f"    {cat:15s}: {cnt:4d}")

    # ── Step 3: Feature Engineering ───────────────────────────
    print("\n⚙️  Step 3: Feature engineering...")
    combined_df = extract_features(combined_df)
    print(f"  Text cleaned, {combined_df['text_clean'].str.len().mean():.1f} avg chars")

    # ── Step 4: Tokenization ───────────────────────────────────
    print("\n🔤 Step 4: Tokenization...")
    import tensorflow as tf

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token='<OOV>',
        lower=True
    )
    tokenizer.fit_on_texts(combined_df['text_clean'])

    sequences = tokenizer.texts_to_sequences(combined_df['text_clean'])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=MAX_LEN, padding='post', truncating='post'
    )

    # Label encoding
    label2idx = {cat: i for i, cat in enumerate(CATEGORIES)}
    idx2label = {i: cat for cat, i in label2idx.items()}
    labels = combined_df['category'].map(label2idx).values
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=len(CATEGORIES))

    print(f"  Vocab size: {len(tokenizer.word_index)}")
    print(f"  Sequence shape: {padded.shape}")

    # ── Step 5: Train/Val/Test split ──────────────────────────
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        padded, labels_onehot, test_size=args.test_split, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    print(f"\n  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Step 6: Build & Train ─────────────────────────────────
    print(f"\n🚀 Step 5: Training ({args.epochs} epochs)...")
    model = build_model(VOCAB_SIZE, len(CATEGORIES), MAX_LEN)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / 'best_model.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=0
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # ── Step 7: Evaluate ──────────────────────────────────────
    print("\n📊 Step 6: Evaluation on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Test Loss:     {test_loss:.4f}")

    # Per-class accuracy
    y_pred = model.predict(X_test, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    print(f"\n  Per-class accuracy:")
    for i, cat in enumerate(CATEGORIES):
        mask = y_true_labels == i
        if mask.sum() > 0:
            acc = (y_pred_labels[mask] == i).mean()
            print(f"    {cat:15s}: {acc*100:.1f}% ({mask.sum()} samples)")

    # ── Step 8: Save ──────────────────────────────────────────
    print("\n💾 Step 7: Saving model & artifacts...")

    # Save tokenizer
    tokenizer_config = tokenizer.to_json()
    with open(OUTPUT_DIR / 'tokenizer.json', 'w') as f:
        f.write(tokenizer_config)

    # Save label mapping
    with open(OUTPUT_DIR / 'label_mapping.json', 'w') as f:
        json.dump({'label2idx': label2idx, 'idx2label': idx2label}, f, indent=2)

    # Save config
    config = {
        'max_len': MAX_LEN,
        'vocab_size': VOCAB_SIZE,
        'embed_dim': EMBED_DIM,
        'lstm_units': LSTM_UNITS,
        'categories': CATEGORIES,
        'test_accuracy': float(test_acc),
        'total_training_samples': len(X_train),
        'augmented_samples': len(synth_df),
    }
    with open(OUTPUT_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save training history
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(hist_data, f, indent=2)

    print(f"\n  Saved to: {OUTPUT_DIR}")
    print(f"  Files: best_model.keras, tokenizer.json, label_mapping.json, config.json")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"  Test Accuracy  : {test_acc*100:.2f}%")
    print(f"  Total samples  : {len(combined_df)}")
    print(f"  Synthetic added: {len(synth_df)}")
    print(f"  Categories     : {len(CATEGORIES)}")
    print(f"  Best model     : {OUTPUT_DIR}/best_model.keras")

    # Quick inference test
    print(f"\n🧪 Quick inference test:")
    test_cases = [
        "makan siang warteg",
        "grab car ke kantor",
        "bayar tagihan listrik pln",
        "netflix subscription",
        "beli baju di shopee",
        "kunjungan dokter umum",
        "transfer bca",
    ]
    test_seq = tokenizer.texts_to_sequences([clean_text(t) for t in test_cases])
    test_pad = tf.keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=MAX_LEN, padding='post')
    preds = model.predict(test_pad, verbose=0)

    for text, pred in zip(test_cases, preds):
        cat = CATEGORIES[np.argmax(pred)]
        conf = np.max(pred)
        print(f"  '{text}' → {cat} ({conf*100:.1f}%)")


if __name__ == '__main__':
    main()
