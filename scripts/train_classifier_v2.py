"""
Train BiLSTM Receipt Line Classifier (V2)
==========================================
Dilatih dari labels.csv yang dihasilkan oleh generate_classification_groundtruth.py.

CSV expected format:
  source, filename, line_index, text, label,
  ocr_confidence, x_min, y_min, x_max, y_max,
  y_center, x_center, width, height

Usage:
  wsl python3 scripts/train_classifier_v2.py
  wsl python3 scripts/train_classifier_v2.py --epochs 80 --batch-size 64
"""

import os
import sys
import argparse
import time
import re
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    LINE_CLASSES, NUM_CLASSES, MAX_TEXT_LENGTH,
    EMBEDDING_DIM, LSTM_UNITS, DENSE_UNITS, DROPOUT_RATE,
    LEARNING_RATE, BATCH_SIZE, EPOCHS, RANDOM_SEED, ROOT_DIR
)
from src.model import ReceiptLineClassifier, FocalLoss


# ============================================================
# Feature extraction (pure-python, no external deps)
# ============================================================

DATE_REGEX = re.compile(
    r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})|'           # 24/05/2026, 26-11-17
    r'(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})|'             # 2026-05-24
    r'(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|'
    r'januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)'
    r'\s+\d{2,4})|'                                       # 14 Mar 2025, 14 Maret 2025
    r'((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}[\s,]+\d{2,4})',  # Mar 14 2025
    re.IGNORECASE
)
TIME_REGEX = re.compile(r'\d{1,2}:\d{2}(:\d{2})?')
QTY_REGEX = re.compile(r'\d+\s*[xX\*@]\s*[\d.,]+|@\s*[\d.,]+')
CURRENCY_REGEX = re.compile(r'(rm|rp|idr|usd|sgd|sr|\$|€)\s*[\d.,]+', re.IGNORECASE)


def extract_text_features(text: str) -> list[float]:
    """Extract 10-dim text features.
    
    Returns:
        [digit_ratio, alpha_ratio, upper_ratio, space_ratio, special_ratio,
         char_count, word_count, has_currency, has_date, has_qty]
    """
    if not text:
        return [0.0] * 10
    
    text = str(text)
    n = len(text)
    digits = sum(c.isdigit() for c in text)
    alphas = sum(c.isalpha() for c in text)
    uppers = sum(c.isupper() for c in text)
    spaces = sum(c.isspace() for c in text)
    specials = n - digits - alphas - spaces
    
    return [
        digits / max(n, 1),
        alphas / max(n, 1),
        uppers / max(alphas, 1),
        spaces / max(n, 1),
        specials / max(n, 1),
        n / 100.0,                                    # char_count normalized
        len(text.split()) / 20.0,                     # word_count normalized
        1.0 if CURRENCY_REGEX.search(text) else 0.0,
        1.0 if (DATE_REGEX.search(text) or TIME_REGEX.search(text)) else 0.0,
        1.0 if QTY_REGEX.search(text) else 0.0,
    ]


def extract_position_features(row: pd.Series, line_count_per_image: dict) -> list[float]:
    """Extract 5-dim position features.
    
    Returns:
        [y_ratio, x_ratio, width_ratio, height_ratio, line_index_ratio]
    """
    total_lines = line_count_per_image.get(row['filename'], 1)
    return [
        float(row['y_center']),
        float(row['x_center']),
        float(row['width']),
        float(row['height']),
        float(row['line_index']) / max(total_lines, 1),
    ]


def text_to_char_ids(text: str, max_length: int = MAX_TEXT_LENGTH) -> np.ndarray:
    """Convert text to ASCII char ids (0=pad)."""
    if not text:
        text = ""
    text = str(text)
    ids = [min(ord(c), 127) for c in text[:max_length]]
    ids = ids + [0] * (max_length - len(ids))
    return np.array(ids, dtype=np.int32)


# ============================================================
# Data loading
# ============================================================

def load_data(csv_path: Path):
    """Load and prepare training data from CSV."""
    print(f"📂 Loading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Total rows: {len(df)}")
    
    # Drop missing/empty text
    df = df[df['text'].notna() & (df['text'].astype(str).str.len() > 0)].copy()
    df['text'] = df['text'].astype(str)
    
    # Map labels to indices
    label_to_idx = {v: k for k, v in LINE_CLASSES.items()}
    df = df[df['label'].isin(label_to_idx.keys())].copy()
    df['label_idx'] = df['label'].map(label_to_idx).astype(int)
    print(f"   After filtering: {len(df)}")
    
    print(f"\n📊 Label distribution:")
    for label, count in df['label'].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {label:>16s}: {count:>5d} ({pct:.1f}%)")
    
    # Compute total lines per image (for line_index_ratio)
    line_count_per_image = df.groupby('filename')['line_index'].max().to_dict()
    line_count_per_image = {k: v + 1 for k, v in line_count_per_image.items()}
    
    # Char IDs
    print("\n🔤 Encoding char ids...")
    char_ids = np.array([text_to_char_ids(t) for t in df['text'].values], dtype=np.int32)
    
    # Text features (10 dims)
    print("📝 Extracting text features...")
    text_features = np.array(
        [extract_text_features(t) for t in df['text'].values],
        dtype=np.float32
    )
    
    # Position features (5 dims)
    print("📐 Extracting position features...")
    pos_features = np.array(
        [extract_position_features(row, line_count_per_image) for _, row in df.iterrows()],
        dtype=np.float32
    )
    
    labels = tf.keras.utils.to_categorical(df['label_idx'].values, num_classes=NUM_CLASSES)
    
    return char_ids, text_features, pos_features, labels, df


def split_data(char_ids, text_features, pos_features, labels, df):
    """Stratified train/val/test split."""
    label_idx = df['label_idx'].values
    indices = np.arange(len(df))
    
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_SEED, stratify=label_idx
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=RANDOM_SEED,
        stratify=label_idx[temp_idx]
    )
    
    def take(idx):
        return (char_ids[idx], text_features[idx], pos_features[idx]), labels[idx]
    
    return take(train_idx), take(val_idx), take(test_idx)


def oversample_minorities(char_ids, text_features, pos_features, labels, target_ratio=0.5):
    """Duplicate minority classes to balance training data.
    
    target_ratio: minority class size = target_ratio * majority class size.
    """
    label_idx = np.argmax(labels, axis=1)
    counts = np.bincount(label_idx, minlength=NUM_CLASSES)
    max_count = counts.max()
    target = int(max_count * target_ratio)
    
    new_idx = []
    for c in range(NUM_CLASSES):
        cls_idx = np.where(label_idx == c)[0]
        if len(cls_idx) == 0:
            continue
        new_idx.append(cls_idx)
        if len(cls_idx) < target:
            need = target - len(cls_idx)
            sampled = np.random.choice(cls_idx, size=need, replace=True)
            new_idx.append(sampled)
    
    new_idx = np.concatenate(new_idx)
    np.random.shuffle(new_idx)
    
    return (char_ids[new_idx], text_features[new_idx], pos_features[new_idx], labels[new_idx])


# ============================================================
# Training
# ============================================================

def train(train_data, val_data, epochs: int, batch_size: int, lr: float):
    """Train the model with early stopping and class weights."""
    (train_chars, train_tf, train_pf), train_labels = train_data
    (val_chars, val_tf, val_pf), val_labels = val_data
    
    # Compute focal alpha weights from training distribution
    counts = np.sum(train_labels, axis=0)
    total = counts.sum()
    alpha = total / (NUM_CLASSES * (counts + 1))
    alpha = alpha / alpha.sum() * NUM_CLASSES
    
    print("\n🎯 Class weights (focal alpha):")
    for i, w in enumerate(alpha):
        print(f"   {LINE_CLASSES[i]:>16s}: {w:.4f}")
    
    # Build model
    print("\n🧠 Building model...")
    model = ReceiptLineClassifier()
    
    # Build with dummy input
    dummy_chars = tf.zeros((1, MAX_TEXT_LENGTH), dtype=tf.int32)
    dummy_tf = tf.zeros((1, 10), dtype=tf.float32)
    dummy_pf = tf.zeros((1, 5), dtype=tf.float32)
    _ = model((dummy_chars, dummy_tf, dummy_pf), training=False)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha.astype(np.float32))
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    model.summary()
    
    # Datasets
    train_ds = tf.data.Dataset.from_tensor_slices(
        ((train_chars, train_tf, train_pf), train_labels)
    ).shuffle(20000, seed=RANDOM_SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices(
        ((val_chars, val_tf, val_pf), val_labels)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Output paths
    out_dir = ROOT_DIR / "models" / "classifier_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "best_weights.weights.h5"
    log_dir = ROOT_DIR / "logs" / f"classifier_v2_{int(time.time())}"
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(weights_path),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]
    
    print(f"\n🚀 Training... ({epochs} epochs, batch {batch_size}, lr {lr})")
    print(f"   Logs: {log_dir}")
    print(f"   Best weights: {weights_path}\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Reload best weights
    model.load_weights(str(weights_path))
    print(f"\n✅ Training complete. Best weights at: {weights_path}")
    
    return model, history


def evaluate(model, test_data):
    """Evaluate on test set with classification report."""
    (test_chars, test_tf, test_pf), test_labels = test_data
    
    print("\n" + "=" * 65)
    print("📊 TEST EVALUATION")
    print("=" * 65)
    
    predictions = model((test_chars, test_tf, test_pf), training=False).numpy()
    pred_idx = np.argmax(predictions, axis=1)
    true_idx = np.argmax(test_labels, axis=1)
    
    # Overall
    acc = np.mean(pred_idx == true_idx)
    print(f"\nOverall test accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # Per-class report
    target_names = [LINE_CLASSES[i] for i in range(NUM_CLASSES)]
    print("\n" + classification_report(
        true_idx, pred_idx, target_names=target_names, digits=4, zero_division=0
    ))
    
    # Confusion matrix
    print("\n📊 Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(true_idx, pred_idx, labels=list(range(NUM_CLASSES)))
    
    header = "        " + " ".join(f"{name[:6]:>7s}" for name in target_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {target_names[i][:6]:>6s} " + " ".join(f"{v:>7d}" for v in row))
    
    return acc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to labels CSV (default: data/classification_groundtruth/labels.csv)')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--no-oversample', action='store_true', help='Skip oversampling')
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Detect GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🖥️  GPU detected: {len(gpus)} device(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"   ⚠️  Memory growth: {e}")
    else:
        print("🖥️  Running on CPU")
    
    csv_path = Path(args.csv) if args.csv else (ROOT_DIR / "data" / "combined_groundtruth" / "classification_labels.csv")
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        print("   Run: python scripts/generate_combined_groundtruth.py --api-keys YOUR_KEY")
        sys.exit(1)
    
    print("=" * 65)
    print("🏷️  Receipt Line Classifier V2 — Training")
    print("=" * 65)
    
    # Load data
    char_ids, text_features, pos_features, labels, df = load_data(csv_path)
    
    # Split before oversampling (so val/test are realistic)
    train_data, val_data, test_data = split_data(char_ids, text_features, pos_features, labels, df)
    print(f"\n📦 Split sizes:")
    print(f"   Train: {len(train_data[1])}")
    print(f"   Val:   {len(val_data[1])}")
    print(f"   Test:  {len(test_data[1])}")
    
    # Oversample TRAIN only
    if not args.no_oversample:
        print("\n🔄 Oversampling minority classes (train only)...")
        chars_o, tf_o, pf_o, lab_o = oversample_minorities(
            train_data[0][0], train_data[0][1], train_data[0][2], train_data[1]
        )
        train_data = ((chars_o, tf_o, pf_o), lab_o)
        print(f"   New train size: {len(lab_o)}")
    
    # Train
    model, history = train(train_data, val_data, args.epochs, args.batch_size, args.lr)
    
    # Evaluate
    evaluate(model, test_data)
    
    print("\n" + "=" * 65)
    print("✅ Done!")
    print(f"   Weights: models/classifier_v2/best_weights.weights.h5")
    print("   Use in app: update web/simple_app.py to load from classifier_v2")
    print("=" * 65)


if __name__ == '__main__':
    main()
