"""
Train Classifier V5 - INDONESIA ONLY with Context
===================================================
Training khusus data Indonesia untuk akurasi lebih baik pada struk Indonesia.

Improvements:
- Data: Hanya struk Indonesia (11K+ samples)
- Context: Tetap pakai ±2 lines context
- Class weights: Boosted untuk critical classes (GRAND_TOTAL, SUBTOTAL, TAX)
- Validation: Strict untuk GRAND_TOTAL

Usage:
    python scripts/train_classifier_v5_indonesia.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LINE_CLASSES, MAX_TEXT_LENGTH

# Override CONTEXT_WINDOW for better structure understanding
CONTEXT_WINDOW = 4  # Look at ±4 lines (was 2)

LINE_CLASSES_LIST = [LINE_CLASSES[i] for i in sorted(LINE_CLASSES.keys())]
NUM_CLASSES = len(LINE_CLASSES_LIST)

print("=" * 80)
print("CLASSIFIER V5 - INDONESIA ONLY (with Context)")
print("=" * 80)
print(f"\n📋 Classes ({NUM_CLASSES}): {', '.join(LINE_CLASSES_LIST)}")
print(f"🔍 Context window: {CONTEXT_WINDOW} lines before + {CONTEXT_WINDOW} lines after")
print(f"🇮🇩 Dataset: Indonesia receipts only")

# Load Indonesia data
csv_path = PROJECT_ROOT / "data" / "combined_groundtruth" / "struk_indonesia_semua.csv"
print(f"\n📂 Loading data from: {csv_path}")

df = pd.read_csv(csv_path)
print(f"✅ Loaded {len(df):,} samples (Indonesia only)")

# Filter valid labels
df = df[df['label'].isin(LINE_CLASSES_LIST)]
print(f"✅ After filtering: {len(df):,} samples")

# Add missing feature columns if not present
required_features = [
    'digit_ratio', 'alpha_ratio', 'upper_ratio', 'space_ratio', 'special_ratio',
    'char_count', 'word_count', 'has_currency', 'has_date_pattern', 'has_qty_pattern',
    'y_ratio', 'x_ratio', 'width_ratio', 'height_ratio', 'line_index_ratio'
]

# Generate features if missing
if 'digit_ratio' not in df.columns:
    print("\n🔧 Generating text features...")
    
    def extract_features(row):
        import re
        text = str(row['text']) if pd.notna(row['text']) else ""
        n = len(text)
        
        # Text features
        if n == 0:
            text_feats = [0.0] * 10
        else:
            digits = sum(c.isdigit() for c in text)
            alphas = sum(c.isalpha() for c in text)
            uppers = sum(c.isupper() for c in text)
            spaces = sum(c.isspace() for c in text)
            specials = n - digits - alphas - spaces
            has_currency = 1.0 if re.search(r'(rm|rp|idr|usd|\$|€)', text, re.I) else 0.0
            has_date = 1.0 if re.search(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', text) else 0.0
            has_qty = 1.0 if re.search(r'\d+\s*[xX\*@]', text) else 0.0
            
            text_feats = [
                digits / n, alphas / n, uppers / max(alphas, 1),
                spaces / n, specials / n, n / 100.0,
                len(text.split()) / 20.0,
                has_currency, has_date, has_qty
            ]
        
        # Position features (normalize if needed)
        y_ratio = float(row.get('y_center', 0.5))
        x_ratio = float(row.get('x_center', 0.5))
        width_ratio = float(row.get('width', 0.1))
        height_ratio = float(row.get('height', 0.05))
        line_idx_ratio = 0.5  # Will be calculated per group
        
        return text_feats + [y_ratio, x_ratio, width_ratio, height_ratio, line_idx_ratio]
    
    # Apply feature extraction
    features = df.apply(extract_features, axis=1, result_type='expand')
    feature_names = required_features
    for idx, name in enumerate(feature_names):
        df[name] = features[idx]
    
    # Fix line_index_ratio per filename group
    for filename, group in df.groupby('filename'):
        n_lines = len(group)
        df.loc[group.index, 'line_index_ratio'] = group['line_index'] / max(n_lines, 1)
    
    print("✅ Features generated")

print("\n📊 Label distribution:")
label_counts = df['label'].value_counts()
for label in LINE_CLASSES_LIST:
    count = label_counts.get(label, 0)
    pct = count / len(df) * 100 if len(df) > 0 else 0
    marker = " ⭐" if label in ['GRAND_TOTAL', 'SUBTOTAL', 'TAX', 'DISCOUNT'] else ""
    print(f"  {label:20s}: {count:>6,} ({pct:>5.1f}%){marker}")

# Feature extraction functions
def char_ids(text, max_len=MAX_TEXT_LENGTH):
    text = str(text) if pd.notna(text) else ""
    ids = [min(ord(c), 127) for c in text[:max_len]]
    return ids + [0] * (max_len - len(ids))

print("\n🔧 Building context-aware features...")

# Group by receipt to maintain structure
grouped = df.groupby('filename')

X_chars_list = []
X_text_list = []
X_pos_list = []
X_context_chars_list = []
X_context_text_list = []
y_list = []

for filename, group in grouped:
    group = group.sort_values('line_index').reset_index(drop=True)
    n_lines = len(group)
    
    for i in range(n_lines):
        curr_row = group.iloc[i]
        
        # Current line features
        X_chars_list.append(char_ids(curr_row['text']))
        X_text_list.append([
            curr_row['digit_ratio'], curr_row['alpha_ratio'], curr_row['upper_ratio'],
            curr_row['space_ratio'], curr_row['special_ratio'], curr_row['char_count'],
            curr_row['word_count'], curr_row['has_currency'], curr_row['has_date_pattern'],
            curr_row['has_qty_pattern']
        ])
        X_pos_list.append([
            curr_row['y_center'], curr_row['x_center'], curr_row['width'],
            curr_row['height'], curr_row['line_index_ratio']
        ])
        
        # Context features
        context_chars = []
        context_text = []
        
        for offset in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1):
            if offset == 0:
                continue
            
            ctx_idx = i + offset
            if 0 <= ctx_idx < n_lines:
                ctx_row = group.iloc[ctx_idx]
                context_chars.extend(char_ids(ctx_row['text'])[:20])
                context_text.extend([
                    ctx_row['digit_ratio'], ctx_row['alpha_ratio'], ctx_row['upper_ratio'],
                    ctx_row['space_ratio'], ctx_row['special_ratio']
                ])
            else:
                context_chars.extend([0] * 20)
                context_text.extend([0.0] * 5)
        
        X_context_chars_list.append(context_chars)
        X_context_text_list.append(context_text)
        y_list.append(LINE_CLASSES_LIST.index(curr_row['label']))

# Convert to numpy
X_chars = np.array(X_chars_list, dtype=np.int32)
X_text = np.array(X_text_list, dtype=np.float32)
X_pos = np.array(X_pos_list, dtype=np.float32)
X_context_chars = np.array(X_context_chars_list, dtype=np.int32)
X_context_text = np.array(X_context_text_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)

print(f"✅ Features ready:")
print(f"   X_chars: {X_chars.shape}")
print(f"   X_text: {X_text.shape}")
print(f"   X_pos: {X_pos.shape}")
print(f"   X_context_chars: {X_context_chars.shape}")
print(f"   X_context_text: {X_context_text.shape}")
print(f"   y: {y.shape}")

# Split with stratification
X_chars_train, X_chars_test, \
X_text_train, X_text_test, \
X_pos_train, X_pos_test, \
X_context_chars_train, X_context_chars_test, \
X_context_text_train, X_context_text_test, \
y_train, y_test = train_test_split(
    X_chars, X_text, X_pos, X_context_chars, X_context_text, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

print(f"\n📊 Split: Train={len(y_train):,}, Test={len(y_test):,}")

# Moderate oversampling for minority classes
print("\n📈 Oversampling minority classes...")
train_label_counts = pd.Series(y_train).value_counts()
median_count = int(train_label_counts.median())
target_count = max(median_count, 500)

print(f"   Target count per class: {target_count:,}")

oversample_indices = []
for class_idx, class_name in enumerate(LINE_CLASSES_LIST):
    class_mask = y_train == class_idx
    class_train_indices = [i for i, m in enumerate(class_mask) if m]
    current_count = len(class_train_indices)
    
    if current_count > 0 and current_count < target_count:
        n_to_add = target_count - current_count
        oversample_from = np.random.choice(class_train_indices, size=n_to_add, replace=True)
        oversample_indices.extend(oversample_from)
        marker = " ⭐" if class_name in ['GRAND_TOTAL', 'SUBTOTAL', 'TAX'] else ""
        print(f"  {class_name:20s}: {current_count:>6,} → {target_count:>6,} (+{n_to_add:>6,}){marker}")
    else:
        print(f"  {class_name:20s}: {current_count:>6,} (no oversampling)")

if oversample_indices:
    X_chars_train = np.vstack([X_chars_train, X_chars_train[oversample_indices]])
    X_text_train = np.vstack([X_text_train, X_text_train[oversample_indices]])
    X_pos_train = np.vstack([X_pos_train, X_pos_train[oversample_indices]])
    X_context_chars_train = np.vstack([X_context_chars_train, X_context_chars_train[oversample_indices]])
    X_context_text_train = np.vstack([X_context_text_train, X_context_text_train[oversample_indices]])
    y_train = np.concatenate([y_train, y_train[oversample_indices]])
    print(f"\n✅ Training set after oversampling: {len(y_train):,} samples")

# Shuffle
shuffle_idx = np.random.permutation(len(y_train))
X_chars_train = X_chars_train[shuffle_idx]
X_text_train = X_text_train[shuffle_idx]
X_pos_train = X_pos_train[shuffle_idx]
X_context_chars_train = X_context_chars_train[shuffle_idx]
X_context_text_train = X_context_text_train[shuffle_idx]
y_train = y_train[shuffle_idx]

# Class weights with EXTRA boost for critical classes
print("\n⚖️  Computing class weights...")
base_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(base_weights)}

# CRITICAL BOOST - Fokus ke yang sering salah
CRITICAL_BOOST = {
    'GRAND_TOTAL': 5.0,      # PALING PENTING
    'SUBTOTAL': 3.5,
    'TAX': 3.0,
    'DISCOUNT': 2.5,
    'SERVICE_CHARGE': 3.0,
    'CASH_PAYMENT': 2.0,
}

for class_name, boost in CRITICAL_BOOST.items():
    idx = LINE_CLASSES_LIST.index(class_name)
    original = class_weights[idx]
    class_weights[idx] *= boost
    print(f"  {class_name:20s}: {original:.2f} → {class_weights[idx]:.2f} (boosted {boost}x) ⭐")

# Build model
print("\n🏗️  Building context-aware model (V5 Indonesia)...")

from tensorflow.keras import layers, Model

input_chars = layers.Input(shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name='char_ids')
input_text = layers.Input(shape=(10,), dtype=tf.float32, name='text_feats')
input_pos = layers.Input(shape=(5,), dtype=tf.float32, name='pos_feats')
input_context_chars = layers.Input(shape=(X_context_chars.shape[1],), dtype=tf.int32, name='context_chars')
input_context_text = layers.Input(shape=(X_context_text.shape[1],), dtype=tf.float32, name='context_text')

# Current line
char_emb = layers.Embedding(128, 64, mask_zero=False, name='char_embedding')(input_chars)
lstm_out = layers.Bidirectional(layers.LSTM(128, return_sequences=False), name='bi_lstm')(char_emb)
lstm_drop = layers.Dropout(0.3)(lstm_out)

text_dense = layers.Dense(64, activation='relu', name='text_dense')(input_text)
text_bn = layers.BatchNormalization()(text_dense)
text_drop = layers.Dropout(0.3)(text_bn)

pos_dense = layers.Dense(32, activation='relu', name='pos_dense')(input_pos)

# Context
context_chars_cast = layers.Lambda(lambda x: tf.cast(x, tf.float32), name='context_chars_cast')(input_context_chars)
context_chars_dense = layers.Dense(64, activation='relu', name='context_chars_dense')(context_chars_cast)
context_text_dense = layers.Dense(32, activation='relu', name='context_text_dense')(input_context_text)

# Merge
merged = layers.Concatenate()([lstm_drop, text_drop, pos_dense, context_chars_dense, context_text_dense])
dense1 = layers.Dense(128, activation='relu', name='merge_dense')(merged)
bn1 = layers.BatchNormalization()(dense1)
drop1 = layers.Dropout(0.4)(bn1)

output = layers.Dense(NUM_CLASSES, activation='softmax', name='output')(drop1)

model = Model(
    inputs=[input_chars, input_text, input_pos, input_context_chars, input_context_text],
    outputs=output
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✅ Model built with {model.count_params():,} parameters")

# Callbacks
output_dir = PROJECT_ROOT / "models" / "classifier_v5_indonesia"
output_dir.mkdir(exist_ok=True, parents=True)

# TensorBoard log directory
log_dir = output_dir / "logs"
log_dir.mkdir(exist_ok=True)

print(f"\n📊 TensorBoard logs will be saved to: {log_dir}")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(output_dir / "best_weights.weights.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(
        str(output_dir / "training_history.csv")
    ),
    # TensorBoard callback - monitors ALL metrics
    tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,        # Log weight histograms every epoch
        write_graph=True,         # Visualize model graph
        write_images=False,       # Don't log images (no image data)
        update_freq='epoch',      # Update per epoch
        profile_batch=0,          # Disable profiling (can slow down)
        embeddings_freq=0,        # Disable embeddings visualization
    )
]

# Train
print("\n🚀 Starting training...")
print(f"   Dataset: Indonesia only ({len(y_train):,} train, {len(y_test):,} test)")
print(f"   Epochs: 100")
print(f"   Batch size: 128")
print(f"   Context window: ±{CONTEXT_WINDOW} lines")
print(f"   Critical classes boosted: {list(CRITICAL_BOOST.keys())}")
print()

history = model.fit(
    [X_chars_train, X_text_train, X_pos_train, X_context_chars_train, X_context_text_train],
    y_train,
    validation_data=(
        [X_chars_test, X_text_test, X_pos_test, X_context_chars_test, X_context_text_test],
        y_test
    ),
    epochs=100,
    batch_size=128,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print("\n" + "=" * 80)
print("EVALUATION")
print("=" * 80)

test_loss, test_acc = model.evaluate(
    [X_chars_test, X_text_test, X_pos_test, X_context_chars_test, X_context_text_test],
    y_test,
    verbose=0
)
print(f"\n📊 Test Accuracy: {test_acc * 100:.2f}%")
print(f"📊 Test Loss: {test_loss:.4f}")

# Per-class metrics
from sklearn.metrics import classification_report
y_pred = model.predict(
    [X_chars_test, X_text_test, X_pos_test, X_context_chars_test, X_context_text_test],
    verbose=0
)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n📊 Per-class metrics:")
print(classification_report(
    y_test,
    y_pred_classes,
    target_names=LINE_CLASSES_LIST,
    digits=3,
    zero_division=0
))

# CRITICAL classes performance
print("\n🎯 CRITICAL CLASSES PERFORMANCE (Indonesia receipts):")
for class_name in CRITICAL_BOOST.keys():
    idx = LINE_CLASSES_LIST.index(class_name)
    mask = y_test == idx
    if mask.sum() > 0:
        acc = (y_pred_classes[mask] == idx).mean()
        marker = "✅" if acc >= 0.85 else "⚠️" if acc >= 0.70 else "❌"
        print(f"  {marker} {class_name:20s}: {acc * 100:>6.2f}% ({mask.sum():>4} samples)")

# Save final
model.save_weights(str(output_dir / "final_weights.weights.h5"))
print(f"\n✅ Model saved to: {output_dir}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE - V5 INDONESIA")
print("=" * 80)
print(f"\n📁 Best weights: {output_dir / 'best_weights.weights.h5'}")
print(f"📊 Training log: {output_dir / 'training_history.csv'}")
print(f"📊 TensorBoard logs: {log_dir}")
print(f"\n💡 Model V5 trained on Indonesia data only for better accuracy!")
print(f"   Expected: Significant improvement on GRAND_TOTAL detection")
print(f"\n🔍 To view TensorBoard:")
print(f"   tensorboard --logdir={log_dir}")
print(f"   Then open: http://localhost:6006")
