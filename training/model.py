"""
OCR FinSight - Training Module
Custom training loop (GradientTape + TensorBoard) for ReceiptLineClassifier.
Model architecture lives in app/services/classifier.py.
"""

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

from app.config import (
    NUM_CLASSES, LINE_CLASSES, LEARNING_RATE, BATCH_SIZE, EPOCHS,
    MAX_TEXT_LENGTH, NUM_STATISTICAL_FEATURES, NUM_POSITION_FEATURES,
    TRAIN_SPLIT, RANDOM_SEED, LOGS_DIR, MODELS_DIR, ROOT_DIR,
)
from app.services.classifier import ReceiptLineClassifier


# ── Custom Loss ─────────────────────────────────────────────────────────────────

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss to handle class imbalance."""

    def __init__(self, gamma=2.0, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.math.pow(1.0 - y_pred, self.gamma)
        fl = weight * ce
        if self.alpha is not None:
            fl = fl * self.alpha
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))


# ── Custom Callback ──────────────────────────────────────────────────────────────

class PerClassAccuracyCallback(tf.keras.callbacks.Callback):
    """Log per-class accuracy to TensorBoard each epoch."""

    def __init__(self, validation_data, log_dir, class_names=None, classifier_model=None, **kwargs):
        super().__init__(**kwargs)
        self.val_features = validation_data[0]
        self.val_labels = validation_data[1]
        self.class_names = class_names or [LINE_CLASSES[i] for i in range(NUM_CLASSES)]
        self.writer = tf.summary.create_file_writer(str(log_dir / "per_class"))
        self._classifier_model = classifier_model

    def on_epoch_end(self, epoch, logs=None):
        preds = self._classifier_model(self.val_features, training=False)
        pred_classes = tf.argmax(preds, axis=-1)
        true_classes = tf.argmax(self.val_labels, axis=-1)
        with self.writer.as_default():
            for i, name in enumerate(self.class_names):
                mask = tf.equal(true_classes, i)
                if tf.reduce_sum(tf.cast(mask, tf.int32)) == 0:
                    continue
                acc = tf.reduce_mean(tf.cast(
                    tf.equal(tf.boolean_mask(pred_classes, mask), i), tf.float32))
                tf.summary.scalar(f"accuracy/{name}", acc, step=epoch)
            self.writer.flush()


# ── Data utilities ───────────────────────────────────────────────────────────────

def text_to_char_ids(text: str, max_length: int = MAX_TEXT_LENGTH) -> np.ndarray:
    ids = [min(ord(c), 127) for c in text[:max_length]]
    return np.array(ids + [0] * (max_length - len(ids)), dtype=np.int32)


def load_training_data(csv_path: Path = None):
    """Load labeled CSV and return (train, val, test) splits."""
    if csv_path is None:
        csv_path = ROOT_DIR / "data" / "labeled_lines.csv"

    print(f"[Data] Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    label_to_idx = {v: k for k, v in LINE_CLASSES.items()}
    df['label_idx'] = df['label'].map(label_to_idx)
    df = df.dropna(subset=['label_idx'])
    df['label_idx'] = df['label_idx'].astype(int)

    # Oversampling minority classes
    max_count = df['label'].value_counts().max()
    target = max(max_count // 3, 500)
    dfs = []
    for _, group in df.groupby('label'):
        if len(group) < target:
            dfs.append(group.sample(target, replace=True, random_state=RANDOM_SEED))
        else:
            dfs.append(group)
    df = pd.concat(dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"[Data] Samples after oversampling: {len(df)}")

    char_ids = np.array([text_to_char_ids(str(t)) for t in df['text'].values])

    text_feat_cols = [
        'digit_ratio', 'alpha_ratio', 'upper_ratio', 'space_ratio', 'special_ratio',
        'char_count', 'word_count', 'has_currency', 'has_date_pattern', 'has_qty_pattern',
    ]
    text_features = df[text_feat_cols].values.astype(np.float32)
    text_features[:, text_feat_cols.index('char_count')] /= 100.0
    text_features[:, text_feat_cols.index('word_count')] /= 20.0

    pos_feat_cols = ['y_ratio', 'x_ratio', 'width_ratio', 'height_ratio', 'line_index_ratio']
    pos_features = df[pos_feat_cols].values.astype(np.float32)

    labels = tf.keras.utils.to_categorical(df['label_idx'].values, num_classes=NUM_CLASSES)
    texts = df['text'].values.astype(str)
    indices = np.arange(len(df))

    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED,
                                            stratify=df['label_idx'].values)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=RANDOM_SEED,
                                          stratify=df['label_idx'].values[temp_idx])

    def make_split(idx):
        return (char_ids[idx], text_features[idx], pos_features[idx]), labels[idx], texts[idx]

    print(f"[Data] Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return make_split(train_idx), make_split(val_idx), make_split(test_idx)


# ── Training step functions ──────────────────────────────────────────────────────

@tf.function
def train_step(model, inputs, labels, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        preds = model(inputs, training=True)
        loss = loss_fn(labels, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, preds


@tf.function
def val_step(model, inputs, labels, loss_fn):
    preds = model(inputs, training=False)
    return loss_fn(labels, preds), preds


# ── Main training loop ───────────────────────────────────────────────────────────

def train_model(train_data, val_data, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train with GradientTape + TensorBoard. Returns best model."""
    (tr_chars, tr_tfeats, tr_pfeats), tr_labels, _ = train_data
    (va_chars, va_tfeats, va_pfeats), va_labels, _ = val_data

    train_ds = tf.data.Dataset.from_tensor_slices((
        {'chars': tr_chars, 'tf': tr_tfeats, 'pf': tr_pfeats}, tr_labels
    )).shuffle(10000, seed=RANDOM_SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((
        {'chars': va_chars, 'tf': va_tfeats, 'pf': va_pfeats}, va_labels
    )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    counts = np.sum(tr_labels, axis=0)
    alpha = counts.sum() / (NUM_CLASSES * (counts + 1))
    alpha = alpha / alpha.sum() * NUM_CLASSES

    model = ReceiptLineClassifier()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha)

    dummy = (
        tf.zeros((1, MAX_TEXT_LENGTH), dtype=tf.int32),
        tf.zeros((1, NUM_STATISTICAL_FEATURES), dtype=tf.float32),
        tf.zeros((1, NUM_POSITION_FEATURES), dtype=tf.float32),
    )
    model(dummy, training=False)
    model.summary()

    log_dir = LOGS_DIR / f"run_{int(time.time())}"
    train_writer = tf.summary.create_file_writer(str(log_dir / "train"))
    val_writer = tf.summary.create_file_writer(str(log_dir / "val"))
    per_class_cb = PerClassAccuracyCallback(
        validation_data=((va_chars, va_tfeats, va_pfeats), va_labels),
        log_dir=log_dir, classifier_model=model)

    train_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.CategoricalAccuracy()
    val_loss = tf.keras.metrics.Mean()
    val_acc = tf.keras.metrics.CategoricalAccuracy()

    best_val_acc = 0.0
    print(f"\nTensorBoard: {log_dir}\nTraining for {epochs} epochs...\n")

    for epoch in range(epochs):
        for m in [train_loss, train_acc, val_loss, val_acc]:
            m.reset_state()

        for batch in train_ds:
            feats, lbls = batch
            inputs = (feats['chars'], feats['tf'], feats['pf'])
            loss, preds = train_step(model, inputs, lbls, optimizer, loss_fn)
            train_loss.update_state(loss)
            train_acc.update_state(lbls, preds)

        for batch in val_ds:
            feats, lbls = batch
            inputs = (feats['chars'], feats['tf'], feats['pf'])
            loss, preds = val_step(model, inputs, lbls, loss_fn)
            val_loss.update_state(loss)
            val_acc.update_state(lbls, preds)

        tl, ta = train_loss.result().numpy(), train_acc.result().numpy()
        vl, va = val_loss.result().numpy(), val_acc.result().numpy()

        with train_writer.as_default():
            tf.summary.scalar('loss', tl, step=epoch)
            tf.summary.scalar('accuracy', ta, step=epoch)
        with val_writer.as_default():
            tf.summary.scalar('loss', vl, step=epoch)
            tf.summary.scalar('accuracy', va, step=epoch)

        per_class_cb.on_epoch_end(epoch)

        marker = " *" if va > best_val_acc else ""
        print(f"  Epoch {epoch+1:>3d}/{epochs} | "
              f"loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} val_acc={va:.4f}{marker}")

        if va > best_val_acc:
            best_val_acc = va
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model.save_weights(str(MODELS_DIR / "best_weights.weights.h5"))

    print(f"\nBest val acc: {best_val_acc:.4f}")
    model.load_weights(str(MODELS_DIR / "best_weights.weights.h5"))
    return model


def evaluate_model(model, test_data):
    """Print per-class accuracy on test set."""
    (test_chars, test_tfeats, test_pfeats), test_labels, _ = test_data
    preds = model((test_chars, test_tfeats, test_pfeats), training=False)
    pred_classes = tf.argmax(preds, axis=-1).numpy()
    true_classes = tf.argmax(test_labels, axis=-1).numpy()
    accuracy = np.mean(pred_classes == true_classes)
    print(f"\nTest Accuracy: {accuracy:.4f}\n")
    print(f"  {'Class':>20s} | {'Correct':>7s} | {'Total':>5s} | {'Acc':>6s}")
    print(f"  {'-'*20} | {'-'*7} | {'-'*5} | {'-'*6}")
    for i in range(NUM_CLASSES):
        mask = true_classes == i
        total = mask.sum()
        if total == 0:
            continue
        correct = (pred_classes[mask] == i).sum()
        print(f"  {LINE_CLASSES[i]:>20s} | {correct:>7d} | {total:>5d} | {correct/total:>6.4f}")
    return accuracy
