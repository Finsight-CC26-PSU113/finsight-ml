"""
OCR FinSight - Classifier Service
Model architecture (ReceiptLineClassifier) + unified inference helper.
Training code lives in training/model.py.
"""

import re
import numpy as np
import tensorflow as tf

from app.config import (
    NUM_CLASSES, LINE_CLASSES, EMBEDDING_DIM, LSTM_UNITS,
    DENSE_UNITS, DROPOUT_RATE, MAX_TEXT_LENGTH,
)

# ── Feature regex (compiled once) ──────────────────────────────────────────────

_DATE_RE = re.compile(
    r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})|(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})|'
    r'(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})|'
    r'((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}[\s,]+\d{2,4})',
    re.IGNORECASE,
)
_TIME_RE = re.compile(r'\d{1,2}:\d{2}(:\d{2})?')
_QTY_RE = re.compile(r'\d+\s*[xX\*@]\s*[\d.,]+|@\s*[\d.,]+')
_CURR_RE = re.compile(r'(rm|rp|idr|usd|sgd|sr|\$|€)\s*[\d.,]+', re.IGNORECASE)


# ── Custom Layers ───────────────────────────────────────────────────────────────

class TextFeatureLayer(tf.keras.layers.Layer):
    """Encode statistical text features (10 dims → 32 dims)."""

    def __init__(self, output_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', name='dense1')
        self.bn = tf.keras.layers.BatchNormalization(name='bn')
        self.dense2 = tf.keras.layers.Dense(output_dim, activation='relu', name='dense2')

    def call(self, inputs, training=False):
        return self.dense2(self.bn(self.dense1(inputs), training=training))

    def get_config(self):
        return {**super().get_config(), 'output_dim': self.output_dim}


class PositionFeatureLayer(tf.keras.layers.Layer):
    """Encode spatial position features (5 dims → 16 dims)."""

    def __init__(self, output_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', name='dense1')
        self.dense2 = tf.keras.layers.Dense(output_dim, activation='relu', name='dense2')

    def call(self, inputs, training=False):
        return self.dense2(self.dense1(inputs))

    def get_config(self):
        return {**super().get_config(), 'output_dim': self.output_dim}


# ── Model ───────────────────────────────────────────────────────────────────────

class ReceiptLineClassifier(tf.keras.Model):
    """3-branch BiLSTM receipt line classifier.

    Inputs: (char_ids, text_features, position_features)
    Output: softmax over LINE_CLASSES (12 categories in V2)
    """

    VOCAB_SIZE = 128

    def __init__(self, num_classes=NUM_CLASSES, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.char_embedding = tf.keras.layers.Embedding(
            self.VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True, name='char_embedding')
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=False, name='lstm'),
            name='bi_lstm')
        self.text_dropout = tf.keras.layers.Dropout(DROPOUT_RATE, name='text_dropout')

        self.text_feature_layer = TextFeatureLayer(output_dim=32, name='text_feature_layer')
        self.position_feature_layer = PositionFeatureLayer(output_dim=16, name='position_feature_layer')

        # NOTE: layer names below mirror the saved checkpoint's auto-names so load_weights matches.
        self.merge_dense = tf.keras.layers.Dense(DENSE_UNITS, activation='relu', name='dense')
        self.merge_bn = tf.keras.layers.BatchNormalization(name='batch_normalization')
        self.merge_dropout = tf.keras.layers.Dropout(DROPOUT_RATE, name='merge_dropout')
        self.output_dense = tf.keras.layers.Dense(num_classes, activation='softmax', name='dense_1')

    def call(self, inputs, training=False):
        char_ids, text_feats, pos_feats = inputs
        x_text = self.text_dropout(
            self.bi_lstm(self.char_embedding(char_ids), training=training),
            training=training)
        x_stats = self.text_feature_layer(text_feats, training=training)
        x_pos = self.position_feature_layer(pos_feats, training=training)
        merged = tf.concat([x_text, x_stats, x_pos], axis=-1)
        x = self.merge_dropout(
            self.merge_bn(self.merge_dense(merged), training=training),
            training=training)
        return self.output_dense(x)


# ── Feature extraction helpers ──────────────────────────────────────────────────

def _text_features(text: str) -> list[float]:
    text = str(text) if text else ""
    n = len(text)
    if n == 0:
        return [0.0] * 10
    digits = sum(c.isdigit() for c in text)
    alphas = sum(c.isalpha() for c in text)
    uppers = sum(c.isupper() for c in text)
    spaces = sum(c.isspace() for c in text)
    specials = n - digits - alphas - spaces
    return [
        digits / n, alphas / n, uppers / max(alphas, 1),
        spaces / n, specials / n, n / 100.0,
        len(text.split()) / 20.0,
        1.0 if _CURR_RE.search(text) else 0.0,
        1.0 if (_DATE_RE.search(text) or _TIME_RE.search(text)) else 0.0,
        1.0 if _QTY_RE.search(text) else 0.0,
    ]


def _char_ids(text: str, max_len: int = MAX_TEXT_LENGTH) -> list[int]:
    text = str(text) if text else ""
    ids = [min(ord(c), 127) for c in text[:max_len]]
    return ids + [0] * (max_len - len(ids))


# ── Weight loading ────────────────────────────────────────────────────────────

def load_v2_weights(model, weights_path: str) -> None:
    """Load a Classifier V2 ``.weights.h5`` checkpoint onto ``model``.

    The checkpoint stores its layer tree with ``char_embedding``/``bi_lstm`` at the
    root and the remaining layers under ``/layers/`` (the Keras save-time grouping).
    A plain ``model.load_weights()`` on this TF/Keras version doesn't line that tree
    up with the freshly-built model, so we walk the model layer-by-layer and pull
    each layer's variables from the matching h5 ``vars`` group by name.

    Call AFTER the model has been built (one forward pass) so its variables exist.
    """
    import h5py

    def restore(layer, grp):
        if 'vars' in grp:
            store = {k: grp['vars'][k] for k in grp['vars'].keys()}
            if store:
                layer.load_own_variables(store)
        for child in layer._flatten_layers(include_self=False, recursive=False):
            name = child.name
            if name in grp:
                restore(child, grp[name])
            elif 'layers' in grp and name in grp['layers']:
                restore(child, grp['layers'][name])

    with h5py.File(weights_path, 'r') as f:
        restore(model, f)


# ── Unified inference ───────────────────────────────────────────────────────────

def predict_lines(model, lines: list[dict]) -> tuple[list[str], list[float]]:
    """Run inference on OCR lines.

    Works with both Classifier V2 (direct TF) and V1 (OnlineLearningModel).
    Returns (labels, confidences).
    """
    if not lines:
        return [], []

    if getattr(model, '_is_v2', False):
        n = len(lines)
        chars = np.array([_char_ids(l.get('text', '')) for l in lines], dtype=np.int32)
        tfeats = np.array([_text_features(l.get('text', '')) for l in lines], dtype=np.float32)
        pfeats = np.array([
            [float(l.get('y_center', 0.5)), float(l.get('x_center', 0.5)),
             float(l.get('width', 0.1)), float(l.get('height', 0.05)),
             float(i) / max(n, 1)]
            for i, l in enumerate(lines)
        ], dtype=np.float32)
        preds = model((chars, tfeats, pfeats), training=False).numpy()
        labels = [LINE_CLASSES[int(idx)] for idx in np.argmax(preds, axis=1)]
        confs = [float(c) for c in np.max(preds, axis=1)]
        return labels, confs

    return model.predict(lines)
