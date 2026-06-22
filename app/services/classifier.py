"""
OCR FinSight — Classifier Service (V5 Indonesia)
Context-Aware BiLSTM with ±4 line context window.
"""

from __future__ import annotations

import re
import numpy as np
import tensorflow as tf

from app.core.config import settings
from app.models.classifier_v5 import ContextAwareClassifier

# ── Feature regex (compiled once) ──────────────────────────────────────────

_DATE_RE = re.compile(
    r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})|(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})|"
    r"(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})|"
    r"((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}[\s,]+\d{2,4})",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"\d{1,2}:\d{2}(:\d{2})?")
_QTY_RE = re.compile(r"\d+\s*[xX\*@]\s*[\d.,]+|@\s*[\d.,]+")
_CURR_RE = re.compile(r"(rm|rp|idr|usd|sgd|sr|\$|€)\s*[\d.,]+", re.IGNORECASE)


# ── Feature extraction helpers ─────────────────────────────────────────────


def text_features(text: str) -> list[float]:
    """Extract 10 statistical features from a text string."""
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
        digits / n,
        alphas / n,
        uppers / max(alphas, 1),
        spaces / n,
        specials / n,
        n / 100.0,
        len(text.split()) / 20.0,
        1.0 if _CURR_RE.search(text) else 0.0,
        1.0 if (_DATE_RE.search(text) or _TIME_RE.search(text)) else 0.0,
        1.0 if _QTY_RE.search(text) else 0.0,
    ]


def char_ids(text: str, max_len: int = settings.MAX_TEXT_LENGTH) -> list[int]:
    """Encode text as ASCII character IDs, padded to max_len."""
    text = str(text) if text else ""
    ids = [min(ord(c), 127) for c in text[:max_len]]
    return ids + [0] * (max_len - len(ids))


def generate_context_features(
    lines: list[dict], context_window: int = settings.CONTEXT_WINDOW
) -> tuple[list, list]:
    """Generate context features from neighboring lines.

    Returns (context_chars_list, context_text_list).
    """
    n = len(lines)
    ctx_chars, ctx_text = [], []
    for i in range(n):
        c_chars, c_text = [], []
        for offset in range(-context_window, context_window + 1):
            if offset == 0:
                continue
            idx = i + offset
            if 0 <= idx < n:
                t = lines[idx].get("text", "")
                c_chars.extend(char_ids(t, max_len=20))
                c_text.extend(text_features(t)[:5])
            else:
                c_chars.extend([0] * 20)
                c_text.extend([0.0] * 5)
        ctx_chars.append(c_chars)
        ctx_text.append(c_text)
    return ctx_chars, ctx_text


# ── Model loader ───────────────────────────────────────────────────────────


def load_v5_classifier(
    weights_path: str | None = None,
    context_window: int = settings.CONTEXT_WINDOW,
) -> ContextAwareClassifier:
    """Load Classifier V5 with weights.

    Builds the model with a dummy forward pass, then loads weights.
    """
    path = weights_path or str(settings.CLASSIFIER_V5_WEIGHTS)

    model = ContextAwareClassifier(context_window=context_window)

    # Build with dummy input
    dummy_chars = tf.zeros((1, settings.MAX_TEXT_LENGTH), dtype=tf.int32)
    dummy_tf = tf.zeros((1, 10), dtype=tf.float32)
    dummy_pf = tf.zeros((1, 5), dtype=tf.float32)
    cw = context_window
    dummy_ctx_chars = tf.zeros((1, cw * 2 * 20), dtype=tf.int32)
    dummy_ctx_text = tf.zeros((1, cw * 2 * 5), dtype=tf.float32)
    _ = model(
        (dummy_chars, dummy_tf, dummy_pf, dummy_ctx_chars, dummy_ctx_text),
        training=False,
    )

    model.load_weights(str(path))
    model._is_v2 = True
    model._version = "V5"
    model._is_context_model = True
    model._context_window = context_window
    print(f"[Classifier] V5 loaded from {path}")
    return model


# ── Unified inference ──────────────────────────────────────────────────────


def predict_lines(
    model: ContextAwareClassifier, lines: list[dict]
) -> tuple[list[str], list[float]]:
    """Run V5 inference on OCR lines. Returns (labels, confidences)."""
    if not lines:
        return [], []

    n = len(lines)
    chars = np.array(
        [char_ids(l.get("text", "")) for l in lines], dtype=np.int32
    )
    tfeats = np.array(
        [text_features(l.get("text", "")) for l in lines], dtype=np.float32
    )
    pfeats = np.array(
        [
            [
                float(l.get("y_center", 0.5)),
                float(l.get("x_center", 0.5)),
                float(l.get("width", 0.1)),
                float(l.get("height", 0.05)),
                float(i) / max(n, 1),
            ]
            for i, l in enumerate(lines)
        ],
        dtype=np.float32,
    )

    ctx_chars, ctx_text = generate_context_features(lines, model._context_window)
    preds = model(
        (
            chars,
            tfeats,
            pfeats,
            np.array(ctx_chars, dtype=np.int32),
            np.array(ctx_text, dtype=np.float32),
        ),
        training=False,
    ).numpy()

    labels = [settings.LINE_CLASSES[int(idx)] for idx in np.argmax(preds, axis=1)]
    confs = [float(c) for c in np.max(preds, axis=1)]
    return labels, confs
