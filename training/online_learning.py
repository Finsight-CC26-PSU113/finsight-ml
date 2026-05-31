"""
OCR FinSight - Online Learning
Fine-tune the classifier from user corrections in real-time.
"""

import re
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

from app.config import ROOT_DIR, LINE_CLASSES, MAX_TEXT_LENGTH
from app.services.classifier import ReceiptLineClassifier


def _text_to_char_ids(text: str) -> np.ndarray:
    ids = [min(ord(c), 127) for c in text[:MAX_TEXT_LENGTH]]
    return np.array(ids + [0] * (MAX_TEXT_LENGTH - len(ids)), dtype=np.int32)


def _extract_features(line: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = line.get('text', '')
    char_ids = _text_to_char_ids(text)
    text_feats = np.array([
        len(text), sum(c.isupper() for c in text), sum(c.islower() for c in text),
        sum(c.isdigit() for c in text), sum(c.isspace() for c in text),
        sum(c in '.,;:!?' for c in text),
        1 if any(c.isdigit() for c in text) else 0,
        1 if any(c.isupper() for c in text) else 0,
        1 if any(c in 'Rp$€£¥' for c in text) else 0,
        1 if any(c in '/-.' for c in text) else 0,
    ], dtype=np.float32)
    pos_feats = np.array([
        line.get('y_center', 0.5), line.get('x_center', 0.5),
        line.get('width', 0.1), line.get('height', 0.05),
        line.get('line_index', 0),
    ], dtype=np.float32)
    return char_ids, text_feats, pos_feats


class OnlineLearningModel:
    """Wrapper that fine-tunes the classifier from accumulated user corrections."""

    def __init__(self, base_model_path: Path = None, buffer_size: int = 50):
        if base_model_path is None:
            base_model_path = ROOT_DIR / "models" / "best_weights.weights.h5"

        self.label_to_idx = {v: k for k, v in LINE_CLASSES.items()}
        self.idx_to_label = LINE_CLASSES
        self.correction_buffer = []
        self.buffer_size = buffer_size
        self.learning_rate = 0.0005
        self.is_retraining = False
        self.retrain_count = 0
        self.online_model_path = ROOT_DIR / "models" / "online_model.h5"

        self.base_model = ReceiptLineClassifier()
        dummy = (
            np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32),
            np.zeros((1, 10), dtype=np.float32),
            np.zeros((1, 5), dtype=np.float32),
        )
        self.base_model(dummy, training=False)
        self.base_model.load_weights(str(base_model_path))
        self.base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"[OnlineLearning] Ready — buffer_size={buffer_size}")

    def predict(self, lines: List[Dict]) -> Tuple[List[str], List[float]]:
        if not lines:
            return [], []
        char_ids, tfeats, pfeats = zip(*[_extract_features(l) for l in lines])
        preds = self.base_model.predict(
            [np.array(char_ids), np.array(tfeats), np.array(pfeats)], verbose=0)
        labels = [self.idx_to_label[np.argmax(p)] for p in preds]
        confs = [float(np.max(p)) for p in preds]
        return self._post_process(lines, labels), confs

    def _post_process(self, lines, labels):
        """Rule-based corrections (zone + keyword)."""
        corrected = labels[:]
        n = len(lines)
        total_start = n
        for i, line in enumerate(lines):
            if re.search(r'\b(sub\s*total|total|grand\s*total)\b', line.get('text', '').lower()):
                total_start = i
                break

        for i, (line, label) in enumerate(zip(lines, labels)):
            tl = line.get('text', '').lower().strip()
            ts = line.get('text', '').strip()
            pos = i / max(n, 1)

            if any(re.search(p, ts) for p in [
                r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}'
            ]):
                if not re.match(r'^\d{7,}$', ts):
                    corrected[i] = 'DATE'
                    continue

            if any(k in tl for k in ['total', 'subtotal', 'tunai', 'bayar', 'kembali',
                                      'change', 'amount', 'tax', 'pajak', 'discount', 'diskon']):
                if 'cashier' not in tl and 'kasir' not in tl:
                    corrected[i] = 'TOTAL_PAYMENT'
                    continue

            if i >= total_start:
                if re.match(r'^[\d.,\s]+$', ts) and len(ts) >= 2:
                    corrected[i] = 'TOTAL_PAYMENT'
                    continue

            if pos < 0.15 and i <= 2:
                alpha = sum(c.isalpha() for c in ts)
                upper = sum(c.isupper() for c in ts)
                if alpha > 2 and upper / max(alpha, 1) > 0.5 and label in ['OTHER', 'ITEM_DESC']:
                    corrected[i] = 'STORE'
                    continue

        return corrected

    def add_correction(self, line: Dict, true_label: str):
        char_ids, tfeats, pfeats = _extract_features(line)
        label_idx = self.label_to_idx[true_label]
        one_hot = np.zeros(len(self.idx_to_label))
        one_hot[label_idx] = 1
        self.correction_buffer.append({'char_ids': char_ids, 'text_features': tfeats,
                                        'position_features': pfeats, 'label': one_hot})
        print(f"[OnlineLearning] Buffer: {len(self.correction_buffer)}/{self.buffer_size}")
        if len(self.correction_buffer) >= self.buffer_size:
            self.retrain()

    def retrain(self):
        if self.is_retraining or not self.correction_buffer:
            return
        self.is_retraining = True
        try:
            self.retrain_count += 1
            print(f"[OnlineLearning] Retraining on {len(self.correction_buffer)} corrections...")
            chars = np.array([c['char_ids'] for c in self.correction_buffer])
            tfeats = np.array([c['text_features'] for c in self.correction_buffer])
            pfeats = np.array([c['position_features'] for c in self.correction_buffer])
            labels = np.array([c['label'] for c in self.correction_buffer])

            history = self.base_model.fit(
                [chars, tfeats, pfeats], labels,
                epochs=5, batch_size=16, verbose=0, validation_split=0.2,
            )
            self.base_model.save(self.online_model_path)

            from training.auto_evaluation import get_auto_evaluator
            evaluator = get_auto_evaluator()
            evaluation = evaluator.evaluate_retrain(
                retrain_id=self.retrain_count,
                loss=history.history['loss'][-1],
                accuracy=history.history['accuracy'][-1],
                corrections_count=len(self.correction_buffer),
                buffer_size=self.buffer_size,
            )
            for alert in evaluator.get_alerts():
                if alert['severity'] in ('high', 'medium'):
                    print(f"[AutoEval] {alert['message']}")

            self.correction_buffer = []
        except Exception as e:
            print(f"[OnlineLearning] Retrain failed: {e}")
        finally:
            self.is_retraining = False

    def get_buffer_status(self) -> Dict:
        return {
            'buffer_size': len(self.correction_buffer),
            'buffer_capacity': self.buffer_size,
            'progress': len(self.correction_buffer) / self.buffer_size * 100,
        }


_online_model = None


def get_online_model() -> OnlineLearningModel:
    global _online_model
    if _online_model is None:
        _online_model = OnlineLearningModel()
    return _online_model
