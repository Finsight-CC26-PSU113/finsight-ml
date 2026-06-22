"""
OCR FinSight — Classifier V5 Model Architecture
Context-Aware BiLSTM with ±4 line context window.

5 Inputs:
  1. char_ids:        [batch, MAX_TEXT_LENGTH]    — current line characters
  2. text_feats:      [batch, 10]                 — statistical text features
  3. pos_feats:       [batch, 5]                  — spatial position features
  4. context_chars:   [batch, context_window*2*20] — neighboring line chars
  5. context_text:    [batch, context_window*2*5]  — neighboring line features

Output: softmax over 12 line classes.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model

from app.core.config import settings


class ContextAwareClassifier(Model):
    """Context-aware receipt line classifier (V5 Indonesia)."""

    def __init__(
        self,
        num_classes: int = 12,
        context_window: int = settings.CONTEXT_WINDOW,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.context_window = context_window

        # Context dimensions
        self.context_char_dim = context_window * 2 * 20
        self.context_text_dim = context_window * 2 * 5

        # ── Current line: character branch ──
        self.char_embedding = layers.Embedding(
            128, settings.EMBEDDING_DIM, mask_zero=False, name="char_embedding"
        )
        self.bi_lstm = layers.Bidirectional(
            layers.LSTM(settings.LSTM_UNITS, return_sequences=False), name="bi_lstm"
        )
        self.lstm_dropout = layers.Dropout(settings.DROPOUT_RATE)

        # ── Current line: text feature branch ──
        self.text_dense = layers.Dense(
            settings.DENSE_UNITS, activation="relu", name="text_dense"
        )
        self.text_bn = layers.BatchNormalization(name="text_bn")
        self.text_dropout = layers.Dropout(settings.DROPOUT_RATE)

        # ── Current line: position branch ──
        self.pos_dense = layers.Dense(32, activation="relu", name="pos_dense")

        # ── Context: character branch ──
        self.context_chars_dense = layers.Dense(
            settings.DENSE_UNITS, activation="relu", name="context_chars_dense"
        )

        # ── Context: text feature branch ──
        self.context_text_dense = layers.Dense(
            32, activation="relu", name="context_text_dense"
        )

        # ── Merge + classify ──
        self.merge_dense = layers.Dense(
            settings.DENSE_UNITS, activation="relu", name="merge_dense"
        )
        self.merge_bn = layers.BatchNormalization(name="merge_bn")
        self.merge_dropout = layers.Dropout(settings.DROPOUT_RATE * 1.33)  # 0.4
        self.output_layer = layers.Dense(
            num_classes, activation="softmax", name="output"
        )

    def call(self, inputs, training=False):
        char_ids, text_feats, pos_feats, context_chars, context_text = inputs

        # Current line: text content
        x_chars = self.char_embedding(char_ids)
        x_chars = self.bi_lstm(x_chars, training=training)
        x_chars = self.lstm_dropout(x_chars, training=training)

        # Current line: text features
        x_text = self.text_dense(text_feats)
        x_text = self.text_bn(x_text, training=training)
        x_text = self.text_dropout(x_text, training=training)

        # Current line: position
        x_pos = self.pos_dense(pos_feats)

        # Context: characters (cast int → float)
        x_ctx_chars = self.context_chars_dense(tf.cast(context_chars, tf.float32))

        # Context: text features
        x_ctx_text = self.context_text_dense(context_text)

        # Merge all branches
        merged = tf.concat(
            [x_chars, x_text, x_pos, x_ctx_chars, x_ctx_text], axis=-1
        )
        x = self.merge_dense(merged)
        x = self.merge_bn(x, training=training)
        x = self.merge_dropout(x, training=training)

        return self.output_layer(x)

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "context_window": self.context_window,
        }
