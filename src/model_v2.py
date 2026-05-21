"""
OCR FinSight - Hierarchical BiLSTM + Self-Attention Model (V2)

Architecture:
  Level 1 (Per-line): Char Embedding → Conv1D → line_vector
  Level 2 (Receipt):  [line_vectors] → BiLSTM → Self-Attention → per-line classification

This model sees ALL lines in a receipt at once, so it can:
- Know "RM" after "Total" = TOTAL_PAYMENT
- Know "RM" next to item = ITEM_PRICE/QTY
- Understand receipt structure (header → items → totals → footer)

Requirements:
- TensorFlow 2.x
- Model Subclassing ✅
- Custom Layer ✅ (SelfAttentionLayer, LineEncoderLayer)
- Custom Loss ✅ (FocalLoss)
- Custom Callback ✅ (PerClassAccuracyCallback)
"""

import tensorflow as tf
import numpy as np
from typing import Optional

from src.config import (
    NUM_CLASSES, LINE_CLASSES, MAX_TEXT_LENGTH,
    EMBEDDING_DIM, DROPOUT_RATE, RANDOM_SEED
)


# ==============================================================================
# Custom Layers
# ==============================================================================

class LineEncoderLayer(tf.keras.layers.Layer):
    """Encode a single line's text into a fixed-size vector.
    
    Uses Conv1D (faster than LSTM for per-line encoding) + MaxPool.
    Combined with text features and position features.
    
    Input: (char_ids [batch, max_len], text_features [batch, n_text], pos_features [batch, n_pos])
    Output: line_vector [batch, output_dim]
    """
    
    def __init__(self, output_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        
        # Character encoding branch
        self.char_embedding = tf.keras.layers.Embedding(
            128, EMBEDDING_DIM, mask_zero=False, name='char_emb'
        )
        self.conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same', name='conv1')
        self.conv2 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same', name='conv2')
        self.global_pool = tf.keras.layers.GlobalMaxPooling1D(name='global_pool')
        self.char_dropout = tf.keras.layers.Dropout(0.2, name='char_dropout')
        
        # Text features branch
        self.text_dense = tf.keras.layers.Dense(48, activation='relu', name='text_dense')
        self.text_bn = tf.keras.layers.BatchNormalization(name='text_bn')
        
        # Position features branch
        self.pos_dense = tf.keras.layers.Dense(32, activation='relu', name='pos_dense')
        
        # Merge
        self.merge_dense = tf.keras.layers.Dense(output_dim, activation='relu', name='merge_dense')
        self.merge_bn = tf.keras.layers.BatchNormalization(name='merge_bn')
        self.merge_dropout = tf.keras.layers.Dropout(0.2, name='merge_dropout')
    
    def call(self, inputs, training=False):
        char_ids, text_feats, pos_feats = inputs
        
        # Character branch: Embedding → Conv1D → MaxPool
        x_char = self.char_embedding(char_ids)
        x_char = self.conv1(x_char)
        x_char = self.conv2(x_char)
        x_char = self.global_pool(x_char)
        x_char = self.char_dropout(x_char, training=training)
        
        # Text features branch
        x_text = self.text_dense(text_feats)
        x_text = self.text_bn(x_text, training=training)
        
        # Position features branch
        x_pos = self.pos_dense(pos_feats)
        
        # Merge all
        merged = tf.concat([x_char, x_text, x_pos], axis=-1)
        output = self.merge_dense(merged)
        output = self.merge_bn(output, training=training)
        output = self.merge_dropout(output, training=training)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'output_dim': self.output_dim})
        return config


class SelfAttentionLayer(tf.keras.layers.Layer):
    """Multi-head Self-Attention for sequence of line vectors.
    
    Allows each line to "attend" to other lines in the receipt.
    E.g., "RM" can attend to "Total" above it → learns it's TOTAL_PAYMENT.
    
    Input: [batch, seq_len, dim]
    Output: [batch, seq_len, dim]
    """
    
    def __init__(self, num_heads=4, key_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, name='mha'
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(name='ln1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(name='ln2')
        # FFN will be built dynamically based on input dim
        self.ffn_built = False
    
    def build(self, input_shape):
        dim = input_shape[-1]
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim * 2, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(dim),
        ], name='ffn')
        self.ffn_built = True
        super().build(input_shape)
    
    def call(self, inputs, mask=None, training=False):
        # Self-attention with residual connection
        attn_output = self.attention(inputs, inputs, attention_mask=mask, training=training)
        x = self.layernorm1(inputs + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x, training=training)
        output = self.layernorm2(x + ffn_output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_heads': self.num_heads, 'key_dim': self.key_dim})
        return config


# ==============================================================================
# Custom Loss
# ==============================================================================

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for class imbalance.
    
    Reduces loss for well-classified examples, focuses on hard cases.
    """
    
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


# ==============================================================================
# Custom Callback
# ==============================================================================

class PerClassAccuracyCallback(tf.keras.callbacks.Callback):
    """Log per-class accuracy during training.
    
    Tracks accuracy for each label class (STORE, DATE, TOTAL, etc.)
    """
    
    def __init__(self, validation_data, class_names=None, **kwargs):
        super().__init__(**kwargs)
        self.val_data = validation_data  # (inputs, labels, mask)
        self.class_names = class_names or [LINE_CLASSES[i] for i in range(NUM_CLASSES)]
        self.history = []
    
    def on_epoch_end(self, epoch, logs=None):
        inputs, labels, mask = self.val_data
        
        predictions = self.model(inputs, training=False)
        
        # Flatten predictions and labels (remove padding)
        pred_flat = tf.reshape(predictions, [-1, NUM_CLASSES])
        label_flat = tf.reshape(labels, [-1, NUM_CLASSES])
        mask_flat = tf.reshape(mask, [-1])
        
        # Apply mask
        valid_idx = tf.where(mask_flat > 0)
        pred_valid = tf.gather_nd(pred_flat, valid_idx)
        label_valid = tf.gather_nd(label_flat, valid_idx)
        
        pred_classes = tf.argmax(pred_valid, axis=-1)
        true_classes = tf.argmax(label_valid, axis=-1)
        
        epoch_results = {}
        for i, name in enumerate(self.class_names):
            class_mask = tf.equal(true_classes, i)
            total = tf.reduce_sum(tf.cast(class_mask, tf.int32))
            if total == 0:
                continue
            class_preds = tf.boolean_mask(pred_classes, class_mask)
            class_acc = tf.reduce_mean(tf.cast(tf.equal(class_preds, i), tf.float32))
            epoch_results[name] = float(class_acc)
            
            if logs is not None:
                logs[f'val_acc_{name}'] = float(class_acc)
        
        self.history.append(epoch_results)
        
        # Print summary
        parts = [f"{name}: {acc:.1%}" for name, acc in epoch_results.items()]
        print(f"  Per-class: {' | '.join(parts)}")


# ==============================================================================
# Main Model: Hierarchical BiLSTM + Self-Attention
# ==============================================================================

class HierarchicalReceiptClassifier(tf.keras.Model):
    """Hierarchical model that classifies ALL lines in a receipt at once.
    
    Architecture:
        Level 1: Per-line encoding (Conv1D + features → line_vector)
        Level 2: Sequence modeling (BiLSTM → Self-Attention → classification)
    
    This allows the model to understand receipt structure:
    - Lines near "Total" are likely TOTAL_PAYMENT
    - Lines at top are likely STORE/ADDRESS
    - Lines in middle are likely ITEMS
    """
    
    MAX_LINES = 40  # Max lines per receipt (padded)
    
    def __init__(self, num_classes=NUM_CLASSES, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        # Level 1: Per-line encoder
        self.line_encoder = LineEncoderLayer(output_dim=128, name='line_encoder')
        
        # Level 2: Sequence modeling
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, name='seq_lstm'),
            name='bi_lstm'
        )
        self.lstm_dropout = tf.keras.layers.Dropout(0.3, name='lstm_dropout')
        
        # Self-Attention
        self.self_attention = SelfAttentionLayer(num_heads=4, key_dim=32, name='self_attn')
        
        # Classification head
        self.classify_dense1 = tf.keras.layers.Dense(128, activation='relu', name='cls_dense1')
        self.classify_bn = tf.keras.layers.BatchNormalization(name='cls_bn')
        self.classify_dropout = tf.keras.layers.Dropout(0.3, name='cls_dropout')
        self.classify_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='cls_output')
    
    def call(self, inputs, training=False):
        """Forward pass.
        
        Args:
            inputs: tuple of (char_ids, text_features, pos_features, mask)
                - char_ids: [batch, max_lines, max_text_length]
                - text_features: [batch, max_lines, n_text_features]
                - pos_features: [batch, max_lines, n_pos_features]
                - mask: [batch, max_lines] (1=valid, 0=padding)
        
        Returns:
            predictions: [batch, max_lines, num_classes]
        """
        char_ids, text_feats, pos_feats, mask = inputs
        
        batch_size = tf.shape(char_ids)[0]
        max_lines = tf.shape(char_ids)[1]
        
        # Level 1: Encode each line independently
        # Reshape to [batch * max_lines, ...] for per-line processing
        char_flat = tf.reshape(char_ids, [-1, MAX_TEXT_LENGTH])
        text_flat = tf.reshape(text_feats, [-1, tf.shape(text_feats)[-1]])
        pos_flat = tf.reshape(pos_feats, [-1, tf.shape(pos_feats)[-1]])
        
        line_vectors = self.line_encoder((char_flat, text_flat, pos_flat), training=training)
        
        # Reshape back to [batch, max_lines, line_dim]
        line_dim = line_vectors.shape[-1]
        line_vectors = tf.reshape(line_vectors, [batch_size, max_lines, line_dim])
        
        # Apply mask (zero out padding lines)
        mask_expanded = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        line_vectors = line_vectors * mask_expanded
        
        # Level 2: Sequence modeling
        # BiLSTM for sequential context
        seq_output = self.bi_lstm(line_vectors, training=training)
        seq_output = self.lstm_dropout(seq_output, training=training)
        
        # Self-Attention for global context
        seq_output = self.self_attention(seq_output, training=training)
        
        # Classification per line
        x = self.classify_dense1(seq_output)
        x = self.classify_bn(x, training=training)
        x = self.classify_dropout(x, training=training)
        output = self.classify_output(x)
        
        return output
    
    def predict_receipt(self, lines: list) -> list:
        """Convenience method: predict labels for a list of line dicts.
        
        Args:
            lines: List of dicts with 'text', 'y_center', 'x_center', etc.
            
        Returns:
            List of (label, confidence) tuples
        """
        from src.model_v2_utils import prepare_single_receipt
        
        char_ids, text_feats, pos_feats, mask = prepare_single_receipt(lines)
        
        # Add batch dimension
        inputs = (
            tf.expand_dims(char_ids, 0),
            tf.expand_dims(text_feats, 0),
            tf.expand_dims(pos_feats, 0),
            tf.expand_dims(mask, 0),
        )
        
        predictions = self(inputs, training=False)
        predictions = predictions[0]  # Remove batch dim
        
        results = []
        for i in range(len(lines)):
            pred = predictions[i].numpy()
            idx = np.argmax(pred)
            confidence = float(pred[idx])
            label = LINE_CLASSES[idx]
            results.append((label, confidence))
        
        return results
