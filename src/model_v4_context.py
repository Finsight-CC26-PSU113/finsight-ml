"""
OCR FinSight - Context-Aware Receipt Line Classifier (V4)
==========================================================
Model architecture yang kompatibel dengan weights dari train_classifier_v4_context.py

PERBEDAAN dengan model.py:
- Tambahan input: context_chars dan context_text
- Context dari ±2 lines di sekitar line yang diclassify
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from src.config import MAX_TEXT_LENGTH, NUM_CLASSES, CONTEXT_WINDOW


class ContextAwareClassifier(Model):
    """Context-aware classifier dengan 5 inputs.
    
    Inputs:
        1. char_ids: [batch, MAX_TEXT_LENGTH] - current line characters
        2. text_feats: [batch, 10] - current line text features
        3. pos_feats: [batch, 5] - current line position features
        4. context_chars: [batch, context_dim] - neighboring lines characters
        5. context_text: [batch, context_dim] - neighboring lines text features
    """
    
    def __init__(self, num_classes=NUM_CLASSES, context_window=CONTEXT_WINDOW, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.context_window = context_window
        
        # Context dimensions (from training script)
        # context_window * 2 lines * 20 chars = 80 chars
        # context_window * 2 lines * 5 feats = 20 feats
        self.context_char_dim = context_window * 2 * 20
        self.context_text_dim = context_window * 2 * 5
        
        # Current line: Character embedding
        self.char_embedding = layers.Embedding(128, 64, mask_zero=False, name='char_embedding')
        self.bi_lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=False), name='bi_lstm')
        self.lstm_dropout = layers.Dropout(0.3)
        
        # Current line: Text features
        self.text_dense = layers.Dense(64, activation='relu', name='text_dense')
        self.text_bn = layers.BatchNormalization()
        self.text_dropout = layers.Dropout(0.3)
        
        # Current line: Position features
        self.pos_dense = layers.Dense(32, activation='relu', name='pos_dense')
        
        # Context: Character features
        self.context_chars_dense = layers.Dense(64, activation='relu', name='context_chars_dense')
        
        # Context: Text features
        self.context_text_dense = layers.Dense(32, activation='relu', name='context_text_dense')
        
        # Merge and classify
        self.merge_dense = layers.Dense(128, activation='relu', name='merge_dense')
        self.merge_bn = layers.BatchNormalization()
        self.merge_dropout = layers.Dropout(0.4)
        self.output_layer = layers.Dense(num_classes, activation='softmax', name='output')
    
    def call(self, inputs, training=False):
        """Forward pass.
        
        Args:
            inputs: tuple of (char_ids, text_feats, pos_feats, context_chars, context_text)
        """
        char_ids, text_feats, pos_feats, context_chars, context_text = inputs
        
        # Current line: Text content
        x_chars = self.char_embedding(char_ids)
        x_chars = self.bi_lstm(x_chars, training=training)
        x_chars = self.lstm_dropout(x_chars, training=training)
        
        # Current line: Text features
        x_text = self.text_dense(text_feats)
        x_text = self.text_bn(x_text, training=training)
        x_text = self.text_dropout(x_text, training=training)
        
        # Current line: Position
        x_pos = self.pos_dense(pos_feats)
        
        # Context: Characters (cast to float first)
        context_chars_float = tf.cast(context_chars, tf.float32)
        x_context_chars = self.context_chars_dense(context_chars_float)
        
        # Context: Text features
        x_context_text = self.context_text_dense(context_text)
        
        # Merge all branches
        merged = tf.concat([x_chars, x_text, x_pos, x_context_chars, x_context_text], axis=-1)
        x = self.merge_dense(merged)
        x = self.merge_bn(x, training=training)
        x = self.merge_dropout(x, training=training)
        
        return self.output_layer(x)
    
    def get_config(self):
        return {
            'num_classes': self.num_classes,
            'context_window': self.context_window
        }


def load_v4_context_model(weights_path, num_classes=NUM_CLASSES):
    """Load V4 context-aware model dengan weights.
    
    Args:
        weights_path: Path ke best_weights.weights.h5
        num_classes: Number of output classes
    
    Returns:
        Trained ContextAwareClassifier model
    """
    model = ContextAwareClassifier(num_classes=num_classes)
    
    # Build model dengan dummy input
    dummy_chars = tf.zeros((1, MAX_TEXT_LENGTH), dtype=tf.int32)
    dummy_text = tf.zeros((1, 10), dtype=tf.float32)
    dummy_pos = tf.zeros((1, 5), dtype=tf.float32)
    dummy_context_chars = tf.zeros((1, 2 * 2 * 20), dtype=tf.int32)  # CONTEXT_WINDOW=2
    dummy_context_text = tf.zeros((1, 2 * 2 * 5), dtype=tf.float32)
    
    _ = model((dummy_chars, dummy_text, dummy_pos, dummy_context_chars, dummy_context_text), training=False)
    
    # Load weights
    model.load_weights(str(weights_path))
    
    return model
