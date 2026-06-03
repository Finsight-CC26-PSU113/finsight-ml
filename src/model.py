"""
OCR FinSight - TensorFlow Receipt Line Classifier
Model Subclassing + Custom Layers + Custom Loss + Custom Callback + GradientTape + TensorBoard

Mengklasifikasi setiap baris OCR menjadi: STORE, ADDRESS, DATE, ITEM, TOTAL, OTHER
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import (
    NUM_CLASSES, LINE_CLASSES, EMBEDDING_DIM, LSTM_UNITS, DENSE_UNITS,
    DROPOUT_RATE, LEARNING_RATE, BATCH_SIZE, EPOCHS,
    MAX_TEXT_LENGTH, NUM_POSITION_FEATURES, NUM_STATISTICAL_FEATURES,
    TRAIN_SPLIT, RANDOM_SEED, LOGS_DIR, MODELS_DIR,
    SAVED_MODEL_DIR, KERAS_MODEL_PATH, ROOT_DIR
)


# ==============================================================================
# Custom Layers
# ==============================================================================

class TextFeatureLayer(tf.keras.layers.Layer):
    """Extract & encode statistical features dari teks.
    
    Input: raw statistical features (10 dims)
    Output: encoded features via dense projection
    """
    
    def __init__(self, output_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', name='text_feat_dense1')
        self.bn = tf.keras.layers.BatchNormalization(name='text_feat_bn')
        self.dense2 = tf.keras.layers.Dense(output_dim, activation='relu', name='text_feat_dense2')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn(x, training=training)
        x = self.dense2(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({'output_dim': self.output_dim})
        return config


class PositionFeatureLayer(tf.keras.layers.Layer):
    """Encode spatial position features.
    
    Input: position features (5 dims: y_ratio, x_ratio, width_ratio, height_ratio, line_index_ratio)
    Output: encoded position embedding
    """
    
    def __init__(self, output_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', name='pos_feat_dense1')
        self.dense2 = tf.keras.layers.Dense(output_dim, activation='relu', name='pos_feat_dense2')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({'output_dim': self.output_dim})
        return config


# ==============================================================================
# Custom Loss
# ==============================================================================

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss untuk mengatasi extreme class imbalance.
    
    Penalti eksponensial untuk kelas yang susah ditebak.
    """
    
    def __init__(self, gamma=2.0, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        if alpha is not None:
            self.alpha = tf.constant(alpha, dtype=tf.float32)
        else:
            self.alpha = None
            
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
    """Log per-class accuracy ke TensorBoard setiap epoch.
    
    Memudahkan monitoring class mana yang perform bagus/jelek.
    """
    
    def __init__(self, validation_data, log_dir, class_names=None, classifier_model=None, **kwargs):
        super().__init__(**kwargs)
        self.val_features = validation_data[0]
        self.val_labels = validation_data[1]
        self.class_names = class_names or [LINE_CLASSES[i] for i in range(NUM_CLASSES)]
        self.writer = tf.summary.create_file_writer(str(log_dir / "per_class"))
        self._classifier_model = classifier_model
    
    def on_epoch_end(self, epoch, logs=None):
        predictions = self._classifier_model(self.val_features, training=False)
        pred_classes = tf.argmax(predictions, axis=-1)
        true_classes = tf.argmax(self.val_labels, axis=-1)
        
        with self.writer.as_default():
            for i, name in enumerate(self.class_names):
                mask = tf.equal(true_classes, i)
                if tf.reduce_sum(tf.cast(mask, tf.int32)) == 0:
                    continue
                
                class_preds = tf.boolean_mask(pred_classes, mask)
                class_acc = tf.reduce_mean(tf.cast(tf.equal(class_preds, i), tf.float32))
                
                tf.summary.scalar(f"accuracy/{name}", class_acc, step=epoch)
                
                if logs is not None:
                    logs[f'val_acc_{name}'] = float(class_acc)
            
            self.writer.flush()


# ==============================================================================
# Model (Model Subclassing)
# ==============================================================================

class ReceiptLineClassifier(tf.keras.Model):
    """Receipt line classifier menggunakan Model Subclassing.
    
    Architecture:
    - Branch 1: Character embedding → Bi-LSTM (text content)
    - Branch 2: TextFeatureLayer (statistical features)
    - Branch 3: PositionFeatureLayer (spatial features)
    - Merge → Dense → Dropout → Softmax (6 classes)
    """
    
    VOCAB_SIZE = 128  # ASCII characters
    
    def __init__(self, num_classes=NUM_CLASSES, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        # Branch 1: Character-level text encoding
        self.char_embedding = tf.keras.layers.Embedding(
            self.VOCAB_SIZE, EMBEDDING_DIM,
            mask_zero=False, name='char_embedding'  # Disable masking for cuDNN compatibility
        )
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=False, name='lstm'),
            name='bi_lstm'
        )
        self.text_dropout = tf.keras.layers.Dropout(DROPOUT_RATE, name='text_dropout')
        
        # Branch 2: Statistical text features
        self.text_feature_layer = TextFeatureLayer(output_dim=32, name='text_features')
        
        # Branch 3: Position features
        self.position_feature_layer = PositionFeatureLayer(output_dim=16, name='position_features')
        
        # Merge & classify
        self.merge_dense = tf.keras.layers.Dense(DENSE_UNITS, activation='relu', name='merge_dense')
        self.merge_bn = tf.keras.layers.BatchNormalization(name='merge_bn')
        self.merge_dropout = tf.keras.layers.Dropout(DROPOUT_RATE, name='merge_dropout')
        self.output_dense = tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')
    
    def call(self, inputs, training=False):
        """Forward pass.
        
        Args:
            inputs: tuple of (char_ids, text_features, position_features)
                - char_ids: [batch, MAX_TEXT_LENGTH] int tensor
                - text_features: [batch, 10] float tensor
                - position_features: [batch, 5] float tensor
        """
        char_ids, text_feats, pos_feats = inputs
        
        # Branch 1: Text content via Bi-LSTM
        x_text = self.char_embedding(char_ids)
        x_text = self.bi_lstm(x_text, training=training)
        x_text = self.text_dropout(x_text, training=training)
        
        # Branch 2: Statistical features
        x_stats = self.text_feature_layer(text_feats, training=training)
        
        # Branch 3: Position features
        x_pos = self.position_feature_layer(pos_feats, training=training)
        
        # Merge all branches
        merged = tf.concat([x_text, x_stats, x_pos], axis=-1)
        x = self.merge_dense(merged)
        x = self.merge_bn(x, training=training)
        x = self.merge_dropout(x, training=training)
        output = self.output_dense(x)
        
        return output


# ==============================================================================
# Data Loading
# ==============================================================================

def text_to_char_ids(text: str, max_length: int = MAX_TEXT_LENGTH) -> np.ndarray:
    """Convert text ke array of ASCII character IDs."""
    ids = [min(ord(c), 127) for c in text[:max_length]]
    # Pad to max_length
    ids = ids + [0] * (max_length - len(ids))
    return np.array(ids, dtype=np.int32)


def load_training_data(csv_path: Path = None):
    """Load labeled CSV dan prepare sebagai TF-ready arrays.
    
    Returns:
        (train_data, val_data, test_data) dimana setiap item adalah:
        ((char_ids, text_features, position_features), labels)
    """
    if csv_path is None:
        csv_path = ROOT_DIR / "data" / "labeled_lines.csv"
    
    print(f"[Data] Loading from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[Data] Total samples: {len(df)}")
    
    # Encode labels
    label_to_idx = {v: k for k, v in LINE_CLASSES.items()}
    df['label_idx'] = df['label'].map(label_to_idx)
    df = df.dropna(subset=['label_idx'])
    df['label_idx'] = df['label_idx'].astype(int)
    
    print(f"[Data] After filtering: {len(df)}")
    
    # === OVERSAMPLING ===
    print("[Data] Applying Oversampling to minority classes...")
    max_count = df['label'].value_counts().max() # biasanya OTHER (~4600)
    target_count = max(max_count // 3, 500) # Target min sekitar 1500
    
    dfs = []
    for label, group in df.groupby('label'):
        if len(group) < target_count:
            oversampled = group.sample(target_count, replace=True, random_state=RANDOM_SEED)
            dfs.append(oversampled)
        else:
            dfs.append(group)
            
    df = pd.concat(dfs).reset_index(drop=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"[Data] Total samples after oversampling: {len(df)}")
    
    print(f"[Data] Label distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"  {label}: {count}")
    
    # Character IDs
    char_ids = np.array([text_to_char_ids(str(t)) for t in df['text'].values])
    
    # Text features (10 dims)
    text_feat_cols = [
        'digit_ratio', 'alpha_ratio', 'upper_ratio', 'space_ratio', 'special_ratio',
        'char_count', 'word_count', 'has_currency', 'has_date_pattern', 'has_qty_pattern'
    ]
    text_features = df[text_feat_cols].values.astype(np.float32)
    
    # Normalize char_count and word_count
    for i, col in enumerate(text_feat_cols):
        if col == 'char_count':
            text_features[:, i] /= 100.0  # MAX_TEXT_LENGTH
        elif col == 'word_count':
            text_features[:, i] /= 20.0   # Assumed max word count
    
    # Position features (5 dims)
    pos_feat_cols = ['y_ratio', 'x_ratio', 'width_ratio', 'height_ratio', 'line_index_ratio']
    pos_features = df[pos_feat_cols].values.astype(np.float32)
    
    # One-hot labels
    labels = tf.keras.utils.to_categorical(df['label_idx'].values, num_classes=NUM_CLASSES)
    
    # Split: train/val/test
    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED,
                                            stratify=df['label_idx'].values)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=RANDOM_SEED,
                                          stratify=df['label_idx'].values[temp_idx])
    
    texts = df['text'].values.astype(str)
    
    def make_split(idx):
        return (
            (char_ids[idx], text_features[idx], pos_features[idx]),
            labels[idx],
            texts[idx]
        )
    
    train_data = make_split(train_idx)
    val_data = make_split(val_idx)
    test_data = make_split(test_idx)
    
    print(f"[Data] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return train_data, val_data, test_data


# ==============================================================================
# Custom Training Loop (GradientTape + TensorBoard)
# ==============================================================================

@tf.function
def train_step(model, inputs, labels, optimizer, loss_fn):
    """Single training step dengan GradientTape."""
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, predictions


@tf.function
def val_step(model, inputs, labels, loss_fn):
    """Single validation step."""
    predictions = model(inputs, training=False)
    loss = loss_fn(labels, predictions)
    return loss, predictions


def train_model(train_data, val_data, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Custom training loop dengan GradientTape dan TensorBoard logging.
    
    Returns:
        Trained model
    """
    print("\n" + "=" * 60)
    print("TRAINING - ReceiptLineClassifier")
    print("=" * 60)
    
    # Unpack data
    (train_chars, train_text_feats, train_pos_feats), train_labels, train_texts = train_data
    (val_chars, val_text_feats, val_pos_feats), val_labels, val_texts = val_data
    
    # Create TF datasets
    train_ds = tf.data.Dataset.from_tensor_slices((
        {'chars': train_chars, 'text_feats': train_text_feats, 'pos_feats': train_pos_feats},
        train_labels
    )).shuffle(10000, seed=RANDOM_SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((
        {'chars': val_chars, 'text_feats': val_text_feats, 'pos_feats': val_pos_feats},
        val_labels
    )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Dynamic weights dari train_labels
    train_counts = np.sum(train_labels, axis=0)
    total_train = np.sum(train_counts)
    
    # Inverse frequency weights: total / (NUM_CLASSES * count_i)
    alpha_weights = total_train / (NUM_CLASSES * (train_counts + 1))
    # Normalize agar sum = NUM_CLASSES
    alpha_weights = alpha_weights / np.sum(alpha_weights) * NUM_CLASSES
    
    print("\n[Training] Dynamic Class Weights (Alpha):")
    for i, w in enumerate(alpha_weights):
        print(f"  {LINE_CLASSES[i]:>8s}: {w:.4f}")
        
    # Init model, optimizer, loss
    model = ReceiptLineClassifier()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha_weights)
    
    # Build model with dummy input
    dummy_chars = tf.zeros((1, MAX_TEXT_LENGTH), dtype=tf.int32)
    dummy_text = tf.zeros((1, NUM_STATISTICAL_FEATURES), dtype=tf.float32)
    dummy_pos = tf.zeros((1, NUM_POSITION_FEATURES), dtype=tf.float32)
    model((dummy_chars, dummy_text, dummy_pos), training=False)
    model.summary()
    
    # TensorBoard writers
    log_dir = LOGS_DIR / f"run_{int(time.time())}"
    train_writer = tf.summary.create_file_writer(str(log_dir / "train"))
    val_writer = tf.summary.create_file_writer(str(log_dir / "val"))
    
    # Per-class callback
    val_inputs = (val_chars, val_text_feats, val_pos_feats)
    per_class_cb = PerClassAccuracyCallback(
        validation_data=(val_inputs, val_labels),
        log_dir=log_dir,
        classifier_model=model,
    )
    
    # Metrics
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='val_acc')
    
    best_val_acc = 0.0
    
    print(f"\nTensorBoard logs: {log_dir}", flush=True)
    print(f"Training for {epochs} epochs...\n", flush=True)
    
    for epoch in range(epochs):
        # Reset metrics
        train_loss_metric.reset_state()
        train_acc_metric.reset_state()
        val_loss_metric.reset_state()
        val_acc_metric.reset_state()
        
        # --- Training ---
        for batch in train_ds:
            features = batch[0]
            labels_batch = batch[1]
            inputs = (features['chars'], features['text_feats'], features['pos_feats'])
            
            loss, preds = train_step(model, inputs, labels_batch, optimizer, loss_fn)
            train_loss_metric.update_state(loss)
            train_acc_metric.update_state(labels_batch, preds)
        
        # --- Validation ---
        for batch in val_ds:
            features = batch[0]
            labels_batch = batch[1]
            inputs = (features['chars'], features['text_feats'], features['pos_feats'])
            
            loss, preds = val_step(model, inputs, labels_batch, loss_fn)
            val_loss_metric.update_state(loss)
            val_acc_metric.update_state(labels_batch, preds)
        
        # Get metrics
        t_loss = train_loss_metric.result().numpy()
        t_acc = train_acc_metric.result().numpy()
        v_loss = val_loss_metric.result().numpy()
        v_acc = val_acc_metric.result().numpy()
        
        # Log to TensorBoard
        with train_writer.as_default():
            tf.summary.scalar('loss', t_loss, step=epoch)
            tf.summary.scalar('accuracy', t_acc, step=epoch)
        
        with val_writer.as_default():
            tf.summary.scalar('loss', v_loss, step=epoch)
            tf.summary.scalar('accuracy', v_acc, step=epoch)
        
        # Per-class accuracy callback
        per_class_cb.on_epoch_end(epoch)
        
        # Print progress
        marker = " *" if v_acc > best_val_acc else ""
        print(f"  Epoch {epoch+1:>3d}/{epochs} | "
              f"loss: {t_loss:.4f} acc: {t_acc:.4f} | "
              f"val_loss: {v_loss:.4f} val_acc: {v_acc:.4f}{marker}", flush=True)
        
        # Save best model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model.save_weights(str(MODELS_DIR / "best_weights.weights.h5"))
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Weights saved to: {MODELS_DIR / 'best_weights.weights.h5'}")
    
    # Load best weights
    model.load_weights(str(MODELS_DIR / "best_weights.weights.h5"))
    
    return model


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_model(model, test_data):
    """Evaluate model pada test set."""
    (test_chars, test_text_feats, test_pos_feats), test_labels, test_texts = test_data
    
    predictions = model((test_chars, test_text_feats, test_pos_feats), training=False)
    pred_classes = tf.argmax(predictions, axis=-1).numpy()
    true_classes = tf.argmax(test_labels, axis=-1).numpy()
    
    # Overall accuracy
    accuracy = np.mean(pred_classes == true_classes)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    print("\nPer-class results:")
    print(f"  {'Class':>8s} | {'Correct':>7s} | {'Total':>5s} | {'Accuracy':>8s}")
    print(f"  {'-'*8} | {'-'*7} | {'-'*5} | {'-'*8}")
    
    for i in range(NUM_CLASSES):
        mask = true_classes == i
        total = mask.sum()
        if total == 0:
            continue
        correct = (pred_classes[mask] == i).sum()
        acc = correct / total
        print(f"  {LINE_CLASSES[i]:>8s} | {correct:>7d} | {total:>5d} | {acc:>8.4f}")
    
    return accuracy


# ==============================================================================
# Prediction Helper
# ==============================================================================

def predict_lines(model, ocr_lines: list[dict], image_info: dict) -> list[dict]:
    """Classify OCR lines menggunakan trained model.
    
    Args:
        model: Trained ReceiptLineClassifier
        ocr_lines: List dari OCREngine.read_receipt()
        image_info: Dict {height, width, total_lines}
    
    Returns:
        List of dicts dengan tambahan 'predicted_class' dan 'class_confidence'
    """
    from scripts.generate_labels import extract_text_features, extract_position_features
    
    if not ocr_lines:
        return []
    
    # Prepare features
    char_ids_list = []
    text_feats_list = []
    pos_feats_list = []
    
    text_feat_keys = [
        'digit_ratio', 'alpha_ratio', 'upper_ratio', 'space_ratio', 'special_ratio',
        'char_count', 'word_count', 'has_currency', 'has_date_pattern', 'has_qty_pattern'
    ]
    pos_feat_keys = ['y_ratio', 'x_ratio', 'width_ratio', 'height_ratio', 'line_index_ratio']
    
    for line in ocr_lines:
        char_ids_list.append(text_to_char_ids(line['text']))
        
        tf_dict = extract_text_features(line['text'])
        text_feats_list.append([tf_dict[k] for k in text_feat_keys])
        
        pf_dict = extract_position_features(line, image_info)
        pos_feats_list.append([pf_dict[k] for k in pos_feat_keys])
    
    char_ids = np.array(char_ids_list, dtype=np.int32)
    text_feats = np.array(text_feats_list, dtype=np.float32)
    pos_feats = np.array(pos_feats_list, dtype=np.float32)
    
    # Normalize char_count and word_count (index 5 and 6)
    text_feats[:, 5] /= 100.0  # MAX_TEXT_LENGTH
    text_feats[:, 6] /= 20.0   # Assumed max word count
    
    # Predict
    predictions = model((char_ids, text_feats, pos_feats), training=False)
    pred_classes = tf.argmax(predictions, axis=-1).numpy()
    confidences = tf.reduce_max(predictions, axis=-1).numpy()
    
    # Attach results
    results = []
    for i, line in enumerate(ocr_lines):
        result = dict(line)
        result['predicted_class'] = LINE_CLASSES[int(pred_classes[i])]
        result['class_confidence'] = float(confidences[i])
        result['class_probabilities'] = {
            LINE_CLASSES[j]: float(predictions[i][j]) for j in range(NUM_CLASSES)
        }
        results.append(result)
    
    return results
