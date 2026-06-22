# 📊 TensorBoard Integration Guide

Complete guide for monitoring OCR FinSight classifier training with TensorBoard.

---

## 🎯 Quick Start

### Step 1: Train Model (with TensorBoard logging)

```bash
python scripts/train_classifier_v5_indonesia.py
```

TensorBoard logs are automatically saved to:

```
models/classifier_v5_indonesia/logs/
```

### Step 2: Launch TensorBoard

**Option A: Using convenience script**

```bash
python scripts/view_tensorboard.py
```

**Option B: Direct command**

```bash
tensorboard --logdir=models/classifier_v5_indonesia/logs --port=6006 --host=0.0.0.0
```

### Step 3: Open Browser

Navigate to: **http://localhost:6006**

---

## 📈 Available Dashboards

### 1. Scalars Dashboard

**Metrics tracked per epoch:**

| Metric                | Description                              |
| --------------------- | ---------------------------------------- |
| `epoch_loss`          | Training loss (categorical crossentropy) |
| `epoch_accuracy`      | Training accuracy (%)                    |
| `epoch_val_loss`      | Validation loss                          |
| `epoch_val_accuracy`  | Validation accuracy (%)                  |
| `epoch_learning_rate` | Current learning rate                    |

**How to use:**

- Monitor overfitting: val_loss should decrease with train_loss
- Check convergence: Both losses should plateau
- Verify learning: Accuracy should increase steadily

**Example interpretation:**

```
Epoch 10:
- train_loss: 0.45 ↓
- val_loss: 0.52 ↓
- train_acc: 86.5% ↑
- val_acc: 83.2% ↑
→ Model is learning well, no overfitting yet

Epoch 25:
- train_loss: 0.22 ↓
- val_loss: 0.65 ↑
- train_acc: 94.1% ↑
- val_acc: 81.5% ↓
→ OVERFITTING! Early stopping should trigger soon
```

### 2. Histograms Dashboard

**Visualizes weight distributions:**

- `char_embedding/embeddings` - Character embedding weights
- `bi_lstm/kernel` - LSTM weight matrix
- `merge_dense/kernel` - Final dense layer weights
- `output/bias` - Output layer biases

**How to use:**

- Check weight initialization: Should be centered around 0
- Monitor dead neurons: Biases stuck at 0 or extreme values
- Detect gradient vanishing: Weights not updating (flat histogram)

### 3. Graphs Dashboard

**Model architecture visualization:**

Shows complete TensorFlow computation graph with:

- Input nodes: `char_ids`, `text_feats`, `pos_feats`, `context_chars`, `context_text`
- Hidden layers: Embedding → BiLSTM → Dense → Merge
- Output node: Softmax(12 classes)

**How to use:**

- Verify architecture correctness
- Check tensor shapes at each layer
- Debug connection issues

---

## 🔍 Reading Training Logs

### Example: Good Training Run

```
Epoch 1/100
loss: 1.234 - accuracy: 0.456 - val_loss: 1.123 - val_accuracy: 0.478
↓
Epoch 5/100
loss: 0.856 - accuracy: 0.687 - val_loss: 0.912 - val_accuracy: 0.651
↓
Epoch 10/100
loss: 0.543 - accuracy: 0.823 - val_loss: 0.612 - val_accuracy: 0.789
↓
Epoch 15/100
loss: 0.387 - accuracy: 0.862 - val_loss: 0.521 - val_accuracy: 0.826
✅ Best model so far - weights saved
↓
Epoch 20/100
loss: 0.312 - accuracy: 0.881 - val_loss: 0.534 - val_accuracy: 0.819
⚠️ Validation accuracy decreased
↓
Epoch 25/100
EarlyStopping patience exceeded, restoring best weights from epoch 15
✅ Training complete!
```

### Example: Problematic Training Run

**Problem 1: Not learning**

```
Epoch 1: loss: 2.485 - accuracy: 0.083
Epoch 5: loss: 2.478 - accuracy: 0.085
Epoch 10: loss: 2.471 - accuracy: 0.087
→ Learning rate too low OR bad initialization
```

**Problem 2: Overfitting early**

```
Epoch 5: train_acc: 0.95 - val_acc: 0.65
→ Model memorizing training data
→ Need more data augmentation or dropout
```

**Problem 3: Exploding gradients**

```
Epoch 3: loss: 1.234
Epoch 4: loss: 5.678
Epoch 5: loss: NaN
→ Learning rate too high
→ Need gradient clipping
```

---

## 🎨 Comparing Multiple Runs

### Scenario: Testing different context windows

**Run 1: Context window ±2 lines**

```bash
# Modify CONTEXT_WINDOW = 2 in training script
python scripts/train_classifier_v5_indonesia.py

# Logs saved to: models/classifier_v5_indonesia/logs/
```

**Run 2: Context window ±4 lines**

```bash
# Change log directory in training script:
log_dir = output_dir / "logs_context4"

# Modify CONTEXT_WINDOW = 4
python scripts/train_classifier_v5_indonesia.py
```

**View both in TensorBoard:**

```bash
tensorboard --logdir=models/classifier_v5_indonesia --port=6006
```

TensorBoard will show both runs side-by-side with different colors!

---

## 📁 TensorBoard File Structure

```
models/classifier_v5_indonesia/
├── logs/
│   └── train/
│       ├── events.out.tfevents.1234567890.hostname
│       ├── events.out.tfevents.1234567891.hostname
│       └── ...
│
├── training_history.csv          # CSV backup (for pandas analysis)
├── best_weights.weights.h5       # Best model weights
└── final_weights.weights.h5      # Final epoch weights
```

**File sizes:**

- Each event file: ~5-10MB per 50 epochs
- Total logs: ~20-50MB for full training

---

## 🛠️ Advanced Usage

### Custom Metrics

Add custom metrics to training script:

```python
# In train_classifier_v5_indonesia.py

# Define custom metric
def f1_score_macro(y_true, y_pred):
    # Implementation
    return score

# Add to model compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', f1_score_macro]  # Added!
)
```

Now `f1_score_macro` appears in TensorBoard!

### Per-Class Accuracy Logging

Log critical class accuracy:

```python
# Add custom callback
class PerClassAccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Calculate per-class accuracy
        # Log to TensorBoard
        tf.summary.scalar('grand_total_acc', gt_acc, step=epoch)

callbacks.append(PerClassAccuracyCallback())
```

### Export Plots

From TensorBoard UI:

1. Click on plot
2. Click download icon (bottom left)
3. Choose SVG or PNG format

---

## 🐛 Troubleshooting

### Problem: "No dashboards are active for the current data set"

**Cause**: No event files found

**Solution**:

```bash
# Check if logs exist
ls models/classifier_v5_indonesia/logs/train/

# If empty, retrain model
python scripts/train_classifier_v5_indonesia.py
```

### Problem: TensorBoard shows old runs

**Solution**:

```bash
# Option 1: Clear old logs
rm -rf models/classifier_v5_indonesia/logs/*

# Option 2: Use new log directory
# In training script:
log_dir = output_dir / "logs_v2"
```

### Problem: Port 6006 already in use

**Solution**:

```bash
# Use different port
tensorboard --logdir=models/classifier_v5_indonesia/logs --port=6007

# Or kill existing process
# Windows:
taskkill /F /IM tensorboard.exe

# Linux/Mac:
pkill -f tensorboard
```

### Problem: Can't access from remote machine

**Solution**:

```bash
# Bind to all interfaces
tensorboard --logdir=logs --port=6006 --host=0.0.0.0

# Then access via: http://<server-ip>:6006
```

---

## 📊 Best Practices

### 1. Log Everything

✅ Always enable TensorBoard during training
✅ Keep CSV backup (training_history.csv) for offline analysis
✅ Save logs in git (if <50MB) for reproducibility

### 2. Monitor Regularly

✅ Check TensorBoard every 5-10 epochs during training
✅ Watch for overfitting (val_loss increasing)
✅ Verify learning rate schedule is working

### 3. Compare Experiments

✅ Use descriptive log directory names:

- `logs_context2_lr0.001`
- `logs_context4_lr0.001`
- `logs_context4_boosted`

✅ Document changes in commit messages

### 4. Clean Up Old Runs

❌ Don't accumulate 100s of GB of old logs
✅ Archive important runs, delete failed experiments
✅ Keep only last 5-10 runs

---

## 📚 Additional Resources

### Official Documentation

- [TensorBoard Guide](https://www.tensorflow.org/tensorboard/get_started)
- [TensorFlow Callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)

### Useful Commands

```bash
# View logs from multiple models
tensorboard --logdir=models --port=6006

# Profile performance
tensorboard --logdir=logs --port=6006 --load_fast=false

# Bind to specific IP
tensorboard --logdir=logs --bind_all
```

---

## 📈 Example Screenshots

_After training, TensorBoard should show:_

### Scalars Tab:

- **epoch_accuracy**: Increasing from ~45% to ~86%
- **epoch_val_accuracy**: Increasing from ~48% to ~83%
- **epoch_loss**: Decreasing from ~1.2 to ~0.4
- **epoch_val_loss**: Decreasing from ~1.1 to ~0.5

### Histograms Tab:

- **char_embedding/embeddings**: Normal distribution centered at 0
- **bi_lstm/kernel**: Gradual weight updates over epochs
- **output/bias**: Slight shift towards negative values (sparse classes)

### Graphs Tab:

- Full model architecture with 5 inputs and 1 output
- Clear flow: Inputs → Embeddings → LSTM → Dense → Softmax

---

**🎯 With TensorBoard, you can confidently monitor training and achieve optimal model performance!**
