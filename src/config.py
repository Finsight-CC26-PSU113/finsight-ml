"""
OCR FinSight - Configuration
Centralized configuration for all components.
"""

import os
from pathlib import Path

# ==============================================================================
# Paths
# ==============================================================================
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "fullDataset"
IMAGES_DIR = DATA_DIR / "images"
GDT_DIR = DATA_DIR / "gdt"
CLEAN_DATASET_DIR = ROOT_DIR / "clean_dataset"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# ==============================================================================
# OCR Settings (EasyOCR)
# ==============================================================================
OCR_LANGUAGES = ['en', 'id']  # English + Indonesian
OCR_GPU = True                 # Try GPU first, auto-fallback to CPU if unavailable

# ==============================================================================
# Image Preprocessing
# ==============================================================================
MAX_IMAGE_SIZE = 1920          # Max dimension for preprocessing
PREPROCESSING_VARIANTS = [
    'original',                 # No preprocessing
    'grayscale_thresh',         # Grayscale + adaptive threshold
    'denoise',                  # Denoise + grayscale
    'sharpen',                  # Sharpen + grayscale
]

# ==============================================================================
# Text Line Classification (TensorFlow Model)
# ==============================================================================
LINE_CLASSES = {
    0: 'STORE',
    1: 'ADDRESS_CONTACT',
    2: 'DATE',
    3: 'ITEM_DESC',
    4: 'ITEM_PRICE/QTY',
    5: 'TOTAL_PAYMENT',
    6: 'OTHER',
}
NUM_CLASSES = len(LINE_CLASSES)

# Model hyperparameters
EMBEDDING_DIM = 64
LSTM_UNITS = 128
DENSE_UNITS = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 50

# Feature dimensions
MAX_TEXT_LENGTH = 100          # Max characters per line
NUM_POSITION_FEATURES = 5     # y_ratio, x_ratio, width_ratio, height_ratio, line_index_ratio
NUM_STATISTICAL_FEATURES = 10 # digit_ratio, alpha_ratio, has_currency, etc.

# ==============================================================================
# Training
# ==============================================================================
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# ==============================================================================
# Export
# ==============================================================================
SAVED_MODEL_DIR = MODELS_DIR / "receipt_extractor"
KERAS_MODEL_PATH = MODELS_DIR / "receipt_extractor.keras"

# ==============================================================================
# FastAPI
# ==============================================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
