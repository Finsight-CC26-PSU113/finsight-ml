"""
OCR FinSight - Configuration
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# OCR
OCR_LANGUAGES = ['en', 'id']
OCR_GPU = True

# Image preprocessing
MAX_IMAGE_SIZE = 1920

# Line classifier classes
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
MAX_TEXT_LENGTH = 100
NUM_POSITION_FEATURES = 5
NUM_STATISTICAL_FEATURES = 10

# Training splits
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# FastAPI
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
