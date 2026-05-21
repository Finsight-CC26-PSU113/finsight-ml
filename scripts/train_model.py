"""
OCR FinSight - Training Script
Jalankan: python -m scripts.train_model

Steps:
1. Load labeled data (dari scripts/generate_labels.py output)
2. Train ReceiptLineClassifier
3. Evaluate pada test set
4. Save model
"""

import sys
import os
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_training_data, train_model, evaluate_model
from src.config import EPOCHS, BATCH_SIZE, ROOT_DIR


def main():
    csv_path = ROOT_DIR / "data" / "labeled_lines.csv"
    
    if not csv_path.exists():
        print(f"ERROR: Training data not found at {csv_path}")
        print("Run 'python -m scripts.generate_labels' first!")
        sys.exit(1)
    
    # Load data
    train_data, val_data, test_data = load_training_data(csv_path)
    
    # Train
    model = train_model(train_data, val_data, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Evaluate
    evaluate_model(model, test_data)
    
    print("\nDone! You can now use the pipeline:")
    print("  from src.pipeline import OCRPipeline")
    print("  pipeline = OCRPipeline()")
    print("  result = pipeline.process('path/to/receipt.jpg')")


if __name__ == "__main__":
    main()
