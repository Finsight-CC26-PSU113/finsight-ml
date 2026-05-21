"""
OCR FinSight - End-to-End Pipeline
Menggabungkan semua komponen: Preprocessing → OCR → Classification → Extraction
"""

import numpy as np
from pathlib import Path
from typing import Optional

from src.config import MODELS_DIR, MAX_TEXT_LENGTH, NUM_STATISTICAL_FEATURES, NUM_POSITION_FEATURES
from src.preprocessing import preprocess_for_ocr, load_image, deskew
from src.ocr_engine import OCREngine
from src.model import ReceiptLineClassifier, predict_lines
from src.extractor import ReceiptExtractor


class OCRPipeline:
    """End-to-end OCR pipeline untuk struk belanja.
    
    Usage:
        pipeline = OCRPipeline()
        result = pipeline.process("path/to/receipt.jpg")
        print(result['store'], result['total'])
    """
    
    def __init__(self, weights_path: Optional[str] = None):
        """Initialize semua komponen pipeline.
        
        Args:
            weights_path: Path ke model weights. Default: models/best_weights.weights.h5
        """
        print("[Pipeline] Initializing components...")
        
        # 1. OCR Engine (singleton, loaded once)
        self.ocr = OCREngine()
        
        # 2. TF Model
        self.model = ReceiptLineClassifier()
        
        # Build model with dummy input
        import tensorflow as tf
        dummy_chars = tf.zeros((1, MAX_TEXT_LENGTH), dtype=tf.int32)
        dummy_text = tf.zeros((1, NUM_STATISTICAL_FEATURES), dtype=tf.float32)
        dummy_pos = tf.zeros((1, NUM_POSITION_FEATURES), dtype=tf.float32)
        self.model((dummy_chars, dummy_text, dummy_pos), training=False)
        
        # Load weights
        if weights_path is None:
            weights_path = str(MODELS_DIR / "best_weights.weights.h5")
        
        if Path(weights_path).exists():
            self.model.load_weights(weights_path)
            print(f"[Pipeline] Model weights loaded from {weights_path}")
        else:
            print(f"[Pipeline] WARNING: No weights found at {weights_path}")
            print("[Pipeline] Model will use random weights (untrained)")
        
        # 3. Extractor
        self.extractor = ReceiptExtractor()
        
        print("[Pipeline] Ready.")
    
    def process(self, image_path: str | Path) -> dict:
        """Process satu gambar struk → structured data.
        
        Args:
            image_path: Path ke file gambar.
            
        Returns:
            Dict berisi:
            - store: nama toko
            - date: tanggal
            - items: list of {name, qty, price}
            - total: float
            - address: alamat
            - raw_lines: semua baris + klasifikasi
            - metadata: info processing
        """
        image_path = Path(image_path)
        
        # Step 1: Preprocessing
        img = load_image(image_path)
        img = deskew(img)
        
        # Step 2: OCR
        lines, image_info = self.ocr.read_receipt_with_image_info(img)
        
        if not lines:
            return {
                'store': '', 'date': '', 'items': [], 'total': 0.0,
                'address': '', 'raw_lines': [],
                'metadata': {'filename': image_path.name, 'ocr_lines': 0, 'status': 'no_text_detected'}
            }
        
        # Step 3: Classification
        classified_lines = predict_lines(self.model, lines, image_info)
        
        # Step 4: Extraction
        result = self.extractor.extract(classified_lines)
        
        # Add metadata
        result['metadata'] = {
            'filename': image_path.name,
            'ocr_lines': len(lines),
            'image_size': f"{image_info['width']}x{image_info['height']}",
            'status': 'success'
        }
        
        return result
    
    def process_batch(self, image_paths: list[str | Path]) -> list[dict]:
        """Process multiple images."""
        results = []
        for path in image_paths:
            try:
                result = self.process(path)
                results.append(result)
            except Exception as e:
                results.append({
                    'store': '', 'date': '', 'items': [], 'total': 0.0,
                    'metadata': {'filename': str(path), 'status': f'error: {e}'}
                })
        return results
