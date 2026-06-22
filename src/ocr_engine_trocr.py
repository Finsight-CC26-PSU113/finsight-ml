"""
OCR FinSight - TrOCR Engine (Microsoft Transformer-based OCR)
===============================================================
TrOCR wrapper untuk receipt OCR dengan accuracy tinggi.

TrOCR advantages:
- Transformer-based (state-of-the-art)
- Excellent for printed text (receipts, documents)
- No aggressive horizontal merging
- Better layout understanding
- Can be fine-tuned

Installation:
    pip install transformers torch pillow
    
Usage:
    from src.ocr_engine_trocr import TrOCREngine
    engine = TrOCREngine()
    lines = engine.read_receipt(image)
"""

import numpy as np
import torch
from PIL import Image
from typing import Optional, List, Dict
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from src.config import OCR_GPU
from src.text_post_processor import get_post_processor


class TrOCREngine:
    """Wrapper TrOCR dengan output terstruktur compatible dengan OCREngine.
    
    Uses sliding window approach untuk extract text dari full receipt image.
    
    Usage:
        engine = TrOCREngine()
        lines = engine.read_receipt(image)
    """
    
    _instance: Optional['TrOCREngine'] = None
    _processor: Optional[TrOCRProcessor] = None
    _model: Optional[VisionEncoderDecoderModel] = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton: satu instance TrOCREngine untuk satu runtime."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, gpu: bool = None):
        """Initialize TrOCR model.
        
        Args:
            gpu: Use GPU (default: auto-detect from config)
        """
        if self._processor is not None:
            return  # Sudah diinisialisasi
        
        # Auto-detect GPU
        if gpu is None:
            self._gpu = OCR_GPU and torch.cuda.is_available()
        else:
            self._gpu = gpu and torch.cuda.is_available()
        
        print(f"[TrOCR] Loading Microsoft TrOCR model (gpu={self._gpu})...")
        
        try:
            # Load processor and model
            self._processor = TrOCRProcessor.from_pretrained(
                'microsoft/trocr-base-printed'
            )
            self._model = VisionEncoderDecoderModel.from_pretrained(
                'microsoft/trocr-base-printed'
            )
            
            # Move to GPU if available
            if self._gpu:
                self._model = self._model.to('cuda')
                print("[TrOCR] Model moved to GPU")
            
            self._model.eval()  # Set to evaluation mode
            
            print(f"[TrOCR] Ready ✓")
        except Exception as e:
            print(f"[TrOCR] ❌ Initialization failed: {e}")
            raise
    
    def read_receipt(self, image: np.ndarray, apply_post_processing: bool = True) -> List[Dict]:
        """Baca teks dari gambar struk menggunakan line-by-line approach.
        
        Args:
            image: Gambar (BGR atau grayscale) sebagai numpy array.
            apply_post_processing: Apply text corrections (fix O/0, I/1, etc.)
            
        Returns:
            List of dicts, sorted dari atas ke bawah, setiap dict berisi:
            - text: string teks yang terbaca
            - bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            - confidence: float 0-1
            - y_center: float, posisi vertikal tengah
            - x_center: float, posisi horizontal tengah
            - width: float, lebar bbox
            - height: float, tinggi bbox
            - line_index: int, urutan baris (0 = paling atas)
        """
        import cv2
        
        # Convert to PIL Image (RGB)
        if len(image.shape) == 2:
            # Grayscale to RGB
            image_pil = Image.fromarray(image).convert('RGB')
        elif image.shape[2] == 3:
            # BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
        else:
            image_pil = Image.fromarray(image)
        
        # Use line detection approach: split image into horizontal bands
        lines = self._extract_lines_sliding_window(image_pil, image.shape[:2])
        
        if not lines:
            return []
        
        # Sort dari atas ke bawah
        lines = sorted(lines, key=lambda l: l['y_center'])
        
        # Add line_index
        for i, line in enumerate(lines):
            line['line_index'] = i
        
        # Apply post-processing to fix common OCR errors
        if apply_post_processing:
            post_processor = get_post_processor()
            lines = post_processor.process_lines(lines)
        
        return lines
    
    def _extract_lines_sliding_window(self, image_pil: Image.Image, original_shape: tuple) -> List[Dict]:
        """Extract text lines using sliding window approach.
        
        Strategy: Split image into horizontal bands (simulating line detection)
        then run TrOCR on each band.
        """
        width, height = image_pil.size
        
        # Estimate line height (typically 2-4% of image height for receipts)
        estimated_line_height = max(int(height * 0.035), 30)  # Min 30px
        stride = max(int(estimated_line_height * 0.8), 20)  # Min 20px to prevent infinite loop
        
        lines = []
        y_pos = 0
        max_iterations = 1000  # Safety limit to prevent infinite loop
        iteration = 0
        
        while y_pos < height and iteration < max_iterations:
            iteration += 1
            
            # Extract horizontal band
            y_end = min(y_pos + estimated_line_height * 2, height)  # 2x height for context
            
            # Skip if band is too small
            if y_end - y_pos < 10:
                break
            
            band = image_pil.crop((0, y_pos, width, y_end))
            
            # Run OCR on band
            text, confidence = self._recognize_text(band)
            
            if text and text.strip() and confidence > 0.5:
                # Calculate normalized bbox
                y_min_norm = y_pos / original_shape[0]
                y_max_norm = y_end / original_shape[0]
                y_center_norm = (y_min_norm + y_max_norm) / 2
                
                lines.append({
                    'text': text.strip(),
                    'bbox': [
                        [0, y_pos], 
                        [width, y_pos],
                        [width, y_end],
                        [0, y_end]
                    ],
                    'confidence': confidence,
                    'y_center': y_center_norm,
                    'x_center': 0.5,
                    'width': 1.0,
                    'height': (y_end - y_pos) / original_shape[0],
                    'x_min': 0.0,
                    'y_min': y_min_norm,
                    'x_max': 1.0,
                    'y_max': y_max_norm,
                })
            
            y_pos += stride
        
        if iteration >= max_iterations:
            print(f"[TrOCR] Warning: Reached max iterations ({max_iterations})")
        
        # Merge overlapping detections
        lines = self._merge_overlapping_lines(lines)
        
        return lines
    
    def _recognize_text(self, image: Image.Image) -> tuple[str, float]:
        """Run TrOCR inference on image patch.
        
        Returns:
            (text, confidence) tuple
        """
        try:
            # Prepare image
            pixel_values = self._processor(
                image, 
                return_tensors="pt"
            ).pixel_values
            
            if self._gpu:
                pixel_values = pixel_values.to('cuda')
            
            # Generate text
            with torch.no_grad():
                generated_ids = self._model.generate(
                    pixel_values,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode text
            text = self._processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # TrOCR doesn't return confidence directly, use beam score as proxy
            # For now, use fixed high confidence if text is detected
            confidence = 0.95 if text.strip() else 0.0
            
            return text, confidence
            
        except Exception as e:
            print(f"[TrOCR] Recognition error: {e}")
            return "", 0.0
    
    def _merge_overlapping_lines(self, lines: List[Dict]) -> List[Dict]:
        """Merge lines yang overlap (dari sliding window)."""
        if not lines:
            return lines
        
        # Sort by y_center
        lines = sorted(lines, key=lambda l: l['y_center'])
        
        merged = []
        current = lines[0]
        
        for next_line in lines[1:]:
            # Check vertical overlap
            overlap = (
                min(current['y_max'], next_line['y_max']) - 
                max(current['y_min'], next_line['y_min'])
            )
            
            if overlap > current['height'] * 0.5:
                # Significant overlap - check if same text or extension
                if current['text'] in next_line['text'] or next_line['text'] in current['text']:
                    # Keep longer text
                    if len(next_line['text']) > len(current['text']):
                        current = next_line
                else:
                    # Different text - keep both if text is substantially different
                    merged.append(current)
                    current = next_line
            else:
                # No overlap - save current and move to next
                merged.append(current)
                current = next_line
        
        # Add last line
        merged.append(current)
        
        return merged
    
    def read_receipt_with_image_info(self, image: np.ndarray) -> tuple[List[Dict], Dict]:
        """Baca teks + return metadata gambar untuk fitur posisi.
        
        Returns:
            Tuple (lines, image_info) dimana image_info berisi:
            - height: tinggi gambar
            - width: lebar gambar
            - total_lines: jumlah baris terdeteksi
        """
        lines = self.read_receipt(image)
        
        h, w = image.shape[:2]
        
        image_info = {
            'height': h,
            'width': w,
            'total_lines': len(lines),
        }
        
        return lines, image_info
