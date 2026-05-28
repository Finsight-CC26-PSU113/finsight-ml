"""
OCR FinSight - OCR Engine
EasyOCR wrapper yang mengembalikan baris teks terstruktur dari gambar struk.
"""

import easyocr
import numpy as np
from typing import Optional
from src.config import OCR_LANGUAGES, OCR_GPU
from src.text_post_processor import get_post_processor


class OCREngine:
    """Wrapper EasyOCR dengan output terstruktur.
    
    Menggunakan singleton pattern agar model EasyOCR hanya
    di-load sekali ke memori GPU.
    
    Usage:
        engine = OCREngine()
        lines = engine.read_receipt(image)
        # lines = [
        #     {"text": "INDOMARET", "bbox": [...], "confidence": 0.95,
        #      "y_center": 120, "x_center": 300, "width": 200, "height": 25,
        #      "line_index": 0},
        #     ...
        # ]
    """
    
    _instance: Optional['OCREngine'] = None
    _reader: Optional[easyocr.Reader] = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton: satu instance OCREngine untuk satu runtime."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, languages: list[str] = None, gpu: bool = None):
        """Initialize EasyOCR reader.
        
        Args:
            languages: List kode bahasa (default dari config).
            gpu: Gunakan GPU (default: auto-detect, fall back to CPU).
        """
        if self._reader is not None:
            return  # Sudah diinisialisasi
            
        self._languages = languages or OCR_LANGUAGES
        
        # Auto-detect GPU availability for cross-platform compatibility
        if gpu is None:
            gpu_requested = OCR_GPU
            if gpu_requested:
                try:
                    import torch
                    gpu_available = torch.cuda.is_available()
                    if not gpu_available:
                        print("[OCREngine] GPU requested but not available, falling back to CPU")
                    self._gpu = gpu_available
                except ImportError:
                    print("[OCREngine] PyTorch not available, using CPU")
                    self._gpu = False
            else:
                self._gpu = False
        else:
            self._gpu = gpu
        
        print(f"[OCREngine] Loading EasyOCR model (languages={self._languages}, gpu={self._gpu})...")
        self._reader = easyocr.Reader(
            self._languages,
            gpu=self._gpu,
            verbose=False
        )
        print("[OCREngine] Model loaded.")
    
    def read_receipt(self, image: np.ndarray, apply_post_processing: bool = True) -> list[dict]:
        """Baca teks dari gambar struk.
        
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
        # EasyOCR accepts both BGR and grayscale
        raw_results = self._reader.readtext(image)
        
        if not raw_results:
            return []
        
        # Parse raw results ke format terstruktur
        lines = []
        for bbox, text, confidence in raw_results:
            # bbox = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            bbox_arr = np.array(bbox)
            
            x_min = float(bbox_arr[:, 0].min())
            x_max = float(bbox_arr[:, 0].max())
            y_min = float(bbox_arr[:, 1].min())
            y_max = float(bbox_arr[:, 1].max())
            
            lines.append({
                'text': text.strip(),
                'bbox': bbox,
                'confidence': float(confidence),
                'y_center': (y_min + y_max) / 2,
                'x_center': (x_min + x_max) / 2,
                'width': x_max - x_min,
                'height': y_max - y_min,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
            })
        
        # Sort dari atas ke bawah
        lines = self._sort_lines(lines)
        
        # Tambahkan line_index
        for i, line in enumerate(lines):
            line['line_index'] = i
        
        # Apply post-processing to fix common OCR errors
        if apply_post_processing:
            post_processor = get_post_processor()
            lines = post_processor.process_lines(lines)
        
        return lines
    
    def _sort_lines(self, lines: list[dict]) -> list[dict]:
        """Sort baris dari atas ke bawah, lalu kiri ke kanan.
        
        Baris yang punya y_center mirip (selisih < threshold) 
        dianggap satu baris horizontal → sort by x.
        """
        if not lines:
            return lines
        
        # Hitung threshold: rata-rata tinggi baris / 2
        avg_height = np.mean([l['height'] for l in lines])
        y_threshold = avg_height * 0.6
        
        # Sort by y_center dulu
        lines.sort(key=lambda l: l['y_center'])
        
        # Group baris yang punya y_center mirip
        grouped = []
        current_group = [lines[0]]
        
        for i in range(1, len(lines)):
            if abs(lines[i]['y_center'] - current_group[0]['y_center']) < y_threshold:
                current_group.append(lines[i])
            else:
                # Sort group by x, lalu tambahkan
                current_group.sort(key=lambda l: l['x_center'])
                grouped.extend(current_group)
                current_group = [lines[i]]
        
        # Jangan lupa group terakhir
        current_group.sort(key=lambda l: l['x_center'])
        grouped.extend(current_group)
        
        return grouped
    
    def read_receipt_with_image_info(self, image: np.ndarray) -> tuple[list[dict], dict]:
        """Baca teks + return metadata gambar untuk fitur posisi.
        
        Returns:
            Tuple (lines, image_info) dimana image_info berisi:
            - height: tinggi gambar
            - width: lebar gambar
            - total_lines: jumlah baris terdeteksi
        """
        lines = self.read_receipt(image)
        
        h, w = image.shape[:2]
        
        # Normalize bbox coordinates to 0-1 for UI
        for line in lines:
            line['x_min'] = line['x_min'] / w
            line['x_max'] = line['x_max'] / w
            line['y_min'] = line['y_min'] / h
            line['y_max'] = line['y_max'] / h
            line['x_center'] = line['x_center'] / w
            line['y_center'] = line['y_center'] / h
            line['width'] = line['width'] / w
            line['height'] = line['height'] / h
        
        image_info = {
            'height': h,
            'width': w,
            'total_lines': len(lines),
        }
        
        return lines, image_info
