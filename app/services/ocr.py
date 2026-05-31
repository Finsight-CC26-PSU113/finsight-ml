"""
OCR FinSight - OCR Engine
EasyOCR wrapper returning structured text lines from a receipt image.
"""

import easyocr
import numpy as np
from typing import Optional
from app.config import OCR_LANGUAGES, OCR_GPU
from app.services.text_post_processor import get_post_processor


class OCREngine:
    """Singleton EasyOCR wrapper with structured output."""

    _instance: Optional['OCREngine'] = None
    _reader: Optional[easyocr.Reader] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, languages: list[str] = None, gpu: bool = None):
        if self._reader is not None:
            return

        self._languages = languages or OCR_LANGUAGES

        if gpu is None:
            if OCR_GPU:
                try:
                    import torch
                    self._gpu = torch.cuda.is_available()
                    if not self._gpu:
                        print("[OCREngine] GPU not available, using CPU")
                except ImportError:
                    self._gpu = False
            else:
                self._gpu = False
        else:
            self._gpu = gpu

        print(f"[OCREngine] Loading EasyOCR (languages={self._languages}, gpu={self._gpu})...")
        self._reader = easyocr.Reader(self._languages, gpu=self._gpu, verbose=False)
        print("[OCREngine] Ready.")

    def read_receipt(self, image: np.ndarray, apply_post_processing: bool = True) -> list[dict]:
        """Read text from a receipt image.

        Returns sorted list of dicts: text, bbox, confidence, y/x center, width, height, line_index.
        """
        raw = self._reader.readtext(image)
        if not raw:
            return []

        lines = []
        for bbox, text, conf in raw:
            arr = np.array(bbox)
            x_min, x_max = float(arr[:, 0].min()), float(arr[:, 0].max())
            y_min, y_max = float(arr[:, 1].min()), float(arr[:, 1].max())
            lines.append({
                'text': text.strip(),
                'bbox': bbox,
                'confidence': float(conf),
                'y_center': (y_min + y_max) / 2,
                'x_center': (x_min + x_max) / 2,
                'width': x_max - x_min,
                'height': y_max - y_min,
                'x_min': x_min, 'y_min': y_min,
                'x_max': x_max, 'y_max': y_max,
            })

        lines = self._sort_lines(lines)
        for i, line in enumerate(lines):
            line['line_index'] = i

        if apply_post_processing:
            lines = get_post_processor().process_lines(lines)

        return lines

    def _sort_lines(self, lines: list[dict]) -> list[dict]:
        """Sort top-to-bottom, left-to-right, grouping same-row tokens."""
        if not lines:
            return lines
        avg_height = np.mean([l['height'] for l in lines])
        threshold = avg_height * 0.6
        lines.sort(key=lambda l: l['y_center'])

        grouped, current = [], [lines[0]]
        for line in lines[1:]:
            if abs(line['y_center'] - current[0]['y_center']) < threshold:
                current.append(line)
            else:
                current.sort(key=lambda l: l['x_center'])
                grouped.extend(current)
                current = [line]
        current.sort(key=lambda l: l['x_center'])
        grouped.extend(current)
        return grouped

    def read_receipt_with_image_info(self, image: np.ndarray) -> tuple[list[dict], dict]:
        """Read text and normalize bbox coordinates to 0-1."""
        lines = self.read_receipt(image)
        h, w = image.shape[:2]
        for line in lines:
            line['x_min'] /= w;  line['x_max'] /= w
            line['y_min'] /= h;  line['y_max'] /= h
            line['x_center'] /= w; line['y_center'] /= h
            line['width'] /= w;  line['height'] /= h
        return lines, {'height': h, 'width': w, 'total_lines': len(lines)}
