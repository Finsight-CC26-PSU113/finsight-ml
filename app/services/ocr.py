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

    def __init__(self, languages: list[str] = None, gpu: bool = None, model_path: str = None):
        """Initialize EasyOCR reader.

        Args:
            languages: Language codes (default from config).
            gpu: Use GPU (default: auto-detect, fall back to CPU).
            model_path: Path to fine-tuned recognizer weights (optional).
        """
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

        # Resolve fine-tuned model path (ignore if it doesn't exist)
        self._model_path = str(model_path) if model_path else None
        if self._model_path:
            from pathlib import Path
            if Path(self._model_path).exists():
                print(f"[OCREngine] Found fine-tuned model: {self._model_path}")
            else:
                print(f"[OCREngine] Fine-tuned model not found at {self._model_path}, using default model")
                self._model_path = None

        print(f"[OCREngine] Loading EasyOCR (languages={self._languages}, gpu={self._gpu})...")
        self._reader = easyocr.Reader(self._languages, gpu=self._gpu, verbose=False)

        # Load fine-tuned recognizer weights if available
        self.finetuned_loaded = False
        if self._model_path:
            try:
                import torch
                if hasattr(self._reader, 'recognizer'):
                    state_dict = torch.load(
                        self._model_path,
                        map_location='cuda' if self._gpu else 'cpu',
                    )
                    self._reader.recognizer.load_state_dict(state_dict)
                    self._reader.recognizer.eval()
                    self.finetuned_loaded = True
                    print("[OCREngine] Fine-tuned recognizer loaded.")
                else:
                    print("[OCREngine] Could not access recognizer; using default model")
            except Exception as e:
                print(f"[OCREngine] Failed to load fine-tuned model ({e}); using default model")

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

        # Post-OCR horizontal merge: join fragmented tokens on the same row
        # (e.g. EasyOCR splitting "Beef Teriyaki Ramen" into 3 boxes).
        lines = self._merge_horizontal(lines)

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

    def _merge_horizontal(self, lines: list[dict]) -> list[dict]:
        """Merge horizontally adjacent boxes on the same row into one logical line.

        Conservative strategy: only merges boxes whose horizontal gap is < avg_height * 0.6
        (≈ 1 character space). Column gaps on receipts are typically 5–15× height so they
        won't be merged.
        """
        if not lines:
            return lines

        avg_height = float(np.mean([l['height'] for l in lines]))
        y_thresh = avg_height * 0.6
        gap_thresh = avg_height * 0.6

        # Group into vertical rows
        rows: list[list[dict]] = []
        for ln in lines:
            placed = False
            for row in rows:
                if abs(ln['y_center'] - row[0]['y_center']) < y_thresh:
                    row.append(ln)
                    placed = True
                    break
            if not placed:
                rows.append([ln])

        merged_output: list[dict] = []
        for row in rows:
            row.sort(key=lambda l: l['x_min'])
            cluster = [row[0]]
            for nxt in row[1:]:
                if nxt['x_min'] - cluster[-1]['x_max'] <= gap_thresh:
                    cluster.append(nxt)
                else:
                    merged_output.append(self._combine_cluster(cluster, avg_height))
                    cluster = [nxt]
            merged_output.append(self._combine_cluster(cluster, avg_height))

        merged_output.sort(key=lambda l: (round(l['y_center'] / max(avg_height, 1e-6)), l['x_min']))
        return merged_output

    @staticmethod
    def _is_punct_or_short_digit(text: str) -> bool:
        """True if fragment is punctuation or a short digit (e.g. ',' '.' '000')."""
        t = text.strip()
        if not t:
            return True
        if all(c in '.,;:- ' for c in t):
            return True
        if len(t) <= 4 and all(c.isdigit() or c in '.,' for c in t):
            return True
        return False

    def _combine_cluster(self, cluster: list[dict], avg_height: float) -> dict:
        """Combine boxes in the same cluster into one line dict."""
        if len(cluster) == 1:
            return cluster[0]

        parts = [cluster[0]['text']]
        for i in range(1, len(cluster)):
            gap = cluster[i]['x_min'] - cluster[i - 1]['x_max']
            close_gap = gap < avg_height * 0.2
            punct_glue = (
                OCREngine._is_punct_or_short_digit(cluster[i - 1]['text'])
                or OCREngine._is_punct_or_short_digit(cluster[i]['text'])
            )
            sep = '' if (close_gap or punct_glue) else ' '
            parts.append(sep + cluster[i]['text'])

        combined = ''.join(parts).strip()
        combined = combined.replace(' ,', ',').replace(' .', '.').replace(' :', ':')

        all_xs = [c['x_min'] for c in cluster] + [c['x_max'] for c in cluster]
        all_ys = [c['y_min'] for c in cluster] + [c['y_max'] for c in cluster]
        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys), max(all_ys)

        return {
            'text': combined,
            'bbox': [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            'confidence': min(c['confidence'] for c in cluster),
            'y_center': (y_min + y_max) / 2,
            'x_center': (x_min + x_max) / 2,
            'width': x_max - x_min,
            'height': y_max - y_min,
            'x_min': x_min, 'y_min': y_min,
            'x_max': x_max, 'y_max': y_max,
        }

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
