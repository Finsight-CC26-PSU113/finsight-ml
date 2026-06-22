"""
OCR FinSight - PaddleOCR Engine
================================
PaddleOCR wrapper dengan interface yang sama seperti ocr_engine.py
untuk easy migration dari EasyOCR ke PaddleOCR.

PaddleOCR advantages:
- 2-3x faster than EasyOCR
- Better accuracy on small text and receipts
- Better multilingual support (80+ languages)
- Actively maintained by Baidu

Installation:
    pip install paddlepaddle-gpu  # For GPU
    # OR
    pip install paddlepaddle  # For CPU only
    
    pip install paddleocr

Usage:
    from src.ocr_engine_paddle import PaddleOCREngine
    engine = PaddleOCREngine()
    lines = engine.read_receipt(image)
"""

import numpy as np
from typing import Optional
from paddleocr import PaddleOCR
from src.config import OCR_GPU
from src.text_post_processor import get_post_processor


class PaddleOCREngine:
    """Wrapper PaddleOCR dengan output terstruktur yang compatible dengan OCREngine.
    
    Menggunakan singleton pattern agar model PaddleOCR hanya
    di-load sekali ke memori GPU.
    
    Usage:
        engine = PaddleOCREngine()
        lines = engine.read_receipt(image)
        # lines = [
        #     {"text": "INDOMARET", "bbox": [...], "confidence": 0.95,
        #      "y_center": 120, "x_center": 300, "width": 200, "height": 25,
        #      "line_index": 0},
        #     ...
        # ]
    """
    
    _instance: Optional['PaddleOCREngine'] = None
    _reader: Optional[PaddleOCR] = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton: satu instance PaddleOCREngine untuk satu runtime."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, lang: str = 'latin', gpu: bool = None):
        """Initialize PaddleOCR reader.
        
        Args:
            lang: Language code ('latin', 'en', 'ch', etc.)
                  For Indonesian receipts, use 'latin' (covers English + Indonesian)
            gpu: Use GPU (default: auto-detect from config)
        """
        if self._reader is not None:
            return  # Sudah diinisialisasi
            
        self._lang = lang
        
        # Auto-detect GPU
        if gpu is None:
            self._gpu = OCR_GPU
            if self._gpu:
                try:
                    import paddle
                    if not paddle.is_compiled_with_cuda():
                        print("[PaddleOCR] GPU requested but CUDA not available, using CPU")
                        self._gpu = False
                except Exception:
                    print("[PaddleOCR] Cannot detect GPU, using CPU")
                    self._gpu = False
        else:
            self._gpu = gpu
        
        print(f"[PaddleOCR] Loading PaddleOCR (lang={self._lang}, gpu={self._gpu})...")
        
        try:
            # Initialize PaddleOCR with GPU support
            self._reader = PaddleOCR(
                use_angle_cls=True,  # Text orientation detection
                lang=self._lang,     # Language model
                use_gpu=self._gpu,   # Enable GPU if available
                gpu_mem=500,         # GPU memory limit in MB
                show_log=True,       # Show detailed logs
            )
            print(f"[PaddleOCR] Ready ✓")
        except ValueError as e:
            # Parameter not supported in this version
            error_msg = str(e)
            if "Unknown argument" in error_msg:
                print(f"[PaddleOCR] Parameter incompatibility, trying without gpu_mem...")
                try:
                    # Try without gpu_mem parameter
                    self._reader = PaddleOCR(
                        use_angle_cls=True,
                        lang=self._lang,
                        use_gpu=self._gpu,
                    )
                    print(f"[PaddleOCR] Ready ✓")
                except Exception as e2:
                    print(f"[PaddleOCR] Trying minimal mode...")
                    # Absolute minimal - just lang
                    self._reader = PaddleOCR(lang=self._lang)
                    print(f"[PaddleOCR] Ready ✓ (minimal mode, GPU not configured)")
            else:
                raise
        except Exception as e:
            error_msg = str(e)
            if "paddlepaddle" in error_msg.lower() or "paddle_static" in error_msg.lower():
                print(f"[PaddleOCR] ❌ PaddlePaddle core not installed")
                print(f"[PaddleOCR]    Install with: pip install paddlepaddle")
                raise ImportError("PaddlePaddle not installed") from e
            else:
                print(f"[PaddleOCR] ❌ Initialization failed: {e}")
                raise
        
        print(f"[PaddleOCR] Ready ✓")
    
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
        # PaddleOCR expects RGB, convert if needed
        if len(image.shape) == 2:
            # Grayscale to RGB
            image_rgb = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 3:
            # Assume BGR, convert to RGB
            import cv2
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Run OCR
        # result = [
        #   [bbox, (text, confidence)],
        #   ...
        # ]
        try:
            # Try with cls parameter (text classification/orientation)
            raw_results = self._reader.ocr(image_rgb, cls=True)
        except TypeError:
            # cls parameter not supported in this version, use default
            raw_results = self._reader.ocr(image_rgb)
        
        if not raw_results or not raw_results[0]:
            return []
        
        # Parse raw results ke format terstruktur
        lines = []
        for item in raw_results[0]:
            bbox = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text, confidence = item[1]
            
            # Convert bbox to numpy for easier computation
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
        
        # Post-OCR row merging: gabungkan box di baris-y yang sama jadi satu logical line
        lines = self._merge_horizontal(lines)
        
        # Tambahkan line_index
        for i, line in enumerate(lines):
            line['line_index'] = i
        
        # Apply post-processing to fix common OCR errors
        if apply_post_processing:
            post_processor = get_post_processor()
            lines = post_processor.process_lines(lines)
        
        return lines
    
    def _sort_lines(self, lines: list[dict]) -> list[dict]:
        """Sort baris dari atas ke bawah, lalu kiri ke kanan."""
        if not lines:
            return lines
        
        avg_height = np.mean([l['height'] for l in lines])
        y_threshold = avg_height * 0.6
        
        lines.sort(key=lambda l: l['y_center'])
        
        grouped = []
        current_group = [lines[0]]
        
        for i in range(1, len(lines)):
            if abs(lines[i]['y_center'] - current_group[0]['y_center']) < y_threshold:
                current_group.append(lines[i])
            else:
                current_group.sort(key=lambda l: l['x_center'])
                grouped.extend(current_group)
                current_group = [lines[i]]
        
        current_group.sort(key=lambda l: l['x_center'])
        grouped.extend(current_group)
        
        return grouped
    
    def _merge_horizontal(self, lines: list[dict]) -> list[dict]:
        """Gabungkan box yang ada di baris-y yang sama dan berdekatan secara horizontal."""
        if not lines:
            return lines

        avg_height = float(np.mean([l['height'] for l in lines]))
        y_thresh = avg_height * 0.6
        gap_thresh = avg_height * 0.6

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
            cluster: list[dict] = [row[0]]
            for nxt in row[1:]:
                last = cluster[-1]
                horiz_gap = nxt['x_min'] - last['x_max']
                if horiz_gap <= gap_thresh:
                    cluster.append(nxt)
                else:
                    merged_output.append(self._combine_cluster(cluster, avg_height))
                    cluster = [nxt]
            merged_output.append(self._combine_cluster(cluster, avg_height))

        merged_output.sort(key=lambda l: (round(l['y_center'] / max(avg_height, 1e-6)), l['x_min']))
        return merged_output

    @staticmethod
    def _is_punct_or_short_digit(text: str) -> bool:
        """True jika fragment adalah tanda baca atau digit pendek."""
        t = text.strip()
        if not t:
            return True
        if all(c in '.,;:- ' for c in t):
            return True
        if len(t) <= 4 and all(c.isdigit() or c in '.,' for c in t):
            return True
        return False

    def _combine_cluster(self, cluster: list[dict], avg_height: float) -> dict:
        """Gabungkan beberapa box yang sudah dipastikan satu logical text fragment."""
        if len(cluster) == 1:
            return cluster[0]

        parts = [cluster[0]['text']]
        for i in range(1, len(cluster)):
            gap = cluster[i]['x_min'] - cluster[i - 1]['x_max']
            close_gap = gap < avg_height * 0.2
            punct_glue = (
                PaddleOCREngine._is_punct_or_short_digit(cluster[i - 1]['text'])
                or PaddleOCREngine._is_punct_or_short_digit(cluster[i]['text'])
            )
            sep = '' if (close_gap or punct_glue) else ' '
            parts.append(sep + cluster[i]['text'])
        combined_text = ''.join(parts).strip()
        combined_text = combined_text.replace(' ,', ',').replace(' .', '.').replace(' :', ':')

        all_xs = [c['x_min'] for c in cluster] + [c['x_max'] for c in cluster]
        all_ys = [c['y_min'] for c in cluster] + [c['y_max'] for c in cluster]
        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys), max(all_ys)
        merged_bbox = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]
        merged_conf = min(c['confidence'] for c in cluster)

        return {
            'text': combined_text,
            'bbox': merged_bbox,
            'confidence': merged_conf,
            'y_center': (y_min + y_max) / 2,
            'x_center': (x_min + x_max) / 2,
            'width': x_max - x_min,
            'height': y_max - y_min,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
        }
    
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
