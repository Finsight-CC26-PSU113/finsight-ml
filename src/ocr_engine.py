"""
OCR FinSight - OCR Engine
EasyOCR wrapper yang mengembalikan baris teks terstruktur dari gambar struk.
"""

import sys
import easyocr
import numpy as np
from pathlib import Path
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
    
    def __init__(self, languages: list[str] = None, gpu: bool = None, model_path: str = None):
        """Initialize EasyOCR reader.
        
        Args:
            languages: List kode bahasa (default dari config).
            gpu: Gunakan GPU (default: auto-detect, fall back to CPU).
            model_path: Path to fine-tuned model weights (optional).
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
        
        # Check for fine-tuned model
        self._model_path = model_path
        if model_path:
            from pathlib import Path
            if Path(model_path).exists():
                print(f"[OCREngine] Loading fine-tuned EasyOCR model from {model_path}...")
            else:
                print(f"[OCREngine] Fine-tuned model not found at {model_path}, using default model")
                self._model_path = None
        
        print(f"[OCREngine] Loading EasyOCR model (languages={self._languages}, gpu={self._gpu})...")
        self._reader = easyocr.Reader(
            self._languages,
            gpu=self._gpu,
            verbose=False,
            model_storage_directory=None,  # Use default
            download_enabled=True
        )
        
        # Load fine-tuned weights if available
        if self._model_path:
            try:
                import torch
                print(f"[OCREngine] Loading fine-tuned weights...")
                
                # Build model sama seperti saat training
                easyocr_dir = Path(easyocr.__file__).parent
                sys.path.insert(0, str(easyocr_dir))
                from model.vgg_model import Model as EasyOCRModel
                
                # Character set (must match training)
                import string
                CHARSET = (
                    string.ascii_letters +
                    string.digits +
                    string.punctuation +
                    " "
                )
                CHARSET += "RM"
                CHARSET = "".join(sorted(set(CHARSET)))
                NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank
                
                print(f"[OCREngine]    Building model architecture (num_classes={NUM_CLASSES})...")
                finetuned_model = EasyOCRModel(
                    input_channel=1,
                    output_channel=256,
                    hidden_size=256,
                    num_class=NUM_CLASSES
                )
                
                # Load trained weights
                state_dict = torch.load(self._model_path, map_location='cuda' if self._gpu else 'cpu', 
                                      weights_only=False)
                
                # Remove module. prefix if exists
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                # Load weights
                missing, unexpected = finetuned_model.load_state_dict(state_dict, strict=True)
                
                if not missing and not unexpected:
                    print(f"[OCREngine] ✅ Fine-tuned weights loaded perfectly ({len(state_dict)} params)")
                else:
                    print(f"[OCREngine] ⚠️  Weights loaded with issues:")
                    if missing:
                        print(f"[OCREngine]    Missing: {len(missing)} keys")
                    if unexpected:
                        print(f"[OCREngine]    Unexpected: {len(unexpected)} keys")
                
                # Replace recognizer in reader
                device = torch.device('cuda' if self._gpu else 'cpu')
                finetuned_model = finetuned_model.to(device)
                finetuned_model.eval()
                
                self._reader.recognizer = finetuned_model
                self._reader.character = list(CHARSET)  # Update character list
                
                print(f"[OCREngine] ✅ Fine-tuned recognizer replaced successfully")
                
            except ImportError as e:
                print(f"[OCREngine] Cannot import EasyOCR model architecture: {e}")
                print("[OCREngine] Using default EasyOCR model")
            except Exception as e:
                print(f"[OCREngine] Failed to load fine-tuned model: {e}")
                import traceback
                traceback.print_exc()
                print("[OCREngine] Using default EasyOCR model")
        else:
            print("[OCREngine] Using default EasyOCR model")
    
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
        # EasyOCR detector params: gunakan default supaya tidak merusak character recognition.
        # Kustomisasi ada di post-OCR _merge_horizontal yang bekerja di level layout, bukan karakter.
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

        # Post-OCR row merging: gabungkan box di baris-y yang sama jadi satu logical line.
        # Ini mengatasi kasus EasyOCR mecah "Beef Teriyaki Ramen" jadi 3 box terpisah,
        # atau "23,415" + "257,565" yang sebenarnya 1 baris harga.
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

    def _merge_horizontal(self, lines: list[dict]) -> list[dict]:
        """Gabungkan box yang ada di baris-y yang sama dan **berdekatan secara horizontal**
        menjadi satu logical line.

        Strategi konservatif (hindari merge antar-kolom):
        - 2 box dianggap 1 baris vertikal jika selisih y_center < avg_height * 0.6
        - Dalam baris yang sama, 2 box di-merge HANYA bila gap horizontal < avg_height * 0.6
          (≈ 1 spasi). Gap antar-kolom struk biasanya 5-15× height, jadi aman tidak ke-merge.
        - Concat tanpa spasi bila gap sangat kecil (< 0.2 * height) ATAU salah satu fragment
          adalah pure tanda baca/digit pendek (handle "35 ," + "000" → "35,000").
        """
        if not lines:
            return lines

        avg_height = float(np.mean([l['height'] for l in lines]))
        y_thresh = avg_height * 0.6
        # Threshold konservatif: hanya merge box yang benar-benar berdekatan horizontal.
        gap_thresh = avg_height * 0.6

        # Group ke dalam baris vertikal terlebih dahulu (lines sudah ter-sort by y_center)
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

        # Re-sort hasil akhir top-to-bottom, left-to-right
        merged_output.sort(key=lambda l: (round(l['y_center'] / max(avg_height, 1e-6)), l['x_min']))
        return merged_output

    @staticmethod
    def _is_punct_or_short_digit(text: str) -> bool:
        """True jika fragment adalah tanda baca atau digit pendek (mis. "," "." "00" "000")."""
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
            # Concat tanpa spasi jika:
            # - gap sangat kecil (< 0.2 * height) — kata yang ke-pecah saat OCR
            # - ATAU salah satu fragment hanya tanda baca / digit pendek (e.g. "35," + "000")
            close_gap = gap < avg_height * 0.2
            punct_glue = (
                OCREngine._is_punct_or_short_digit(cluster[i - 1]['text'])
                or OCREngine._is_punct_or_short_digit(cluster[i]['text'])
            )
            sep = '' if (close_gap or punct_glue) else ' '
            parts.append(sep + cluster[i]['text'])
        combined_text = ''.join(parts).strip()
        # Bersihkan spasi sebelum tanda baca
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
