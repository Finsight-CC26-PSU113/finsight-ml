"""
OCR FinSight — OCR Service (PaddleOCR)
Singleton wrapper returning structured text lines from a receipt image.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from app.core.config import settings
from app.services.text_post_processor import get_post_processor


class OCREngine:
    """Singleton PaddleOCR wrapper with structured output."""

    _instance: Optional["OCREngine"] = None
    _reader = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, lang: str = None, gpu: bool = None):
        if self._reader is not None:
            return

        from paddleocr import PaddleOCR

        self._lang = lang or settings.OCR_LANG
        self._gpu = gpu if gpu is not None else settings.OCR_GPU

        if self._gpu:
            try:
                import paddle

                if not paddle.is_compiled_with_cuda():
                    print("[OCREngine] GPU requested but CUDA not available, using CPU")
                    self._gpu = False
            except Exception:
                print("[OCREngine] Cannot detect GPU, using CPU")
                self._gpu = False

        print(f"[OCREngine] Loading PaddleOCR (lang={self._lang}, gpu={self._gpu})...")
        try:
            self._reader = PaddleOCR(
                use_angle_cls=True,
                lang=self._lang,
                use_gpu=self._gpu,
                show_log=False,
            )
        except ValueError:
            self._reader = PaddleOCR(
                use_angle_cls=True, lang=self._lang, use_gpu=self._gpu
            )
        print("[OCREngine] Ready.")

    # ── Public API ──────────────────────────────────────────────────────────

    def read_receipt(
        self, image: np.ndarray, apply_post_processing: bool = True
    ) -> list[dict]:
        """Read text from a receipt image.

        Returns sorted list of dicts with keys:
        text, bbox, confidence, y/x_center, width, height, x_min, y_min, x_max, y_max, line_index.
        """
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 3:
            import cv2

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        try:
            raw_results = self._reader.ocr(image_rgb, cls=True)
        except TypeError:
            raw_results = self._reader.ocr(image_rgb)

        if not raw_results or not raw_results[0]:
            return []

        lines = []
        for item in raw_results[0]:
            bbox = item[0]
            text, confidence = item[1]
            arr = np.array(bbox)
            x_min, x_max = float(arr[:, 0].min()), float(arr[:, 0].max())
            y_min, y_max = float(arr[:, 1].min()), float(arr[:, 1].max())
            lines.append(
                {
                    "text": text.strip(),
                    "bbox": bbox,
                    "confidence": float(confidence),
                    "y_center": (y_min + y_max) / 2,
                    "x_center": (x_min + x_max) / 2,
                    "width": x_max - x_min,
                    "height": y_max - y_min,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                }
            )

        lines = self._sort_lines(lines)
        lines = self._merge_horizontal(lines)
        for i, line in enumerate(lines):
            line["line_index"] = i

        if apply_post_processing:
            lines = get_post_processor().process_lines(lines)

        return lines

    def read_receipt_with_image_info(
        self, image: np.ndarray
    ) -> tuple[list[dict], dict]:
        """Read text and normalize bbox coordinates to 0-1."""
        lines = self.read_receipt(image)
        h, w = image.shape[:2]
        for line in lines:
            line["x_min"] /= w
            line["x_max"] /= w
            line["y_min"] /= h
            line["y_max"] /= h
            line["x_center"] /= w
            line["y_center"] /= h
            line["width"] /= w
            line["height"] /= h
        return lines, {"height": h, "width": w, "total_lines": len(lines)}

    # ── Internal helpers ────────────────────────────────────────────────────

    def _sort_lines(self, lines: list[dict]) -> list[dict]:
        if not lines:
            return lines
        avg_h = np.mean([l["height"] for l in lines])
        threshold = avg_h * 0.6
        lines.sort(key=lambda l: l["y_center"])

        grouped, current = [], [lines[0]]
        for line in lines[1:]:
            if abs(line["y_center"] - current[0]["y_center"]) < threshold:
                current.append(line)
            else:
                current.sort(key=lambda l: l["x_center"])
                grouped.extend(current)
                current = [line]
        current.sort(key=lambda l: l["x_center"])
        grouped.extend(current)
        return grouped

    def _merge_horizontal(self, lines: list[dict]) -> list[dict]:
        if not lines:
            return lines
        avg_h = float(np.mean([l["height"] for l in lines]))
        y_thresh = avg_h * 0.6
        gap_thresh = avg_h * 0.6

        rows: list[list[dict]] = []
        for ln in lines:
            placed = False
            for row in rows:
                if abs(ln["y_center"] - row[0]["y_center"]) < y_thresh:
                    row.append(ln)
                    placed = True
                    break
            if not placed:
                rows.append([ln])

        merged: list[dict] = []
        for row in rows:
            row.sort(key=lambda l: l["x_min"])
            cluster = [row[0]]
            for nxt in row[1:]:
                if nxt["x_min"] - cluster[-1]["x_max"] <= gap_thresh:
                    cluster.append(nxt)
                else:
                    merged.append(self._combine_cluster(cluster, avg_h))
                    cluster = [nxt]
            merged.append(self._combine_cluster(cluster, avg_h))

        merged.sort(key=lambda l: (round(l["y_center"] / max(avg_h, 1e-6)), l["x_min"]))
        return merged

    @staticmethod
    def _is_punct_or_short_digit(text: str) -> bool:
        t = text.strip()
        if not t:
            return True
        if all(c in ".,;:- " for c in t):
            return True
        if len(t) <= 4 and all(c.isdigit() or c in ".," for c in t):
            return True
        return False

    def _combine_cluster(self, cluster: list[dict], avg_h: float) -> dict:
        if len(cluster) == 1:
            return cluster[0]

        parts = [cluster[0]["text"]]
        for i in range(1, len(cluster)):
            gap = cluster[i]["x_min"] - cluster[i - 1]["x_max"]
            close_gap = gap < avg_h * 0.2
            punct_glue = self._is_punct_or_short_digit(
                cluster[i - 1]["text"]
            ) or self._is_punct_or_short_digit(cluster[i]["text"])
            sep = "" if (close_gap or punct_glue) else " "
            parts.append(sep + cluster[i]["text"])

        combined = "".join(parts).strip()
        combined = combined.replace(" ,", ",").replace(" .", ".").replace(" :", ":")

        all_xs = [c["x_min"] for c in cluster] + [c["x_max"] for c in cluster]
        all_ys = [c["y_min"] for c in cluster] + [c["y_max"] for c in cluster]
        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys), max(all_ys)

        return {
            "text": combined,
            "bbox": [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            "confidence": min(c["confidence"] for c in cluster),
            "y_center": (y_min + y_max) / 2,
            "x_center": (x_min + x_max) / 2,
            "width": x_max - x_min,
            "height": y_max - y_min,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }
