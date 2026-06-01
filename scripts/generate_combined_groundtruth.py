"""
Combined Ground Truth Generator (OCR + Classification) using Gemini Vision
===========================================================================
Satu API call menghasilkan dua output:
1. Classification ground truth → CSV untuk training BiLSTM classifier
2. OCR ground truth → labels.txt + crops untuk fine-tune EasyOCR

Pipeline:
    Image → EasyOCR (bbox+text) → Gemini Vision → {corrected_text, label}
    → save: classification CSV row + crop image with corrected text

Categories (7):
    STORE, ADDRESS_CONTACT, DATE, ITEM_DESC, ITEM_PRICE/QTY, TOTAL_PAYMENT, OTHER

Usage:
    python scripts/generate_combined_groundtruth.py --api-key YOUR_KEY
    python scripts/generate_combined_groundtruth.py --api-key YOUR_KEY --max-images 100 --resume
"""

import os
import sys
import json
import time
import argparse
import re
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ROOT_DIR
from src.preprocessing import load_image, deskew
from src.ocr_engine import OCREngine

try:
    import google.generativeai as genai
except ImportError:
    print("❌ Install: pip install google-generativeai")
    sys.exit(1)


# ============================================================
# Config
# ============================================================
DATASET_DIRS = [
    # (path, country_tag)
    (ROOT_DIR / "fullDataset" / "images", "MY"),
    (ROOT_DIR / "Dataset Baru" / "Data Tahap Pertama - 26 April 26" / "Data Bersih Final", "ID1"),
    (ROOT_DIR / "Dataset Baru" / "Data Tahap Kedua-29 April 2026" / "Data Bersih", "ID2"),
]

# Output paths
OUTPUT_DIR = ROOT_DIR / "data" / "combined_groundtruth"

# Classification CSV
CLASS_CSV = OUTPUT_DIR / "classification_labels.csv"

# OCR ground truth
OCR_DIR = OUTPUT_DIR / "ocr"
OCR_CROPS_DIR = OCR_DIR / "images"
OCR_LABELS = OCR_DIR / "labels.txt"

# Progress
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# API rate limiting
DELAY_BETWEEN_REQUESTS = 4.5  # seconds (free tier ~15 req/min)

# Crop settings (for EasyOCR fine-tuning)
CROP_HEIGHT = 32
CROP_PAD_X = 4
CROP_PAD_Y = 2

VALID_LABELS = {
    'STORE', 'ADDRESS_CONTACT', 'DATE',
    'ITEM_DESC', 'ITEM_PRICE/QTY', 
    'SUBTOTAL', 'TAX', 'DISCOUNT', 'SERVICE_CHARGE', 'GRAND_TOTAL',
    'CASH_PAYMENT', 'OTHER'
}

CSV_FIELDS = [
    'source', 'filename', 'line_index', 'text', 'label',
    'ocr_text', 'ocr_confidence',
    'x_min', 'y_min', 'x_max', 'y_max',
    'y_center', 'x_center', 'width', 'height',
]


# ============================================================
# Combined Gemini Prompt
# ============================================================

COMBINED_PROMPT = """You are an expert receipt analyzer with two tasks for EACH line:

1. CORRECT the text (fix OCR errors by looking at the image)
2. CLASSIFY the line into ONE of 12 categories

I will provide you:
- A receipt image
- EasyOCR-extracted lines (with line numbers and possibly errors)

CATEGORIES (12 categories for better accuracy):
- STORE: Store/business name (e.g., INDOMARET, WARUNG SARI, AEON)
- ADDRESS_CONTACT: Address, phone, email, website, NPWP, GST/ROC number
- DATE: Date AND/OR time in ANY format (e.g., 26-11-17, Mar 14 2025, 24/05/2026, 10:30:45, Tanggal 24 Mei 2026, Jam 14:30)
- ITEM_DESC: Product/item name (e.g., Nasi Goreng, Indomie Goreng, Burger)
- ITEM_PRICE/QTY: Item price OR quantity (e.g., Rp 25.000, 2 x 5000, @9.80, RM 8.50)
- SUBTOTAL: Subtotal before tax/service charge (Sub Total, Jumlah, Amount)
- TAX: Tax, GST, VAT, PPN, SST (e.g., "GST 6%", "Tax RM 1.50", "PPN Rp 5.500")
- DISCOUNT: Discounts, vouchers, promotions (e.g., "Member Discount", "Diskon Rp 10.000")
- SERVICE_CHARGE: Service charges, tips (e.g., "Service Charge 10%", "Biaya Layanan")
- GRAND_TOTAL: Final total amount after all adjustments (Total, Grand Total, Total Bayar, Total Amount)
- CASH_PAYMENT: Cash paid, change given (Cash, Tunai, Change, Kembali, Kembalian)
- OTHER: Anything else (cashier name, receipt number, transaction ID, table number, footer text, greetings)

IMPORTANT DISTINCTIONS:
- SUBTOTAL: Amount BEFORE tax/service charge (usually labeled "Sub Total", "Subtotal", "Jumlah")
- TAX: Tax amount or percentage (GST, VAT, PPN, SST, Tax)
- DISCOUNT: Any discount or promotion amount
- SERVICE_CHARGE: Service fee or tip
- GRAND_TOTAL: Final amount to pay AFTER all adjustments (usually labeled "Total", "Grand Total", "Total Amount")
- CASH_PAYMENT: Payment-related (cash given, change returned)

OCR ERROR FIXES (common):
- O ↔ 0 (letter O vs digit 0)
- I ↔ 1 ↔ l
- S ↔ 5
- B ↔ 8
- Wrong special characters
- Typos in store names

OUTPUT FORMAT:
Return ONLY a valid JSON array. EXACTLY same length as input lines.
Each item is an object with three keys: line_num (int), text (string), label (string).

EXAMPLE:
[
  {"line_num": 0, "text": "INDOMARET", "label": "STORE"},
  {"line_num": 1, "text": "Jl. Sudirman No. 123", "label": "ADDRESS_CONTACT"},
  {"line_num": 2, "text": "14 Mar 2025 10:30", "label": "DATE"},
  {"line_num": 3, "text": "Indomie Goreng", "label": "ITEM_DESC"},
  {"line_num": 4, "text": "Rp 3.500", "label": "ITEM_PRICE/QTY"},
  {"line_num": 5, "text": "Sub Total", "label": "SUBTOTAL"},
  {"line_num": 6, "text": "Rp 45.000", "label": "SUBTOTAL"},
  {"line_num": 7, "text": "PPN 11%", "label": "TAX"},
  {"line_num": 8, "text": "Rp 4.950", "label": "TAX"},
  {"line_num": 9, "text": "Member Discount", "label": "DISCOUNT"},
  {"line_num": 10, "text": "Rp 5.000", "label": "DISCOUNT"},
  {"line_num": 11, "text": "Total Bayar", "label": "GRAND_TOTAL"},
  {"line_num": 12, "text": "Rp 44.950", "label": "GRAND_TOTAL"}
]

CRITICAL RULES:
1. Output array MUST have EXACTLY same number of items as input
2. Keep line order matching input
3. Use ONLY the 12 categories listed (exact spelling, case-sensitive)
4. If a line is unreadable noise, keep its original text and label as OTHER
5. Do NOT merge or split lines
6. In the "text" field, ESCAPE any double-quotes as \\" so JSON stays valid
7. Do NOT include line breaks INSIDE a string value
8. Replace any tab characters with a single space
9. Return ONLY the JSON array, no markdown, no explanation, no surrounding text
10. For lines with just numbers after a label, use the SAME category as the label (e.g., if "Sub Total" is SUBTOTAL, the next line with amount should also be SUBTOTAL)

Here are the lines from EasyOCR (in reading order):
"""


# ============================================================
# Gemini
# ============================================================

def init_gemini(api_key: str, model_name: str = 'gemini-3.1-flash-lite'):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    print(f"✅ Gemini Vision initialized ({model_name})")
    return model


class GeminiKeyRotator:
    """Manage multiple Gemini API keys with auto-rotation on quota errors."""
    
    def __init__(self, api_keys: list[str], model_name: str = 'gemini-3.1-flash-lite'):
        self.api_keys = [k.strip() for k in api_keys if k.strip()]
        self.model_name = model_name
        self.current_idx = 0
        self.dead_keys = set()  # keys that hit quota
        if not self.api_keys:
            raise ValueError("At least one API key required")
        self._init_current()
    
    def _init_current(self):
        """Initialize Gemini with current key."""
        key = self.api_keys[self.current_idx]
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(self.model_name)
        masked = key[:10] + "..." + key[-4:]
        print(f"🔑 Active API key: #{self.current_idx + 1}/{len(self.api_keys)} ({masked})")
    
    def rotate(self) -> bool:
        """Rotate to next available key. Returns False if no keys left."""
        self.dead_keys.add(self.current_idx)
        for offset in range(1, len(self.api_keys) + 1):
            next_idx = (self.current_idx + offset) % len(self.api_keys)
            if next_idx not in self.dead_keys:
                self.current_idx = next_idx
                self._init_current()
                return True
        print("❌ All API keys exhausted!")
        return False
    
    def is_quota_error(self, exception) -> bool:
        """Detect quota/rate-limit related errors."""
        msg = str(exception).lower()
        return any(kw in msg for kw in [
            'quota', 'rate limit', 'rate_limit', 'resource_exhausted',
            'permission_denied', '429', '403'
        ])


def _extract_objects_from_text(response_text: str) -> list[dict]:
    """Fallback parser: extract objects one-by-one when full JSON parse fails.
    
    Tries to find {"line_num": N, "text": "...", "label": "..."} patterns.
    """
    items = []
    # Match each object greedily (handle escaped quotes inside text)
    pattern = re.compile(
        r'\{\s*"line_num"\s*:\s*(\d+)\s*,\s*'
        r'"text"\s*:\s*"((?:\\.|[^"\\])*)"\s*,\s*'
        r'"label"\s*:\s*"([A-Z_/]+)"\s*\}',
        re.DOTALL
    )
    for match in pattern.finditer(response_text):
        try:
            line_num = int(match.group(1))
            # Unescape JSON string
            text = match.group(2).encode().decode('unicode_escape')
            label = match.group(3)
            items.append({'line_num': line_num, 'text': text, 'label': label})
        except Exception:
            continue
    return items


def call_gemini_combined(rotator: 'GeminiKeyRotator', image_path: Path, lines: list[dict]) -> list[dict] | None:
    """Send image + lines, get [{line_num, text, label}, ...] for each line.
    Auto-rotates API keys on quota errors.
    
    Returns:
        List of dicts {line_num, text, label}, or None if failed
    """
    try:
        ocr_context = "\n".join(
            f'{i}: "{line.get("text", "")}"' for i, line in enumerate(lines)
        )
        img = Image.open(image_path)
        full_prompt = COMBINED_PROMPT + ocr_context
        
        # Try current key, rotate on quota error
        max_attempts = len(rotator.api_keys)
        last_exception = None
        response_text = None
        for attempt in range(max_attempts):
            try:
                response = rotator.model.generate_content([full_prompt, img])
                response_text = response.text.strip()
                break
            except Exception as e:
                last_exception = e
                if rotator.is_quota_error(e):
                    print(f"\n  🔄 Key quota hit, rotating...")
                    if not rotator.rotate():
                        return None
                else:
                    raise
        if response_text is None:
            print(f"  ⚠️  All keys exhausted: {last_exception}")
            return None
        
        # Strip markdown code fences if any
        if response_text.startswith("```"):
            response_text = re.sub(r'^```\w*\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)
        
        # Try strict JSON parse first
        result = None
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: extract objects via regex
            print(f"  🔧 JSON malformed, using fallback parser...", end=" ")
            result = _extract_objects_from_text(response_text)
            if not result:
                print(f"failed")
                return None
            print(f"recovered {len(result)} items")
        
        if not isinstance(result, list):
            return None
        
        # Build aligned list (1:1 with input lines)
        aligned = [None] * len(lines)
        for item in result:
            if not isinstance(item, dict):
                continue
            idx = item.get('line_num', -1)
            if not (0 <= idx < len(lines)):
                continue
            
            text = str(item.get('text', '')).strip()
            label = str(item.get('label', 'OTHER')).strip()
            if label not in VALID_LABELS:
                label = 'OTHER'
            
            aligned[idx] = {
                'line_num': idx,
                'text': text or lines[idx].get('text', '').strip(),
                'label': label,
            }
        
        # Fill any missing entries with OCR fallback
        for i in range(len(lines)):
            if aligned[i] is None:
                aligned[i] = {
                    'line_num': i,
                    'text': lines[i].get('text', '').strip(),
                    'label': 'OTHER',
                }
        
        return aligned
    
    except Exception as e:
        print(f"  ⚠️  Gemini error: {type(e).__name__}: {e}")
        return None


# ============================================================
# OCR + Crop helpers
# ============================================================

def get_ocr_lines(image_path: Path, ocr_engine: OCREngine):
    """Run EasyOCR on image."""
    try:
        img = load_image(image_path)
        img = deskew(img)
        lines, img_info = ocr_engine.read_receipt_with_image_info(img)
        return img, lines, img_info
    except Exception as e:
        print(f"  ⚠️  OCR error: {e}")
        return None, None, None


def crop_line(img: np.ndarray, line: dict, img_h: int, img_w: int):
    """Crop one line from image, return grayscale resized to height=32."""
    x_min = int(line['x_min'] * img_w) - CROP_PAD_X
    y_min = int(line['y_min'] * img_h) - CROP_PAD_Y
    x_max = int(line['x_max'] * img_w) + CROP_PAD_X
    y_max = int(line['y_max'] * img_h) + CROP_PAD_Y
    
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)
    
    if x_max <= x_min or y_max <= y_min:
        return None
    if (x_max - x_min) < 5 or (y_max - y_min) < 5:
        return None
    
    if len(img.shape) == 3:
        crop = img[y_min:y_max, x_min:x_max]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop = img[y_min:y_max, x_min:x_max]
    
    h, w = crop.shape
    new_w = max(int(w * CROP_HEIGHT / h), 1)
    return cv2.resize(crop, (new_w, CROP_HEIGHT), interpolation=cv2.INTER_CUBIC)


def text_alignment_ok(ocr_text: str, gt_text: str) -> bool:
    """Check if Gemini's text is reasonably aligned with EasyOCR's.
    Used to filter misaligned crops for OCR fine-tuning.
    """
    ocr_text = ocr_text.strip()
    gt_text = gt_text.strip()
    if not ocr_text or not gt_text:
        return False
    
    # Length ratio
    len_ratio = max(len(ocr_text), len(gt_text)) / max(min(len(ocr_text), len(gt_text)), 1)
    if len_ratio > 3.0:
        return False
    
    # Char overlap (case-insensitive)
    ocr_set = set(c.lower() for c in ocr_text if c.isalnum())
    gt_set = set(c.lower() for c in gt_text if c.isalnum())
    if len(ocr_set) >= 2 and len(gt_set) >= 2:
        overlap = len(ocr_set & gt_set) / len(ocr_set | gt_set)
        if overlap < 0.30:
            return False
    
    # Numeric vs text mismatch
    ocr_is_num = sum(1 for c in ocr_text if c.isdigit() or c in '.,') / max(len(ocr_text), 1) > 0.5
    gt_is_num = sum(1 for c in gt_text if c.isdigit() or c in '.,') / max(len(gt_text), 1) > 0.5
    if ocr_is_num != gt_is_num:
        return False
    
    return True


# ============================================================
# Output writers
# ============================================================

def append_classification_csv(rows: list[dict]):
    """Append rows to classification CSV."""
    file_exists = CLASS_CSV.exists()
    with open(CLASS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_ocr_pair(crop_img: np.ndarray, gt_text: str, crop_id: int):
    """Save crop + ground truth text for OCR fine-tuning."""
    crop_filename = f"crop_{crop_id:06d}.png"
    crop_path = OCR_CROPS_DIR / crop_filename
    cv2.imwrite(str(crop_path), crop_img)
    with open(OCR_LABELS, 'a', encoding='utf-8') as f:
        f.write(f"images/{crop_filename}\t{gt_text}\n")


# ============================================================
# Progress
# ============================================================

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'completed': [], 'failed': [],
        'total_class_rows': 0, 'total_ocr_crops': 0
    }


def save_progress(progress: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def collect_all_images():
    """Collect (path, source_tag) tuples from configured datasets."""
    all_imgs = []
    for dir_path, tag in DATASET_DIRS:
        if not dir_path.exists():
            print(f"⚠️  Dir not found: {dir_path}")
            continue
        imgs = sorted(list(dir_path.glob("*.jpg")) +
                      list(dir_path.glob("*.png")) +
                      list(dir_path.glob("*.jpeg")))
        for img in imgs:
            all_imgs.append((img, tag))
        print(f"  📁 {tag}: {len(imgs)} images from {dir_path.name}")
    return all_imgs


# ============================================================
# Main pipeline
# ============================================================

def process_image(image_path: Path, source_tag: str, rotator: GeminiKeyRotator, ocr_engine, progress, crop_counter):
    """Process one image: OCR + Gemini → save classification + OCR ground truth.
    
    Returns: (n_class_rows, n_ocr_crops) or (-1, -1) on failure.
    """
    img, lines, img_info = get_ocr_lines(image_path, ocr_engine)
    if img is None or not lines:
        return (-1, -1)
    
    if len(lines) > 80:
        print(f"  ⚠️  Skip: too many lines ({len(lines)})")
        return (-1, -1)
    
    # Single Gemini call: get text + label per line
    gemini_results = call_gemini_combined(rotator, image_path, lines)
    if gemini_results is None:
        return (-1, -1)
    
    img_h = img_info['height']
    img_w = img_info['width']
    
    class_rows = []
    ocr_crops_saved = 0
    skipped_misalign = 0
    
    for line, gem in zip(lines, gemini_results):
        gt_text = gem['text']
        label = gem['label']
        
        # ====== Classification CSV row ======
        # Use Gemini-corrected text as the canonical text for classification
        # (better quality than raw OCR for training)
        class_rows.append({
            'source': source_tag,
            'filename': image_path.name,
            'line_index': lines.index(line),
            'text': gt_text,
            'label': label,
            'ocr_text': line.get('text', ''),
            'ocr_confidence': round(float(line.get('confidence', 0.0)), 4),
            'x_min': round(float(line.get('x_min', 0.0)), 6),
            'y_min': round(float(line.get('y_min', 0.0)), 6),
            'x_max': round(float(line.get('x_max', 0.0)), 6),
            'y_max': round(float(line.get('y_max', 0.0)), 6),
            'y_center': round(float(line.get('y_center', 0.5)), 6),
            'x_center': round(float(line.get('x_center', 0.5)), 6),
            'width': round(float(line.get('width', 0.1)), 6),
            'height': round(float(line.get('height', 0.05)), 6),
        })
        
        # ====== OCR ground truth (crop + text) ======
        # Only save if alignment is good (avoid feeding wrong text to OCR)
        ocr_text = line.get('text', '').strip()
        clean_gt = gt_text.strip()
        if not clean_gt or len(clean_gt) < 1:
            continue
        if not any(c.isalnum() for c in clean_gt):
            continue
        if not text_alignment_ok(ocr_text, clean_gt):
            skipped_misalign += 1
            continue
        
        crop = crop_line(img, line, img_h, img_w)
        if crop is None:
            continue
        
        save_ocr_pair(crop, clean_gt, crop_counter[0])
        crop_counter[0] += 1
        ocr_crops_saved += 1
    
    # Write classification rows
    append_classification_csv(class_rows)
    progress['total_class_rows'] += len(class_rows)
    progress['total_ocr_crops'] = crop_counter[0]
    
    if skipped_misalign > 0:
        print(f"  (skipped {skipped_misalign} misaligned for OCR)", end=" ")
    
    return (len(class_rows), ocr_crops_saved)


def main():
    parser = argparse.ArgumentParser(description='Combined ground truth generator (classification + OCR)')
    parser.add_argument('--api-keys', type=str, nargs='+', help='One or more Gemini API keys (auto-rotates on quota)')
    parser.add_argument('--api-key', type=str, help='[Deprecated] Single Gemini API key (use --api-keys)')
    parser.add_argument('--model', type=str, default='gemini-3.1-flash-lite', help='Gemini model name')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to process')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    parser.add_argument('--delay', type=float, default=DELAY_BETWEEN_REQUESTS, help='Delay between API calls (s)')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle images across sources for balanced dataset')
    parser.add_argument('--priority-source', type=str, nargs='+', default=None,
                        help='Process these sources first (e.g., --priority-source ID1 ID2). Available: MY, ID1, ID2')
    args = parser.parse_args()
    
    # Build API keys list (support both --api-keys and --api-key)
    api_keys = []
    if args.api_keys:
        api_keys.extend(args.api_keys)
    if args.api_key:
        api_keys.append(args.api_key)
    if not api_keys:
        env_key = os.environ.get('GEMINI_API_KEY')
        if env_key:
            api_keys.append(env_key)
    
    if not api_keys:
        print("❌ No API key. Use --api-keys KEY1 KEY2 ... or set GEMINI_API_KEY")
        return
    
    print("=" * 70)
    print("🏷️  Combined Ground Truth Generator (OCR + Classification)")
    print("=" * 70)
    print(f"🔑 Loaded {len(api_keys)} API key(s)")
    
    # Setup
    OCR_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress() if args.resume else {
        'completed': [], 'failed': [],
        'total_class_rows': 0, 'total_ocr_crops': 0
    }
    
    if not args.resume:
        if CLASS_CSV.exists():
            CLASS_CSV.unlink()
        if OCR_LABELS.exists():
            OCR_LABELS.unlink()
    
    # Init models
    print("\n📦 Initializing models...")
    rotator = GeminiKeyRotator(api_keys, model_name=args.model)
    ocr_engine = OCREngine()
    print("✅ EasyOCR ready")
    
    # Collect images
    print(f"\n📁 Collecting images:")
    all_images = collect_all_images()
    print(f"📊 Total: {len(all_images)}")
    
    # Reorder by priority source (if specified)
    if args.priority_source:
        priority = list(args.priority_source)
        print(f"🎯 Priority sources: {priority}")
        priority_imgs = [(img, tag) for img, tag in all_images if tag in priority]
        other_imgs = [(img, tag) for img, tag in all_images if tag not in priority]
        all_images = priority_imgs + other_imgs
    
    # Shuffle for balanced sampling across sources (if requested)
    if args.shuffle:
        import random
        random.seed(42)
        random.shuffle(all_images)
        print(f"🔀 Shuffled images for balanced source distribution")
    
    completed_set = set(progress['completed'])
    failed_set = set(progress['failed'])
    remaining = [(img, tag) for img, tag in all_images
                 if img.name not in completed_set and img.name not in failed_set]
    if args.max_images:
        remaining = remaining[:args.max_images]
    
    print(f"✅ Done: {len(completed_set)} | ❌ Failed: {len(failed_set)} | 🎯 To process: {len(remaining)}")
    print(f"📋 Existing class rows: {progress['total_class_rows']}")
    print(f"🖼️  Existing OCR crops: {progress['total_ocr_crops']}")
    print()
    
    crop_counter = [progress['total_ocr_crops']]
    processed = 0
    errors = 0
    
    for idx, (image_path, source_tag) in enumerate(remaining):
        filename = image_path.name
        print(f"[{idx+1}/{len(remaining)}] [{source_tag}] {filename}...", end=" ", flush=True)
        
        try:
            n_class, n_ocr = process_image(
                image_path, source_tag, rotator, ocr_engine, progress, crop_counter
            )
            
            if n_class < 0:
                print("❌")
                progress['failed'].append(filename)
                errors += 1
            else:
                print(f"✅ {n_class} cls, {n_ocr} ocr crops")
                progress['completed'].append(filename)
                processed += 1
            
            # Save progress every 5 images
            if (idx + 1) % 5 == 0:
                save_progress(progress)
                print(f"   💾 saved (total: {progress['total_class_rows']} cls / {progress['total_ocr_crops']} ocr)")
            
            time.sleep(args.delay)
        
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted")
            break
        except Exception as e:
            print(f"❌ {e}")
            progress['failed'].append(filename)
            errors += 1
            time.sleep(args.delay)
    
    save_progress(progress)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"  Processed         : {processed}")
    print(f"  Failed            : {errors}")
    print(f"  Classification rows: {progress['total_class_rows']}")
    print(f"  OCR crops          : {progress['total_ocr_crops']}")
    print(f"  Class CSV          : {CLASS_CSV}")
    print(f"  OCR labels.txt     : {OCR_LABELS}")
    print(f"  OCR crops dir      : {OCR_CROPS_DIR}")
    
    if progress['total_class_rows'] > 0:
        print(f"\n✅ Both datasets ready!")
        print(f"   Classifier training: python scripts/train_classifier_v2.py")
        print(f"   OCR fine-tuning:     python scripts/finetune_easyocr.py")


if __name__ == '__main__':
    main()
