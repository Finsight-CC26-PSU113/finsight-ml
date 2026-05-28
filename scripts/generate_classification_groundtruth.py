"""
Generate Classification Ground Truth using Gemini Vision API
=============================================================
Untuk training BiLSTM classifier per-baris struk.

Pipeline:
1. Load gambar struk (Malaysia + Indonesia)
2. EasyOCR detect bbox + text per line
3. Send (image + lines) ke Gemini Vision → label per line
4. Save CSV: filename, line_index, text, label, bbox features

Categories:
- STORE             : Nama toko
- ADDRESS_CONTACT   : Alamat, telp, email, NPWP
- DATE              : Tanggal & waktu (any format including "Mar 14 2025")
- ITEM_DESC         : Deskripsi barang
- ITEM_PRICE/QTY    : Harga & jumlah barang
- TOTAL_PAYMENT     : Total akhir (yang harus dibayar)
- OTHER             : Footer, kasir, greetings, dll

Usage:
    wsl python3 scripts/generate_classification_groundtruth.py --api-key YOUR_KEY
    wsl python3 scripts/generate_classification_groundtruth.py --api-key YOUR_KEY --max-images 500 --resume
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
    # (path, country_tag, max_images_to_take)
    (ROOT_DIR / "fullDataset" / "images", "MY", None),                                                    # Malaysia
    (ROOT_DIR / "Dataset Baru" / "Data Tahap Pertama - 26 April 26" / "Data Bersih Final", "ID1", None),  # Indo 1
    (ROOT_DIR / "Dataset Baru" / "Data Tahap Kedua-29 April 2026" / "Data Bersih", "ID2", None),          # Indo 2
]

OUTPUT_DIR = ROOT_DIR / "data" / "classification_groundtruth"
LABELS_CSV = OUTPUT_DIR / "labels.csv"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Rate limiting (Gemini free tier: 15 req/min)
DELAY_BETWEEN_REQUESTS = 4.5  # seconds

# Valid labels
VALID_LABELS = {
    'STORE', 'ADDRESS_CONTACT', 'DATE',
    'ITEM_DESC', 'ITEM_PRICE/QTY', 'TOTAL_PAYMENT', 'OTHER'
}


# ============================================================
# Gemini Prompt
# ============================================================
CLASSIFICATION_PROMPT = """You are an expert receipt analyzer.

I will provide you:
1. A receipt image
2. EasyOCR-extracted lines from the receipt (with line numbers)

Your task: For EACH line, classify it into ONE of these categories:

CATEGORIES:
- STORE: Store/business name (e.g., "INDOMARET", "WARUNG SARI", "AEON")
- ADDRESS_CONTACT: Address, phone, email, website, NPWP, GST/ROC number
- DATE: Date and/or time (ANY format like "26-11-17", "Mar 14 2025", "24/05/2026", "10:30:45", "Tanggal: 24 Mei 2026")
- ITEM_DESC: Product/item descriptions (e.g., "Nasi Goreng", "Indomie Goreng")
- ITEM_PRICE/QTY: Item prices, quantities, or qty x price (e.g., "Rp 25.000", "2 x 5000", "@9.80", "RM 8.50")
- TOTAL_PAYMENT: Final total payment, subtotal, tax, discount, cash, change
  (Words like: Total, Subtotal, Grand Total, Tunai, Cash, PPN, Tax, Diskon, Kembali, Change)
- OTHER: Anything else (cashier, greetings, footer text, transaction ID, table number, etc.)

CRITICAL RULES:
1. Output a JSON array with EXACTLY the same number of items as input lines
2. Each item: {"line_num": <index>, "label": "<CATEGORY>"}
3. Use ONLY the 7 categories listed above (exact spelling, case-sensitive)
4. For dates in any format including word months ("Mar 14 2025") → DATE
5. Time-only lines like "10:30 AM" or "Jam: 14:30" → DATE
6. Phone numbers and addresses → ADDRESS_CONTACT
7. Transaction IDs, receipt numbers (like "TRX-001234") → OTHER
8. Member numbers, table numbers → OTHER
9. Return ONLY the JSON array, no explanation, no markdown

EXAMPLE OUTPUT:
[
  {"line_num": 0, "label": "STORE"},
  {"line_num": 1, "label": "ADDRESS_CONTACT"},
  {"line_num": 2, "label": "DATE"},
  {"line_num": 3, "label": "ITEM_DESC"},
  {"line_num": 4, "label": "ITEM_PRICE/QTY"},
  {"line_num": 5, "label": "TOTAL_PAYMENT"}
]

Here are the lines from EasyOCR (in reading order):
"""


# ============================================================
# Gemini Functions
# ============================================================

def init_gemini(api_key: str, model_name: str = 'gemini-2.5-flash-lite'):
    """Initialize Gemini Vision model."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    print(f"✅ Gemini Vision initialized ({model_name})")
    return model


def classify_lines_with_gemini(model, image_path: Path, lines: list[dict]) -> list[str] | None:
    """Send image + lines to Gemini, get classification per line.
    
    Returns:
        List of labels (same length as lines), None if failed
    """
    try:
        # Build context
        ocr_context = "\n".join(
            f'{i}: "{line.get("text", "")}"' for i, line in enumerate(lines)
        )
        
        # Load image
        img = Image.open(image_path)
        
        # Send to Gemini
        full_prompt = CLASSIFICATION_PROMPT + ocr_context
        response = model.generate_content([full_prompt, img])
        response_text = response.text.strip()
        
        # Clean markdown
        if response_text.startswith("```"):
            response_text = re.sub(r'^```\w*\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)
        
        result = json.loads(response_text)
        
        if not isinstance(result, list):
            return None
        
        # Build labels (1:1 with lines)
        labels = ['OTHER'] * len(lines)
        for item in result:
            if not isinstance(item, dict):
                continue
            idx = item.get('line_num', -1)
            label = item.get('label', 'OTHER').strip()
            if 0 <= idx < len(lines) and label in VALID_LABELS:
                labels[idx] = label
        
        return labels
        
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  ⚠️  Gemini error: {type(e).__name__}: {e}")
        return None


# ============================================================
# OCR Function
# ============================================================

def get_ocr_lines(image_path: Path, ocr_engine: OCREngine) -> tuple[list[dict], dict] | None:
    """Run EasyOCR on image."""
    try:
        img = load_image(image_path)
        img = deskew(img)
        lines, img_info = ocr_engine.read_receipt_with_image_info(img)
        return lines, img_info
    except Exception as e:
        print(f"  ⚠️  OCR error: {e}")
        return None


# ============================================================
# Progress Management
# ============================================================

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'completed': [], 'failed': [], 'total_rows': 0}


def save_progress(progress: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


# ============================================================
# Image Collection
# ============================================================

def collect_all_images() -> list[tuple[Path, str]]:
    """Collect all images from configured datasets.
    
    Returns:
        List of (image_path, country_tag)
    """
    all_images = []
    for dir_path, tag, max_n in DATASET_DIRS:
        if not dir_path.exists():
            print(f"⚠️  Dir not found: {dir_path}")
            continue
        
        imgs = sorted(list(dir_path.glob("*.jpg")) + 
                     list(dir_path.glob("*.png")) + 
                     list(dir_path.glob("*.jpeg")))
        if max_n:
            imgs = imgs[:max_n]
        
        for img in imgs:
            all_images.append((img, tag))
        print(f"  📁 {tag}: {len(imgs)} images from {dir_path.name}")
    
    return all_images


# ============================================================
# CSV Writing
# ============================================================

CSV_FIELDS = [
    'source', 'filename', 'line_index', 'text', 'label',
    'ocr_confidence',
    'x_min', 'y_min', 'x_max', 'y_max',
    'y_center', 'x_center', 'width', 'height',
]


def append_csv_rows(rows: list[dict]):
    """Append rows to labels CSV."""
    file_exists = LABELS_CSV.exists()
    with open(LABELS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# Main Pipeline
# ============================================================

def process_image(
    image_path: Path,
    source_tag: str,
    gemini_model,
    ocr_engine: OCREngine,
    progress: dict,
) -> int:
    """Process one image: OCR + Gemini classification + save CSV.
    
    Returns:
        Number of rows added, -1 if failed
    """
    # OCR
    ocr_result = get_ocr_lines(image_path, ocr_engine)
    if not ocr_result:
        return -1
    
    lines, img_info = ocr_result
    if not lines:
        return -1
    
    # Skip if too many lines (likely noisy image, also keeps Gemini token usage down)
    if len(lines) > 80:
        print(f"  ⚠️  Skip: too many lines ({len(lines)})")
        return -1
    
    # Classify with Gemini
    labels = classify_lines_with_gemini(gemini_model, image_path, lines)
    if labels is None:
        return -1
    
    if len(labels) != len(lines):
        print(f"  ⚠️  Length mismatch: {len(labels)} vs {len(lines)}")
        return -1
    
    # Build CSV rows
    rows = []
    for i, (line, label) in enumerate(zip(lines, labels)):
        rows.append({
            'source': source_tag,
            'filename': image_path.name,
            'line_index': i,
            'text': line.get('text', ''),
            'label': label,
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
    
    append_csv_rows(rows)
    progress['total_rows'] += len(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description='Generate classification ground truth with Gemini Vision')
    parser.add_argument('--api-key', type=str, help='Gemini API key')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash-lite', help='Gemini model')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to process (default: all)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    parser.add_argument('--delay', type=float, default=DELAY_BETWEEN_REQUESTS, help='Delay between API calls (seconds)')
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("❌ No API key!")
        print("   Use: --api-key YOUR_KEY")
        return
    
    print("=" * 70)
    print("🏷️  Receipt Line Classification — Ground Truth Generator")
    print("=" * 70)
    
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress() if args.resume else {'completed': [], 'failed': [], 'total_rows': 0}
    
    # Clear CSV if not resuming
    if not args.resume and LABELS_CSV.exists():
        LABELS_CSV.unlink()
    
    # Init models
    print("\n📦 Initializing models...")
    gemini_model = init_gemini(api_key, args.model)
    ocr_engine = OCREngine()
    print("✅ EasyOCR ready")
    
    # Collect images
    print(f"\n📁 Collecting images from {len(DATASET_DIRS)} sources:")
    all_images = collect_all_images()
    print(f"📊 Total images: {len(all_images)}")
    
    # Filter already processed
    completed_set = set(progress['completed'])
    failed_set = set(progress['failed'])
    remaining = [(img, tag) for img, tag in all_images 
                 if img.name not in completed_set and img.name not in failed_set]
    
    if args.max_images:
        remaining = remaining[:args.max_images]
    
    print(f"✅ Already done: {len(completed_set)}")
    print(f"❌ Already failed: {len(failed_set)}")
    print(f"🎯 To process: {len(remaining)}")
    print(f"📦 Existing rows: {progress['total_rows']}")
    print(f"⏱️  Delay between requests: {args.delay}s")
    print()
    
    processed = 0
    errors = 0
    
    for idx, (image_path, source_tag) in enumerate(remaining):
        filename = image_path.name
        print(f"[{idx+1}/{len(remaining)}] [{source_tag}] {filename}...", end=" ", flush=True)
        
        try:
            n_rows = process_image(image_path, source_tag, gemini_model, ocr_engine, progress)
            
            if n_rows < 0:
                print("❌ Failed")
                progress['failed'].append(filename)
                errors += 1
            else:
                print(f"✅ {n_rows} lines")
                progress['completed'].append(filename)
                processed += 1
            
            # Save progress every 5 images
            if (idx + 1) % 5 == 0:
                save_progress(progress)
                print(f"   💾 Saved (total {progress['total_rows']} rows)")
            
            time.sleep(args.delay)
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Interrupted!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            progress['failed'].append(filename)
            errors += 1
            time.sleep(args.delay)
    
    # Final save
    save_progress(progress)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"  Processed     : {processed}")
    print(f"  Failed        : {errors}")
    print(f"  Total rows    : {progress['total_rows']}")
    print(f"  Output CSV    : {LABELS_CSV}")
    print(f"  Progress file : {PROGRESS_FILE}")
    
    if progress['total_rows'] > 0:
        print(f"\n✅ Dataset ready!")
        print(f"   Next: wsl python3 scripts/train_classifier_v2.py")


if __name__ == '__main__':
    main()
