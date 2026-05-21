"""
Step 1: Generate OCR Ground Truth menggunakan Gemini Vision API
================================================================
Script ini:
1. Load gambar struk
2. Kirim ke Gemini Vision → minta ground truth teks per baris
3. Crop setiap baris menggunakan bbox dari EasyOCR
4. Simpan pasangan (crop_image, ground_truth_text)

Output:
    data/ocr_groundtruth/
        images/          ← crop gambar per baris (grayscale, height=32px)
        labels.txt       ← format: "images/xxx.png\tground_truth_text"
        metadata.json    ← info lengkap per sample

Usage:
    wsl python3 scripts/generate_ocr_groundtruth.py --api-key YOUR_KEY --max-images 100
    wsl python3 scripts/generate_ocr_groundtruth.py --api-key YOUR_KEY --max-images 500
"""

import os
import sys
import json
import time
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ROOT_DIR
from src.preprocessing import load_image, deskew
from src.ocr_engine import OCREngine

# Gemini
try:
    import google.generativeai as genai
except ImportError:
    print("❌ Install: pip install google-generativeai")
    sys.exit(1)

# ============================================================
# Paths
# ============================================================
IMAGES_DIR = ROOT_DIR / "fullDataset" / "images"
OUTPUT_DIR = ROOT_DIR / "data" / "ocr_groundtruth"
CROPS_DIR = OUTPUT_DIR / "images"
LABELS_FILE = OUTPUT_DIR / "labels.txt"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Rate limiting (Gemini free tier: 15 req/min)
DELAY_BETWEEN_REQUESTS = 4.5  # seconds

# Crop settings (EasyOCR recognition input)
CROP_HEIGHT = 32   # EasyOCR uses 32px height
CROP_PAD_X = 4     # horizontal padding
CROP_PAD_Y = 2     # vertical padding

# ============================================================
# Gemini Prompt
# ============================================================
GROUNDTRUTH_PROMPT = """You are an expert OCR correction system for receipt images.

I will provide you:
1. A receipt image
2. EasyOCR's reading of each line (which may have errors)

Your task: For EACH line, provide the CORRECT text by looking at the image.
Keep the SAME number of lines and SAME order as EasyOCR provided.
Just CORRECT what EasyOCR got wrong - do NOT merge or split lines.

Common OCR errors to fix:
- O ↔ 0 (letter O vs digit 0)
- I ↔ 1 ↔ l (letter I vs digit 1 vs lowercase L)
- S ↔ 5
- B ↔ 8
- typos in store names, addresses
- wrong characters in prices
- missing or wrong punctuation

Return a JSON array - SAME LENGTH as input - with the corrected text:
[
  {"line_num": 0, "corrected": "INDOMARET"},
  {"line_num": 1, "corrected": "Jl. Sudirman No. 123"},
  ...
]

CRITICAL RULES:
1. Output array MUST have EXACTLY the same number of items as input
2. Keep line order matching the input order
3. If a line is correct, just repeat it
4. If a line is unreadable noise (like garbage characters), keep it as-is
5. Do NOT add or remove lines
6. Do NOT merge multiple lines into one
7. Return ONLY the JSON array, no explanation, no markdown

Here are the EasyOCR lines (in reading order):
"""


# ============================================================
# Gemini Functions
# ============================================================

def init_gemini(api_key: str, model_name: str = 'gemini-3.1-flash-lite'):
    """Initialize Gemini Vision model."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    print(f"✅ Gemini Vision initialized ({model_name})")
    return model


def get_groundtruth_from_gemini(model, image_path: Path, easyocr_lines: list[dict]) -> list[str] | None:
    """Send receipt image + EasyOCR lines to Gemini, get corrected text per line.
    
    Strategy: 1:1 line correction - Gemini sees image AND EasyOCR output,
    then corrects each line keeping the SAME order and count.
    
    Returns:
        List of corrected strings (same length as easyocr_lines)
        None if failed
    """
    try:
        # Build EasyOCR text list for Gemini context
        ocr_lines_text = "\n".join(
            f'{i}: "{line.get("text", "")}"' for i, line in enumerate(easyocr_lines)
        )
        
        # Load image for Gemini
        img = Image.open(image_path)
        
        # Send to Gemini with both image and OCR context
        full_prompt = GROUNDTRUTH_PROMPT + ocr_lines_text
        response = model.generate_content([full_prompt, img])
        response_text = response.text.strip()
        
        # Clean markdown code blocks if present
        if response_text.startswith("```"):
            response_text = re.sub(r'^```\w*\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)
        
        # Parse JSON
        result = json.loads(response_text)
        
        # Validate: must have same length as input
        if not isinstance(result, list):
            print(f"  ⚠️  Response not a list")
            return None
        
        # Build corrected text list (1:1 with easyocr_lines)
        corrected = [None] * len(easyocr_lines)
        for item in result:
            if not isinstance(item, dict):
                continue
            idx = item.get('line_num', -1)
            text = item.get('corrected', '')
            if 0 <= idx < len(easyocr_lines):
                corrected[idx] = str(text).strip()
        
        # Fill any missing entries with original EasyOCR text (fallback)
        for i in range(len(easyocr_lines)):
            if corrected[i] is None:
                corrected[i] = easyocr_lines[i].get('text', '').strip()
        
        return corrected
        
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse error: {e}")
        print(f"  Response preview: {response_text[:200]}")
        return None
    except Exception as e:
        print(f"  ⚠️  Gemini error: {type(e).__name__}: {e}")
        return None


# ============================================================
# OCR + Crop Functions
# ============================================================

def get_easyocr_lines(image_path: Path, ocr_engine: OCREngine) -> tuple[list[dict], np.ndarray, dict]:
    """Run EasyOCR on image and return lines with bbox info.
    
    Returns:
        (lines, original_image, image_info)
    """
    img = load_image(image_path)
    img = deskew(img)
    lines, img_info = ocr_engine.read_receipt_with_image_info(img)
    return lines, img, img_info


def crop_line_from_image(img: np.ndarray, line: dict, img_h: int, img_w: int) -> np.ndarray | None:
    """Crop a single text line from the image using bbox coordinates.
    
    Args:
        img: Original image (BGR or grayscale)
        line: Line dict with x_min, y_min, x_max, y_max (normalized 0-1)
        img_h, img_w: Image dimensions
        
    Returns:
        Cropped grayscale image resized to height=32px, or None if invalid
    """
    # Convert normalized coords back to pixels
    x_min = int(line['x_min'] * img_w) - CROP_PAD_X
    y_min = int(line['y_min'] * img_h) - CROP_PAD_Y
    x_max = int(line['x_max'] * img_w) + CROP_PAD_X
    y_max = int(line['y_max'] * img_h) + CROP_PAD_Y
    
    # Clamp to image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)
    
    # Validate crop size
    if x_max <= x_min or y_max <= y_min:
        return None
    if (x_max - x_min) < 5 or (y_max - y_min) < 5:
        return None
    
    # Crop
    if len(img.shape) == 3:
        crop = img[y_min:y_max, x_min:x_max]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop = img[y_min:y_max, x_min:x_max]
    
    # Resize to height=32 (EasyOCR standard), maintain aspect ratio
    h, w = crop.shape
    new_w = max(int(w * CROP_HEIGHT / h), 1)
    crop_resized = cv2.resize(crop, (new_w, CROP_HEIGHT), interpolation=cv2.INTER_CUBIC)
    
    return crop_resized


def match_gemini_to_easyocr(gemini_lines: list[dict], easyocr_lines: list[dict]) -> list[tuple]:
    """Match Gemini ground truth lines to EasyOCR bbox lines by position.
    
    Strategy: Match by y_center position overlap.
    
    Returns:
        List of (easyocr_line, gemini_text) tuples
    """
    matched = []
    
    for ocr_line in easyocr_lines:
        ocr_y_center = ocr_line.get('y_center', 0.5)
        
        # Find best matching Gemini line by y position
        best_match = None
        best_distance = float('inf')
        
        for gem_line in gemini_lines:
            gem_y_center = (gem_line['y_start_pct'] + gem_line['y_end_pct']) / 2
            distance = abs(ocr_y_center - gem_y_center)
            
            if distance < best_distance:
                best_distance = distance
                best_match = gem_line
        
        # Accept match if close enough (within 5% of image height)
        if best_match and best_distance < 0.05:
            matched.append((ocr_line, best_match['text']))
        else:
            # Use EasyOCR text as fallback (still useful for training)
            matched.append((ocr_line, ocr_line.get('text', '')))
    
    return matched


# ============================================================
# Progress Management
# ============================================================

def load_progress() -> dict:
    """Load progress file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'completed': [], 'failed': [], 'total_crops': 0}


def save_progress(progress: dict):
    """Save progress file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


# ============================================================
# Main Pipeline
# ============================================================

def process_image(
    image_path: Path,
    gemini_model,
    ocr_engine: OCREngine,
    metadata: list,
    crop_counter: list,  # mutable counter
) -> int:
    """Process one image: EasyOCR first, then Gemini correction per line.
    
    Returns:
        Number of crops generated, -1 if failed
    """
    filename = image_path.name
    
    # Step 1: Get EasyOCR lines with bbox FIRST
    try:
        easyocr_lines, img, img_info = get_easyocr_lines(image_path, ocr_engine)
    except Exception as e:
        print(f"  ⚠️  OCR error: {e}")
        return -1
    
    if not easyocr_lines:
        print(f"  ⚠️  No OCR lines detected")
        return -1
    
    img_h = img_info['height']
    img_w = img_info['width']
    
    # Step 2: Send image + EasyOCR lines to Gemini for per-line correction
    corrected_texts = get_groundtruth_from_gemini(gemini_model, image_path, easyocr_lines)
    if corrected_texts is None:
        return -1
    
    if len(corrected_texts) != len(easyocr_lines):
        print(f"  ⚠️  Length mismatch: {len(corrected_texts)} vs {len(easyocr_lines)}")
        return -1
    
    # Step 3: Crop each line and save with corrected text as ground truth
    crops_saved = 0
    skipped_misalign = 0
    
    for ocr_line, ground_truth_text in zip(easyocr_lines, corrected_texts):
        # Skip empty or very short ground truth
        if not ground_truth_text or len(ground_truth_text.strip()) < 1:
            continue
        
        # Skip lines with only special characters (likely noise)
        clean_text = ground_truth_text.strip()
        if not any(c.isalnum() for c in clean_text):
            continue
        
        # ALIGNMENT CHECK: Skip if Gemini text is wildly different from EasyOCR text
        # This catches cases where Gemini returned wrong line content
        ocr_text = ocr_line.get('text', '').strip()
        if ocr_text and clean_text:
            # Rule 1: Length must be similar
            len_ratio = max(len(ocr_text), len(clean_text)) / max(min(len(ocr_text), len(clean_text)), 1)
            if len_ratio > 3.0:
                skipped_misalign += 1
                continue
            
            # Rule 2: Char overlap must be reasonable
            ocr_alnum = set(c.lower() for c in ocr_text if c.isalnum())
            gt_alnum = set(c.lower() for c in clean_text if c.isalnum())
            
            if len(ocr_alnum) >= 2 and len(gt_alnum) >= 2:
                overlap = len(ocr_alnum & gt_alnum) / len(ocr_alnum | gt_alnum)
                if overlap < 0.30:  # less than 30% char overlap = misaligned
                    skipped_misalign += 1
                    continue
            
            # Rule 3: For mostly-digit lines (prices/numbers), require strict overlap
            ocr_is_numeric = sum(1 for c in ocr_text if c.isdigit() or c in '.,') / max(len(ocr_text), 1) > 0.5
            gt_is_numeric = sum(1 for c in clean_text if c.isdigit() or c in '.,') / max(len(clean_text), 1) > 0.5
            
            # If one is numeric and other is text, definitely misaligned
            if ocr_is_numeric != gt_is_numeric:
                skipped_misalign += 1
                continue
            
            # Rule 4: For numeric lines, require digit-set overlap
            if ocr_is_numeric and gt_is_numeric:
                ocr_digits = set(c for c in ocr_text if c.isdigit())
                gt_digits = set(c for c in clean_text if c.isdigit())
                if ocr_digits and gt_digits:
                    digit_overlap = len(ocr_digits & gt_digits) / len(ocr_digits | gt_digits)
                    if digit_overlap < 0.40:
                        skipped_misalign += 1
                        continue
        
        # Crop the line
        crop = crop_line_from_image(img, ocr_line, img_h, img_w)
        if crop is None:
            continue
        
        # Save crop
        crop_id = crop_counter[0]
        crop_filename = f"crop_{crop_id:06d}.png"
        crop_path = CROPS_DIR / crop_filename
        
        cv2.imwrite(str(crop_path), crop)
        
        # Save to labels.txt
        with open(LABELS_FILE, 'a', encoding='utf-8') as f:
            f.write(f"images/{crop_filename}\t{clean_text}\n")
        
        # Save metadata
        metadata.append({
            'crop_id': crop_id,
            'crop_file': crop_filename,
            'source_image': filename,
            'ground_truth': clean_text,
            'easyocr_text': ocr_line.get('text', ''),
            'ocr_confidence': ocr_line.get('confidence', 0.0),
            'y_center': ocr_line.get('y_center', 0.5),
            'x_center': ocr_line.get('x_center', 0.5),
            'crop_width': crop.shape[1],
            'crop_height': crop.shape[0],
        })
        
        crop_counter[0] += 1
        crops_saved += 1
    
    if skipped_misalign > 0:
        print(f"    ⚠️  Skipped {skipped_misalign} misaligned lines", end=" ")
    
    return crops_saved


def main():
    parser = argparse.ArgumentParser(description='Generate OCR ground truth with Gemini Vision')
    parser.add_argument('--api-key', type=str, help='Gemini API key')
    parser.add_argument('--model', type=str, default='gemini-3.1-flash-lite', help='Gemini model name')
    parser.add_argument('--max-images', type=int, default=100, help='Max images to process (default: 100)')
    parser.add_argument('--images-dir', type=str, default=None, help='Custom images directory')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("❌ No API key!")
        print("   Use: --api-key YOUR_KEY")
        print("   Or:  export GEMINI_API_KEY=YOUR_KEY")
        print("   Get free key: https://aistudio.google.com/apikey")
        return
    
    print("=" * 65)
    print("🔍 OCR Ground Truth Generator (Gemini Vision)")
    print("=" * 65)
    
    # Setup output dirs
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load progress
    progress = load_progress() if args.resume else {'completed': [], 'failed': [], 'total_crops': 0}
    
    # Load existing metadata
    metadata = []
    if METADATA_FILE.exists() and args.resume:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # Clear labels file if not resuming
    if not args.resume:
        with open(LABELS_FILE, 'w', encoding='utf-8') as f:
            f.write("")  # Clear
    
    # Init models
    print("\n📦 Initializing models...")
    gemini_model = init_gemini(api_key, args.model)
    
    print("📦 Initializing EasyOCR...")
    ocr_engine = OCREngine()
    print("✅ EasyOCR ready")
    
    # Get images
    images_dir = Path(args.images_dir) if args.images_dir else IMAGES_DIR
    all_images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    # Filter already processed
    remaining = [img for img in all_images if img.name not in progress['completed'] and img.name not in progress['failed']]
    remaining = remaining[:args.max_images]
    
    print(f"\n📁 Images dir: {images_dir}")
    print(f"📊 Total images: {len(all_images)}")
    print(f"✅ Already done: {len(progress['completed'])}")
    print(f"🎯 To process: {len(remaining)}")
    print(f"📦 Existing crops: {progress['total_crops']}")
    print()
    
    # Mutable counter for crop IDs
    crop_counter = [progress['total_crops']]
    
    processed = 0
    errors = 0
    
    for idx, image_path in enumerate(remaining):
        filename = image_path.name
        print(f"[{idx+1}/{len(remaining)}] {filename}...", end=" ", flush=True)
        
        try:
            crops_saved = process_image(
                image_path, gemini_model, ocr_engine, metadata, crop_counter
            )
            
            if crops_saved < 0:
                print(f"❌ Failed")
                progress['failed'].append(filename)
                errors += 1
            else:
                print(f"✅ {crops_saved} crops")
                progress['completed'].append(filename)
                processed += 1
            
            # Save progress every 5 images
            if (idx + 1) % 5 == 0:
                progress['total_crops'] = crop_counter[0]
                save_progress(progress)
                with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                print(f"   💾 Saved ({crop_counter[0]} total crops)")
            
            # Rate limiting
            time.sleep(DELAY_BETWEEN_REQUESTS)
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Interrupted!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            progress['failed'].append(filename)
            errors += 1
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Final save
    progress['total_crops'] = crop_counter[0]
    save_progress(progress)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "=" * 65)
    print("📊 SUMMARY")
    print("=" * 65)
    print(f"  Images processed : {processed}")
    print(f"  Images failed    : {errors}")
    print(f"  Total crops      : {crop_counter[0]}")
    print(f"  Labels file      : {LABELS_FILE}")
    print(f"  Metadata file    : {METADATA_FILE}")
    print(f"  Crops dir        : {CROPS_DIR}")
    
    if crop_counter[0] > 0:
        print(f"\n✅ Dataset ready!")
        print(f"   Next step: python3 scripts/finetune_easyocr.py")
    else:
        print(f"\n⚠️  No crops generated. Check API key and images.")


if __name__ == '__main__':
    main()
