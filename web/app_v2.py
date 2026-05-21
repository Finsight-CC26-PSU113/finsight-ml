"""
OCR FinSight - Web UI (Model V2)

Pipeline:
    Upload → OCR (EasyOCR) → Classify (TF Model) → Format Output

Philosophy:
    - Trust the model's predictions
    - Only do FORMATTING in post-processing (parse date, extract amount)
    - NO rule-based label overrides

Usage:
    python3 web/app_v2.py
"""

import sys
import re
import numpy as np
import tensorflow as tf
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import easyocr
import cv2
import base64

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ROOT_DIR, NUM_CLASSES, LINE_CLASSES, MAX_TEXT_LENGTH
from src.model_v2 import HierarchicalReceiptClassifier
from src.model_v2_utils import (
    prepare_single_receipt, MAX_LINES_PER_RECEIPT,
    NUM_TEXT_FEATURES, NUM_POS_FEATURES
)
from src.preprocessing import deskew

app = Flask(__name__, template_folder='templates_v2', static_folder='static_v2')

model = None
ocr_reader = None


# ============================================================
# Initialization
# ============================================================

def init_model():
    global model
    if model is not None:
        return
    print("Loading TF model...")
    model = HierarchicalReceiptClassifier(num_classes=NUM_CLASSES)
    
    dummy_chars = tf.zeros((1, MAX_LINES_PER_RECEIPT, MAX_TEXT_LENGTH), dtype=tf.int32)
    dummy_text = tf.zeros((1, MAX_LINES_PER_RECEIPT, NUM_TEXT_FEATURES), dtype=tf.float32)
    dummy_pos = tf.zeros((1, MAX_LINES_PER_RECEIPT, NUM_POS_FEATURES), dtype=tf.float32)
    dummy_mask = tf.ones((1, MAX_LINES_PER_RECEIPT), dtype=tf.float32)
    _ = model((dummy_chars, dummy_text, dummy_pos, dummy_mask), training=False)
    
    weights_path = ROOT_DIR / "models" / "v2" / "best_weights.weights.h5"
    model.load_weights(str(weights_path))
    print(f"✅ Model loaded ({weights_path})")


def init_ocr():
    global ocr_reader
    if ocr_reader is not None:
        return
    print("Loading EasyOCR (CPU mode)...")
    ocr_reader = easyocr.Reader(['en', 'id'], gpu=False, verbose=False)
    print("✅ EasyOCR ready")


# ============================================================
# Core Pipeline
# ============================================================

def ocr_image(img):
    """Run OCR and return formatted lines."""
    raw_results = ocr_reader.readtext(img)
    if not raw_results:
        return []
    
    h, w = img.shape[:2]
    lines = []
    for bbox, text, conf in raw_results:
        x_min = min(p[0] for p in bbox) / w
        y_min = min(p[1] for p in bbox) / h
        x_max = max(p[0] for p in bbox) / w
        y_max = max(p[1] for p in bbox) / h
        
        lines.append({
            'text': text,
            'confidence': float(conf),
            'y_center': (y_min + y_max) / 2,
            'x_center': (x_min + x_max) / 2,
            'width': x_max - x_min,
            'height': y_max - y_min,
            'bbox': bbox,
            'bbox_norm': {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max},
        })
    return lines


def classify_lines(lines):
    """Classify lines using the TF model. Trust the model."""
    if not lines:
        return []
    
    # Use only first MAX_LINES_PER_RECEIPT (model capacity)
    lines_for_model = lines[:MAX_LINES_PER_RECEIPT]
    char_ids, text_feats, pos_feats, mask = prepare_single_receipt(lines_for_model)
    
    inputs = (
        tf.expand_dims(char_ids, 0),
        tf.expand_dims(text_feats, 0),
        tf.expand_dims(pos_feats, 0),
        tf.expand_dims(mask, 0),
    )
    
    predictions = model(inputs, training=False)[0].numpy()
    
    results = []
    for i, line in enumerate(lines):
        if i < MAX_LINES_PER_RECEIPT:
            pred = predictions[i]
            idx = int(np.argmax(pred))
            confidence = float(pred[idx])
            label = LINE_CLASSES[idx]
        else:
            # Beyond model capacity: mark as OTHER (model doesn't know)
            label = 'OTHER'
            confidence = 0.5
        
        results.append({
            'text': line['text'],
            'label': label,
            'confidence': round(confidence, 3),
            'ocr_confidence': round(line['confidence'], 3),
            'bbox_norm': {k: round(float(v), 4) for k, v in line['bbox_norm'].items()},
            'bbox': [[int(p[0]), int(p[1])] for p in line['bbox']],
            'y_center': line['y_center'],
        })
    return results


def merge_horizontal_fragments(results):
    """Merge OCR fragments on the same row with the same label.
    
    Pure formatting - doesn't change classifications.
    Only merges ITEM_DESC, STORE, ADDRESS_CONTACT (text fields).
    Does NOT merge TOTAL_PAYMENT or ITEM_PRICE/QTY (each is distinct info).
    """
    if not results:
        return results
    
    Y_THRESHOLD = 0.02
    MERGE_LABELS = {'ITEM_DESC', 'STORE', 'ADDRESS_CONTACT'}
    
    merged = []
    used = set()
    
    for i, r in enumerate(results):
        if i in used:
            continue
        
        if r['label'] not in MERGE_LABELS:
            merged.append(r)
            used.add(i)
            continue
        
        group = [r]
        used.add(i)
        
        for j in range(i + 1, len(results)):
            if j in used:
                continue
            other = results[j]
            if (other['label'] == r['label'] and
                abs(other['bbox_norm']['y_min'] - r['bbox_norm']['y_min']) < Y_THRESHOLD):
                group.append(other)
                used.add(j)
        
        if len(group) == 1:
            merged.append(r)
        else:
            group.sort(key=lambda x: x['bbox_norm']['x_min'])
            merged.append({
                'text': ' '.join(g['text'] for g in group),
                'label': r['label'],
                'confidence': sum(g['confidence'] for g in group) / len(group),
                'ocr_confidence': sum(g['ocr_confidence'] for g in group) / len(group),
                'bbox_norm': {
                    'x_min': min(g['bbox_norm']['x_min'] for g in group),
                    'y_min': min(g['bbox_norm']['y_min'] for g in group),
                    'x_max': max(g['bbox_norm']['x_max'] for g in group),
                    'y_max': max(g['bbox_norm']['y_max'] for g in group),
                },
                'bbox': group[0]['bbox'],
                'y_center': r['y_center'],
            })
    return merged


# ============================================================
# Output Formatting (NOT classification)
# ============================================================

def format_store(results):
    """Pick best STORE name from model's STORE-labeled lines."""
    store_lines = [r['text'] for r in results if r['label'] == 'STORE']
    if not store_lines:
        return 'N/A'
    if len(store_lines) == 1:
        return store_lines[0]
    
    # Deduplicate near-duplicates (OCR reads logo + text)
    seen = []
    for s in store_lines:
        normalized = re.sub(r'[^A-Z]', '', s.upper())
        is_dup = any(normalized in n or n in normalized for n in seen if normalized and n)
        if not is_dup:
            seen.append(normalized)
    
    # Return the longest one (usually most complete)
    return max(store_lines, key=len)


def format_date(results):
    """Parse date from DATE-labeled lines (formatting only)."""
    date_lines = [r['text'] for r in results if r['label'] == 'DATE']
    if not date_lines:
        return 'N/A'
    
    full_text = ' '.join(date_lines)
    # Strip prefix labels
    full_text = re.sub(r'\b(waktu|tanggal|tgl|date|time)\s*:?\s*', '', full_text, flags=re.IGNORECASE).strip()
    
    patterns = [
        r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)',
        r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
        r'(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})',
        r'(\d{1,2}\s+\w{3,9}\s+\d{2,4}\s*\d{1,2}:\d{2}(?::\d{2})?)',
        r'(\d{1,2}\s+\w{3,9}\s+\d{2,4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return full_text


def parse_amount(text):
    """Parse monetary amount from text (formatting only)."""
    cleaned = re.sub(r'[a-zA-Z:_=]+', '', text).strip()
    
    # Find number patterns
    patterns = [
        r'([\d]{1,3}(?:[.,]\d{3})+)',  # 340,000 / 340.000 / 1.234.567
        r'([\d]+[.,][\d]+)',           # 12.50 / 21,60
        r'([\d]+)',                    # 340000
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            num_str = match.group(1)
            # Detect thousand separator vs decimal
            if '.' in num_str:
                last = num_str.split('.')[-1]
                if len(last) == 3:
                    num_str = num_str.replace('.', '')
            if ',' in num_str:
                last = num_str.split(',')[-1]
                if len(last) == 3:
                    num_str = num_str.replace(',', '')
                else:
                    num_str = num_str.replace(',', '.')
            try:
                val = float(num_str)
                if val > 0:
                    return val
            except ValueError:
                pass
    return None


def format_total(results):
    """Pick the most likely 'grand total' from TOTAL_PAYMENT lines.
    
    Strategy: Find line with 'Total' keyword (not 'Tunai/Cash' which is amount paid).
    Fall back to highest amount if no keyword match.
    """
    total_lines = [r for r in results if r['label'] == 'TOTAL_PAYMENT']
    if not total_lines:
        return {'display': 'N/A', 'amount': None, 'raw_lines': []}
    
    # Priority: actual total > payment amount
    # "Total" / "Grand Total" / "Total Bayar" = the bill amount
    # "Tunai" / "Cash" = money handed over (could be more)
    priority_keywords = [
        'total bayar', 'total tagihan', 'grand total', 'total belanja',
        'harga jual', 'subtotal',
        'total',  # generic, last priority before payment
        'pembayaran', 'tunai', 'non tunai', 'cash',
    ]
    
    raw_texts = [r['text'] for r in total_lines]
    
    for keyword in priority_keywords:
        for line in raw_texts:
            if keyword in line.lower() and 'sub' not in line.lower().split('total')[0] if 'total' in line.lower() else True:
                amount = parse_amount(line)
                if amount:
                    label = keyword.title()
                    return {
                        'display': f"{label}: Rp {amount:,.0f}".replace(',', '.'),
                        'amount': amount,
                        'raw_lines': raw_texts,
                    }
    
    # Fallback: largest number found
    amounts = [parse_amount(line) for line in raw_texts]
    amounts = [a for a in amounts if a is not None]
    if amounts:
        amount = max(amounts)
        return {
            'display': f"Total: Rp {amount:,.0f}".replace(',', '.'),
            'amount': amount,
            'raw_lines': raw_texts,
        }
    
    return {'display': ' | '.join(raw_texts[:3]), 'amount': None, 'raw_lines': raw_texts}


def annotate_image(img, results):
    """Draw bboxes on image."""
    label_colors = {
        'STORE': (255, 100, 100),
        'ADDRESS_CONTACT': (100, 200, 255),
        'DATE': (100, 255, 100),
        'ITEM_DESC': (255, 200, 100),
        'ITEM_PRICE/QTY': (200, 150, 255),
        'TOTAL_PAYMENT': (255, 50, 50),
        'OTHER': (180, 180, 180),
    }
    
    annotated = img.copy()
    for r in results:
        bbox = r['bbox']
        color = label_colors.get(r['label'], (180, 180, 180))
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(annotated, [pts], True, color, 2)
        x, y = int(bbox[0][0]), int(bbox[0][1]) - 5
        cv2.putText(annotated, r['label'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return annotated


def img_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    return render_template('index_v2.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Pipeline
    img = deskew(img)
    lines = ocr_image(img)
    
    if not lines:
        return jsonify({'error': 'No text detected'}), 400
    
    # Model classifies
    results = classify_lines(lines)
    
    # Merge horizontal fragments (formatting)
    merged = merge_horizontal_fragments(results)
    
    # Extract structured data (formatting only)
    store = format_store(merged)
    date = format_date(merged)
    items = [r['text'] for r in merged if r['label'] == 'ITEM_DESC']
    prices = [r['text'] for r in merged if r['label'] == 'ITEM_PRICE/QTY']
    total_info = format_total(merged)
    
    annotated = annotate_image(img, results)
    
    return jsonify({
        'success': True,
        'lines': results,
        'extracted': {
            'store': store,
            'date': date,
            'items': items,
            'prices': prices,
            'total': total_info['display'],
            'total_items': len(items),
        },
        'annotated_image': img_to_base64(annotated),
        'stats': {
            'total_lines': len(results),
            'avg_confidence': round(np.mean([r['confidence'] for r in results]) * 100, 1),
        }
    })


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'model': 'v2', 'ocr': 'easyocr_cpu'})


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 50)
    print("🧾 OCR FinSight - Receipt Scanner")
    print("=" * 50)
    init_model()
    init_ocr()
    print("\n🌐 Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)
