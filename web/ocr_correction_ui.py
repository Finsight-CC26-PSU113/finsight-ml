"""
OCR FinSight - OCR Correction UI
Web-based tool untuk manual correction OCR results + labeling.

Features:
- Show receipt image + OCR lines
- Edit OCR text (fix typos)
- Assign labels (STORE, DATE, TOTAL, ITEM, etc.)
- Save progress (resume anytime)
- Export to CSV for training

Usage:
    python web/ocr_correction_ui.py
    Open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import csv
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ROOT_DIR
from src.preprocessing import load_image, deskew
from src.ocr_engine import OCREngine
from src.online_learning import OnlineLearningModel, CorrectionTracker

app = Flask(__name__)

# Configuration
IMAGES_DIR = ROOT_DIR / "fullDataset" / "images"
PROGRESS_FILE = ROOT_DIR / "data" / "ocr_correction_progress.json"
OUTPUT_CSV = ROOT_DIR / "data" / "corrected_labeled_lines.csv"

# Global state
ocr_engine = None
online_model = None
correction_tracker = None
current_data = {
    'images': [],
    'current_index': 0,
    'corrections': {}  # {filename: {lines: [...], labels: [...], predicted_labels: [...]}}
}

# Label options
LABELS = [
    "STORE",
    "DATE",
    "TOTAL_PAYMENT",
    "ITEM_DESC",
    "ITEM_PRICE/QTY",
    "ADDRESS_CONTACT",
    "OTHER"
]


def init_ocr():
    """Initialize OCR engine."""
    global ocr_engine
    if ocr_engine is None:
        print("Initializing OCR engine...")
        ocr_engine = OCREngine()
        print("OCR engine ready!")


def init_online_learning():
    """Initialize online learning components (Model V2)."""
    global online_model, correction_tracker
    
    if online_model is None:
        print("Initializing Model V2 (Hierarchical BiLSTM + Attention)...")
        try:
            # Try Model V2 first
            from src.model_v2 import HierarchicalReceiptClassifier
            from src.model_v2_utils import (
                prepare_single_receipt, MAX_LINES_PER_RECEIPT,
                NUM_TEXT_FEATURES, NUM_POS_FEATURES
            )
            import tensorflow as tf
            import numpy as np
            
            model_v2 = HierarchicalReceiptClassifier()
            
            # Build model
            dummy_chars = tf.zeros((1, MAX_LINES_PER_RECEIPT, 100), dtype=tf.int32)
            dummy_text = tf.zeros((1, MAX_LINES_PER_RECEIPT, NUM_TEXT_FEATURES), dtype=tf.float32)
            dummy_pos = tf.zeros((1, MAX_LINES_PER_RECEIPT, NUM_POS_FEATURES), dtype=tf.float32)
            dummy_mask = tf.ones((1, MAX_LINES_PER_RECEIPT), dtype=tf.float32)
            _ = model_v2((dummy_chars, dummy_text, dummy_pos, dummy_mask), training=False)
            
            # Load weights
            weights_path = ROOT_DIR / "models" / "v2" / "best_weights.weights.h5"
            model_v2.load_weights(str(weights_path))
            
            online_model = model_v2
            print("✅ Model V2 loaded! (88.8% accuracy)")
            print(f"   Architecture: Hierarchical BiLSTM + Self-Attention")
            print(f"   Weights: {weights_path}")
            
        except Exception as e:
            print(f"⚠️  Model V2 failed: {e}")
            print("   Falling back to Model V1...")
            try:
                online_model = OnlineLearningModel(buffer_size=50)
                print("✅ Model V1 (online learning) enabled!")
            except Exception as e2:
                print(f"⚠️  All models disabled: {e2}")
                online_model = None
        
        # Init correction tracker regardless
        try:
            correction_tracker = CorrectionTracker()
        except:
            correction_tracker = None


def load_progress():
    """Load saved progress."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            current_data['current_index'] = data.get('current_index', 0)
            current_data['corrections'] = data.get('corrections', {})
            print(f"Loaded progress: {len(current_data['corrections'])} images corrected")


def save_progress():
    """Save current progress."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'current_index': current_data['current_index'],
            'corrections': current_data['corrections']
        }, f, indent=2, ensure_ascii=False)


def load_images():
    """Load list of images."""
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    current_data['images'] = [img.name for img in images]
    print(f"Found {len(current_data['images'])} images")


def smart_label_lines(lines: list, total_lines: int) -> list:
    """
    Context-aware smart auto-labeling for ALL lines at once.
    Uses zone detection, keyword matching, and neighbor context.
    
    This is much better than labeling line-by-line because it can:
    - Detect zones (header/items/totals)
    - Use neighbor context (RM next to total = TOTAL_PAYMENT)
    - Handle split tokens (e.g., "RM" and "21.60" on separate boxes)
    """
    import re
    
    if not lines:
        return []
    
    labels = ['OTHER'] * total_lines
    
    # ===== PHASE 1: Zone Detection =====
    # Divide receipt into zones based on relative position
    # Zone A: Header (top 20%) - STORE, ADDRESS_CONTACT
    # Zone B: Middle (20%-75%) - ITEM_DESC, ITEM_PRICE/QTY
    # Zone C: Footer (75%-100%) - TOTAL_PAYMENT, OTHER
    
    # Find the "total separator" - first line with total/subtotal keyword
    total_start_idx = total_lines  # default: no total section found
    for i, line in enumerate(lines):
        text_lower = line.get('text', '').lower().strip()
        if re.search(r'\b(sub\s*total|total|grand\s*total)\b', text_lower):
            total_start_idx = i
            break
    
    # ===== PHASE 2: Keyword-based labeling (highest priority) =====
    for i, line in enumerate(lines):
        text = line.get('text', '')
        text_lower = text.lower().strip()
        text_stripped = text.strip()
        relative_pos = i / max(total_lines, 1)
        
        # --- DATE detection (highest priority) ---
        date_patterns = [
            r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',
            r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}',
            r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2}\b',
        ]
        # Also check for date keywords
        date_keywords = ['date', 'tanggal', 'tgl', 'bill start', 'bill end', 'closed bill']
        
        has_date_pattern = any(re.search(p, text) for p in date_patterns)
        has_date_keyword = any(kw in text_lower for kw in date_keywords)
        
        if has_date_pattern or has_date_keyword:
            # Make sure it's not a transaction ID (too many digits without separator)
            if not re.match(r'^\d{7,}$', text_stripped):
                labels[i] = 'DATE'
                continue
        
        # --- TOTAL_PAYMENT detection ---
        total_keywords = [
            'total', 'subtotal', 'sub total', 'grand total',
            'tunai', 'bayar', 'kembali', 'change',
            'amount', 'amt', 'nett', 'net',
            'rounding', 'round', 'adjustment',
            'discount', 'diskon', 'diskaun',
            'tax', 'gst', 'pajak', 'ppn',
            'service charge', 'inclusive',
            'parking fee',
        ]
        # "cash" only counts as TOTAL if in bottom half or near total section
        cash_keywords = ['cash', 'tunai']
        
        is_total_keyword = any(kw in text_lower for kw in total_keywords)
        is_cash_keyword = any(kw in text_lower for kw in cash_keywords)
        
        # Cash in bottom section = TOTAL_PAYMENT, cash in top = OTHER (e.g., "Cash Receipt")
        if is_total_keyword or (is_cash_keyword and (relative_pos > 0.5 or i >= total_start_idx)):
            # Exclude "cashier" which is not a payment
            if 'cashier' not in text_lower and 'kasir' not in text_lower:
                labels[i] = 'TOTAL_PAYMENT'
                continue
        
        # --- STORE detection (top section) ---
        company_suffixes = ['sdn bhd', 'sdn. bhd', 'sdn', 'bhd', 'pt.', 'pt ', 'cv.', 'cv ',
                           'ltd', 'inc', 'corp', 'co.', 'enterprise', 'trading', 'marketing']
        
        if relative_pos < 0.2:
            has_company_suffix = any(s in text_lower for s in company_suffixes)
            
            # Top 3 lines with mostly uppercase = likely STORE
            if i <= 2:
                alpha_count = sum(c.isalpha() for c in text)
                upper_count = sum(c.isupper() for c in text)
                if alpha_count > 2 and upper_count / alpha_count > 0.5:
                    labels[i] = 'STORE'
                    continue
            
            # Company suffix anywhere in top 20% = STORE
            if has_company_suffix:
                labels[i] = 'STORE'
                continue
        
        # --- ADDRESS_CONTACT detection ---
        address_keywords = [
            'jl.', 'jl ', 'jalan', 'jalah', 'jalax',  # include OCR typos
            'lorong', 'taman', 'bandar', 'masai',
            'kec.', 'kec ', 'kel.', 'kel ', 'desa',
            'telp', 'tel:', 'tel ', 'phone', 'fax', 'fax:',
            'email', 'website', 'www.',
            'npwp', 'gst no', 'gst reg', 'roc no', 'roc nu',
            'plaza', 'heights', 'damansara', 'selangor', 'johor',
            'kuala lumpur', 'malaysia', 'indonesia',
            'perhas', 'berkelEy',  # OCR typos for place names
        ]
        # Postal code pattern (5 digits alone or at start)
        is_postal = re.match(r'^\d{5}$', text_stripped)
        
        if any(kw in text_lower for kw in address_keywords) or is_postal:
            # Only if in top 30% (addresses are usually at the top)
            if relative_pos < 0.35:
                labels[i] = 'ADDRESS_CONTACT'
                continue
        
        # --- In TOTAL zone (after total_start_idx) ---
        if i >= total_start_idx:
            # Currency symbols or standalone numbers in total zone = TOTAL_PAYMENT
            currency_patterns = [
                r'^r[mh]\s*[\d.,]+',       # RM 21.60 or RH (OCR typo)
                r'^rp\.?\s*[\d.,]+',        # Rp 50.000
                r'^\$\s*[\d.,]+',           # $10.00
                r'^[\d.,]+\s*(sr|rm|rp)',   # 21.60 RM
                r'^[\d]+[.,]\s*\d+$',       # 21.60 (standalone number in total zone)
            ]
            if any(re.search(p, text_lower) for p in currency_patterns):
                labels[i] = 'TOTAL_PAYMENT'
                continue
            
            # Standalone "RM", "Rp", "RI" (OCR typo for RM) in total zone
            if text_stripped.upper() in ['RM', 'RP', 'RI', 'RH', 'R']:
                labels[i] = 'TOTAL_PAYMENT'
                continue
            
            # Pure numbers in total zone (likely split price)
            if re.match(r'^[\d.,\s]+$', text_stripped) and len(text_stripped) >= 2:
                labels[i] = 'TOTAL_PAYMENT'
                continue
        
        # --- ITEM_PRICE/QTY detection (middle section) ---
        if i < total_start_idx:
            # Skip if it looks like a transaction ID (long digits with x in middle)
            is_transaction_id = re.match(r'^[0-9]{5,}[xX][0-9]+$', text_stripped)
            
            if not is_transaction_id:
                price_patterns = [
                    r'\d+\s*[xX\*@]\s*[\d.,]+',   # 2 x 5.90, 1*4.90
                    r'[xX\*@]\s*[\d.,]+',          # x 5.90
                    r'^r[mh]\s*[\d.,]+',            # RM4.90
                    r'^rp\.?\s*[\d.,]+',            # Rp15.000
                    r'=\s*r[mh]\s*[\d.,]+',         # =RM 90.57
                    r'\d+[.,]\d+\s*(uni|unit|pcs)',  # 45.29 UNI
                ]
                if any(re.search(p, text_lower) for p in price_patterns):
                    labels[i] = 'ITEM_PRICE/QTY'
                    continue
            
            # Standalone number in item zone (likely price if short)
            if re.match(r'^[\d.,\s]+$', text_stripped):
                # Short numbers (< 10 chars) next to items = price
                if len(text_stripped) <= 10 and '.' in text_stripped:
                    labels[i] = 'ITEM_PRICE/QTY'
                    continue
        
        # --- ITEM_DESC detection (middle section, has letters) ---
        if i < total_start_idx and relative_pos > 0.15:
            alpha_count = sum(c.isalpha() for c in text)
            if alpha_count >= 3 and len(text_stripped) >= 3:
                # Not a currency symbol alone or OCR typo of RM
                currency_typos = ['RM', 'RP', 'RI', 'RH', 'R', 'SR', 'NH']
                if text_stripped.upper() not in currency_typos:
                    labels[i] = 'ITEM_DESC'
                    continue
    
    # ===== PHASE 3: Context propagation =====
    # If "RM" or "Rp" is standalone and next to a number, they share the same label
    for i in range(total_lines - 1):
        text_i = lines[i].get('text', '').strip().upper()
        text_next = lines[i + 1].get('text', '').strip()
        
        # "RM" followed by number → both get same label
        if text_i in ['RM', 'RP', 'RI', 'RH'] and re.match(r'^[\d.,]+$', text_next):
            # Determine label based on zone
            if i >= total_start_idx:
                labels[i] = 'TOTAL_PAYMENT'
                labels[i + 1] = 'TOTAL_PAYMENT'
            elif labels[i] == 'OTHER':
                # In item zone, RM + number = ITEM_PRICE/QTY
                labels[i] = 'ITEM_PRICE/QTY'
                labels[i + 1] = 'ITEM_PRICE/QTY'
    
    # ===== PHASE 4: Neighbor smoothing =====
    # If a line is OTHER but surrounded by ADDRESS_CONTACT in top zone, make it ADDRESS
    for i in range(1, total_lines - 1):
        if labels[i] == 'OTHER':
            relative_pos = i / max(total_lines, 1)
            if relative_pos < 0.25:
                # Check neighbors
                if labels[i-1] == 'ADDRESS_CONTACT' and labels[i+1] == 'ADDRESS_CONTACT':
                    labels[i] = 'ADDRESS_CONTACT'
            elif relative_pos < 0.15:
                # Top section: OTHER between STORE = likely STORE
                if labels[i-1] == 'STORE' and labels[i+1] == 'STORE':
                    labels[i] = 'STORE'
    
    # ===== PHASE 5: Footer detection =====
    footer_keywords = ['thank', 'terima kasih', 'forward', 'visit', 'welcome', 
                       'goods sold', 'no refund', 'no exchange', 'copy']
    for i in range(total_lines):
        text_lower = lines[i].get('text', '').lower()
        if any(kw in text_lower for kw in footer_keywords):
            labels[i] = 'OTHER'
    
    return labels


def smart_label_line(text: str, line_index: int, total_lines: int) -> str:
    """Legacy single-line labeling (fallback). Use smart_label_lines() instead."""
    import re
    
    text_lower = text.lower()
    
    # DATE detection
    date_patterns = [
        r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',
        r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}',
    ]
    if any(re.search(p, text) for p in date_patterns):
        return 'DATE'
    
    # TOTAL_PAYMENT detection
    total_keywords = ['total', 'subtotal', 'tunai', 'cash', 'bayar', 'kembali', 'change', 'grand total']
    if any(k in text_lower for k in total_keywords):
        return 'TOTAL_PAYMENT'
    
    # STORE detection (top 3 lines with uppercase)
    if line_index <= 2:
        upper_count = sum(c.isupper() for c in text)
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count > 0 and upper_count / alpha_count > 0.5:
            return 'STORE'
    
    # ADDRESS_CONTACT detection
    address_keywords = ['jl.', 'jalan', 'kec.', 'kel.', 'telp', 'tel', 'phone', 'email', 'website', 'npwp', 'gst', 'roc']
    if any(k in text_lower for k in address_keywords):
        return 'ADDRESS_CONTACT'
    
    # ITEM_PRICE/QTY detection
    price_patterns = [
        r'\d+\s*[xX\*]\s*[\d.,]+',
        r'@\s*[\d.,]+',
        r'^[\d.,]+$',
    ]
    if any(re.search(p, text) for p in price_patterns):
        return 'ITEM_PRICE/QTY'
    
    # ITEM_DESC detection (middle section with letters)
    if 3 <= line_index <= (total_lines - 5):
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count >= 5 and len(text) >= 5:
            return 'ITEM_DESC'
    
    return 'OTHER'


def ocr_image(filename):
    """OCR an image and return lines with smart labels."""
    image_path = IMAGES_DIR / filename
    
    # Load and preprocess
    img = load_image(image_path)
    img = deskew(img)
    
    # OCR
    lines, img_info = ocr_engine.read_receipt_with_image_info(img)
    
    return lines, img_info


@app.route('/')
def index():
    """Main page."""
    return render_template('ocr_correction.html')


@app.route('/api/init', methods=['GET'])
def api_init():
    """Initialize and get first image."""
    init_ocr()
    init_online_learning()  # Initialize online learning
    load_images()
    load_progress()
    
    # Get online learning status
    online_learning_status = {
        'enabled': online_model is not None,
        'buffer_status': online_model.get_buffer_status() if online_model and hasattr(online_model, 'get_buffer_status') else None,
        'total_corrections': len(correction_tracker.corrections) if correction_tracker else 0
    }
    
    return jsonify({
        'total_images': len(current_data['images']),
        'current_index': current_data['current_index'],
        'corrected_count': len(current_data['corrections']),
        'labels': LABELS,
        'online_learning': online_learning_status
    })


@app.route('/api/image/<int:index>', methods=['GET'])
def api_get_image(index):
    """Get image data and OCR lines with ML predictions or smart labels."""
    # Ensure images are loaded
    if not current_data['images']:
        load_images()
    
    if index < 0 or index >= len(current_data['images']):
        return jsonify({'error': f'Invalid index: {index} (total images: {len(current_data["images"])})'}), 400
    
    filename = current_data['images'][index]
    
    # Check if already corrected
    if filename in current_data['corrections']:
        # Return saved corrections
        return jsonify({
            'filename': filename,
            'index': index,
            'total': len(current_data['images']),
            'lines': current_data['corrections'][filename]['lines'],
            'labels': current_data['corrections'][filename]['labels'],
            'predicted_labels': current_data['corrections'][filename].get('predicted_labels', []),
            'is_corrected': True
        })
    
    # OCR the image
    try:
        lines, img_info = ocr_image(filename)
        
        # Format lines
        formatted_lines = [
            {
                'index': i,
                'text': line['text'],
                'confidence': round(line['confidence'], 2),
                'bbox': {
                    'x_min': line['x_min'],
                    'y_min': line['y_min'],
                    'x_max': line['x_max'],
                    'y_max': line['y_max']
                },
                'y_center': line.get('y_center', 0.5),
                'x_center': line.get('x_center', 0.5),
                'width': line.get('width', 0.1),
                'height': line.get('height', 0.05),
                'line_index': i
            }
            for i, line in enumerate(lines)
        ]
        
        # Try ML prediction first (Model V2 or V1)
        predicted_labels = []
        ml_confidences = []
        prediction_method = 'heuristic'
        
        if online_model:
            try:
                # Check if it's Model V2 (HierarchicalReceiptClassifier)
                from src.model_v2 import HierarchicalReceiptClassifier
                if isinstance(online_model, HierarchicalReceiptClassifier):
                    # Model V2: predict all lines at once
                    from src.model_v2_utils import prepare_single_receipt
                    import tensorflow as tf
                    import numpy as np
                    
                    char_ids, text_feats, pos_feats, mask = prepare_single_receipt(formatted_lines)
                    inputs = (
                        tf.expand_dims(char_ids, 0),
                        tf.expand_dims(text_feats, 0),
                        tf.expand_dims(pos_feats, 0),
                        tf.expand_dims(mask, 0),
                    )
                    predictions = online_model(inputs, training=False)
                    predictions = predictions[0].numpy()
                    
                    from src.config import LINE_CLASSES
                    for i in range(len(formatted_lines)):
                        pred = predictions[i]
                        idx = int(np.argmax(pred))
                        confidence = float(pred[idx])
                        predicted_labels.append(LINE_CLASSES[idx])
                        ml_confidences.append(confidence)
                    
                    prediction_method = 'model_v2'
                    print(f"[Model V2] Predicted {len(predicted_labels)} labels (Hierarchical BiLSTM)")
                else:
                    # Model V1: online learning model
                    predicted_labels, ml_confidences = online_model.predict(formatted_lines)
                    prediction_method = 'model_v1'
                    print(f"[Model V1] Predicted {len(predicted_labels)} labels")
            except Exception as e:
                print(f"[ML] Prediction failed: {e}")
                predicted_labels = []
        
        # Fallback to smart heuristic labeling (context-aware)
        if not predicted_labels:
            predicted_labels = smart_label_lines(formatted_lines, len(formatted_lines))
            ml_confidences = [0.85] * len(predicted_labels)
            prediction_method = 'heuristic'
            print(f"[SmartLabel] Predicted {len(predicted_labels)} labels with context-aware heuristics")
        
        return jsonify({
            'filename': filename,
            'index': index,
            'total': len(current_data['images']),
            'lines': formatted_lines,
            'labels': predicted_labels,
            'predicted_labels': predicted_labels,
            'ml_confidences': ml_confidences,
            'is_corrected': False,
            'prediction_method': prediction_method
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/save', methods=['POST'])
def api_save():
    """Save corrections and trigger online learning."""
    data = request.json
    
    filename = data.get('filename')
    lines = data.get('lines')
    labels = data.get('labels')
    predicted_labels = data.get('predicted_labels', [])
    
    if not filename or not lines or not labels:
        return jsonify({'error': 'Missing data'}), 400
    
    # Find corrections (where predicted != corrected)
    corrections_made = []
    
    if predicted_labels and online_model and correction_tracker:
        for i, (line, pred_label, correct_label) in enumerate(zip(lines, predicted_labels, labels)):
            if pred_label != correct_label:
                # This is a correction!
                correction = {
                    'line_index': i,
                    'text': line['text'],
                    'predicted': pred_label,
                    'correct': correct_label,
                    'confidence': line.get('confidence', 0.5)
                }
                corrections_made.append(correction)
                
                # Add to online learning
                online_model.add_correction(line, correct_label)
                
                # Track correction
                correction_tracker.add_correction(correction)
        
        if corrections_made:
            print(f"[OnlineLearning] 📚 Learned from {len(corrections_made)} corrections")
    
    # Save to corrections
    current_data['corrections'][filename] = {
        'lines': lines,
        'labels': labels,
        'predicted_labels': predicted_labels
    }
    
    # Update current index
    current_data['current_index'] = data.get('index', 0)
    
    # Save progress
    save_progress()
    
    # Get updated stats
    buffer_status = online_model.get_buffer_status() if online_model and hasattr(online_model, 'get_buffer_status') else None
    correction_stats = correction_tracker.get_stats() if correction_tracker else None
    
    return jsonify({
        'success': True,
        'corrected_count': len(current_data['corrections']),
        'corrections_made': len(corrections_made),
        'online_learning': {
            'buffer_status': buffer_status,
            'correction_stats': correction_stats
        }
    })


@app.route('/api/export', methods=['GET'])
def api_export():
    """Export corrections to CSV."""
    if not current_data['corrections']:
        return jsonify({'error': 'No corrections to export'}), 400
    
    # Prepare CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    csv_fields = [
        'filename', 'line_index', 'text', 'confidence',
        'x_min', 'y_min', 'x_max', 'y_max',
        'label'
    ]
    
    total_lines = 0
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        
        for filename, data in current_data['corrections'].items():
            lines = data['lines']
            labels = data['labels']
            
            for i, (line, label) in enumerate(zip(lines, labels)):
                row = {
                    'filename': filename,
                    'line_index': i,
                    'text': line['text'],
                    'confidence': line.get('confidence', 1.0),
                    'x_min': line['bbox']['x_min'],
                    'y_min': line['bbox']['y_min'],
                    'x_max': line['bbox']['x_max'],
                    'y_max': line['bbox']['y_max'],
                    'label': label
                }
                writer.writerow(row)
                total_lines += 1
    
    return jsonify({
        'success': True,
        'output_file': str(OUTPUT_CSV),
        'total_lines': total_lines,
        'total_images': len(current_data['corrections'])
    })


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get statistics including learning performance."""
    label_counts = {}
    total_lines = 0
    
    for data in current_data['corrections'].values():
        for label in data['labels']:
            label_counts[label] = label_counts.get(label, 0) + 1
            total_lines += 1
    
    # Get learning performance stats
    learning_performance = None
    
    if correction_tracker:
        total_corrections = len(correction_tracker.corrections)
        buffer_size = online_model.buffer_size if online_model else 10
        model_retrains = total_corrections // buffer_size
        current_buffer = total_corrections % buffer_size
        
        # Estimate accuracy improvement
        # Rough formula: +0.5% per 50 corrections, max 97%
        initial_accuracy = 92.95
        corrections_per_percent = 50
        estimated_improvement = (total_corrections / corrections_per_percent) * 0.5
        estimated_accuracy = min(initial_accuracy + estimated_improvement, 97.0)
        
        learning_performance = {
            'total_corrections': total_corrections,
            'model_retrains': model_retrains,
            'buffer_status': f'{current_buffer}/{buffer_size}',
            'buffer_fill_percent': (current_buffer / buffer_size) * 100,
            'estimated_accuracy': estimated_accuracy,
            'accuracy_improvement': estimated_improvement,
            'initial_accuracy': initial_accuracy
        }
    
    # Get auto-evaluation summary
    auto_eval_summary = None
    auto_eval_alerts = []
    
    try:
        from src.auto_evaluation import get_auto_evaluator
        evaluator = get_auto_evaluator()
        auto_eval_summary = evaluator.get_summary()
        auto_eval_alerts = evaluator.get_alerts()
    except Exception as e:
        print(f"[API] Auto-eval error: {e}")
    
    return jsonify({
        'total_images': len(current_data['images']),
        'corrected_images': len(current_data['corrections']),
        'total_lines': total_lines,
        'label_distribution': label_counts,
        'progress_percent': round(len(current_data['corrections']) / max(len(current_data['images']), 1) * 100, 1),
        'learning_performance': learning_performance,
        'auto_evaluation': auto_eval_summary,
        'alerts': auto_eval_alerts
    })


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve image file."""
    return send_from_directory(IMAGES_DIR, filename)


if __name__ == '__main__':
    print("=" * 60)
    print("OCR FinSight - OCR Correction UI")
    print("=" * 60)
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Progress file: {PROGRESS_FILE}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print()
    print("Starting server...")
    print("Open: http://localhost:5000")
    print("=" * 60)
    
    # Disable debug mode to prevent auto-reload losing data
    app.run(debug=False, host='0.0.0.0', port=5000)
