"""
Utility functions for Model V2 (Hierarchical BiLSTM + Attention).
Data preparation, feature extraction, training helpers.
"""

import numpy as np
import re
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

from src.config import ROOT_DIR, MAX_TEXT_LENGTH, NUM_CLASSES, LINE_CLASSES, RANDOM_SEED

# Max lines per receipt (padded to this length)
MAX_LINES_PER_RECEIPT = 40

# Number of text features per line
NUM_TEXT_FEATURES = 15  # Extended from 10 to 15 (added keyword features)

# Number of position features per line
NUM_POS_FEATURES = 5


# ==============================================================================
# Feature Extraction
# ==============================================================================

def text_to_char_ids(text: str, max_length: int = MAX_TEXT_LENGTH) -> np.ndarray:
    """Convert text to ASCII character IDs."""
    ids = [min(ord(c), 127) for c in text[:max_length]]
    ids = ids + [0] * (max_length - len(ids))
    return np.array(ids, dtype=np.int32)


def extract_text_features(text: str) -> np.ndarray:
    """Extract 15 text features from a line.
    
    Features (15 total):
        0: char_count (normalized)
        1: word_count (normalized)
        2: digit_ratio
        3: alpha_ratio
        4: upper_ratio (of alpha chars)
        5: space_ratio
        6: special_ratio
        7: has_currency (RM, Rp, $)
        8: has_date_pattern (DD/MM/YY)
        9: has_qty_pattern (2 x 5.90)
        10: has_total_keyword (total, cash, change, subtotal)
        11: has_address_keyword (jl., tel, kec., gst)
        12: has_store_keyword (sdn, bhd, pt., cv.)
        13: is_mostly_digits (>70% digits)
        14: text_length_bucket (0=short, 0.5=medium, 1=long)
    """
    text_lower = text.lower()
    char_count = len(text)
    total_chars = max(char_count, 1)
    
    digit_count = sum(c.isdigit() for c in text)
    alpha_count = sum(c.isalpha() for c in text)
    upper_count = sum(c.isupper() for c in text)
    space_count = sum(c.isspace() for c in text)
    special_count = sum(not c.isalnum() and not c.isspace() for c in text)
    
    # Keyword detection
    total_keywords = ['total', 'subtotal', 'cash', 'tunai', 'change', 'kembali', 
                      'bayar', 'amount', 'amt', 'discount', 'diskon', 'tax', 'gst',
                      'rounding', 'service charge', 'nett']
    address_keywords = ['jl.', 'jl ', 'jalan', 'tel', 'telp', 'fax', 'email',
                       'kec.', 'kel.', 'npwp', 'gst no', 'roc', 'www.']
    store_keywords = ['sdn', 'bhd', 'pt.', 'pt ', 'cv.', 'cv ', 'ltd', 'inc',
                     'corp', 'enterprise', 'trading', 'marketing']
    
    has_total_kw = 1 if any(kw in text_lower for kw in total_keywords) else 0
    has_address_kw = 1 if any(kw in text_lower for kw in address_keywords) else 0
    has_store_kw = 1 if any(kw in text_lower for kw in store_keywords) else 0
    
    features = np.array([
        char_count / 100.0,  # normalized
        len(text.split()) / 20.0,  # normalized
        digit_count / total_chars,
        alpha_count / total_chars,
        upper_count / max(alpha_count, 1),
        space_count / total_chars,
        special_count / total_chars,
        1 if re.search(r'(rm|rp|r[mh]|\$|€)', text_lower) else 0,
        1 if re.search(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', text) else 0,
        1 if re.search(r'\d+\s*[xX\*@]\s*[\d.,]+', text) else 0,
        has_total_kw,
        has_address_kw,
        has_store_kw,
        1 if digit_count / total_chars > 0.7 else 0,
        min(char_count / 50.0, 1.0),  # length bucket
    ], dtype=np.float32)
    
    return features


def extract_position_features(line: Dict, total_lines: int, line_index: int) -> np.ndarray:
    """Extract 5 position features.
    
    Features:
        0: y_center (vertical position, 0=top, 1=bottom)
        1: x_center (horizontal position)
        2: width (normalized)
        3: height (normalized)
        4: line_index_ratio (0=first line, 1=last line)
    """
    return np.array([
        line.get('y_center', 0.5),
        line.get('x_center', 0.5),
        line.get('width', 0.1),
        line.get('height', 0.05),
        line_index / max(total_lines - 1, 1),
    ], dtype=np.float32)


# ==============================================================================
# Data Preparation
# ==============================================================================

def prepare_single_receipt(lines: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare a single receipt for model input.
    
    Args:
        lines: List of line dicts
        
    Returns:
        (char_ids, text_features, pos_features, mask)
        All padded to MAX_LINES_PER_RECEIPT
    """
    n_lines = min(len(lines), MAX_LINES_PER_RECEIPT)
    
    char_ids = np.zeros((MAX_LINES_PER_RECEIPT, MAX_TEXT_LENGTH), dtype=np.int32)
    text_features = np.zeros((MAX_LINES_PER_RECEIPT, NUM_TEXT_FEATURES), dtype=np.float32)
    pos_features = np.zeros((MAX_LINES_PER_RECEIPT, NUM_POS_FEATURES), dtype=np.float32)
    mask = np.zeros(MAX_LINES_PER_RECEIPT, dtype=np.float32)
    
    for i in range(n_lines):
        line = lines[i]
        text = line.get('text', '')
        
        char_ids[i] = text_to_char_ids(text)
        text_features[i] = extract_text_features(text)
        pos_features[i] = extract_position_features(line, len(lines), i)
        mask[i] = 1.0
    
    return char_ids, text_features, pos_features, mask


def load_training_data_v2(csv_path: Path = None) -> Tuple:
    """Load labeled CSV and prepare as receipt-level training data.
    
    Groups lines by filename (receipt), pads to MAX_LINES_PER_RECEIPT.
    
    Returns:
        (train_data, val_data) where each is:
        ((char_ids, text_feats, pos_feats, mask), labels)
    """
    if csv_path is None:
        csv_path = ROOT_DIR / "data" / "labeled_lines_v2.csv"
    
    print(f"[Data V2] Loading from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[Data V2] Total lines: {len(df)}")
    print(f"[Data V2] Total receipts: {df['filename'].nunique()}")
    
    # Label mapping
    label_to_idx = {v: k for k, v in LINE_CLASSES.items()}
    
    # Group by receipt
    receipts = []
    for filename, group in df.groupby('filename'):
        group = group.sort_values('line_index')
        
        lines = []
        labels = []
        
        for _, row in group.iterrows():
            line = {
                'text': str(row.get('text', '')),
                'y_center': row.get('y_center', row.get('y_ratio', 0.5)),
                'x_center': row.get('x_center', row.get('x_ratio', 0.5)),
                'width': row.get('width', row.get('width_ratio', 0.1)),
                'height': row.get('height', row.get('height_ratio', 0.05)),
            }
            lines.append(line)
            
            label_str = row.get('label', 'OTHER')
            label_idx = label_to_idx.get(label_str, 6)  # 6 = OTHER
            labels.append(label_idx)
        
        receipts.append({
            'filename': filename,
            'lines': lines,
            'labels': labels,
        })
    
    print(f"[Data V2] Loaded {len(receipts)} receipts")
    
    # Print label distribution
    all_labels = []
    for r in receipts:
        all_labels.extend(r['labels'])
    
    print("[Data V2] Label distribution:")
    for idx, name in LINE_CLASSES.items():
        count = all_labels.count(idx)
        print(f"  {name:>15s}: {count:>5d} ({count/len(all_labels)*100:.1f}%)")
    
    # Prepare arrays
    n_receipts = len(receipts)
    
    all_char_ids = np.zeros((n_receipts, MAX_LINES_PER_RECEIPT, MAX_TEXT_LENGTH), dtype=np.int32)
    all_text_feats = np.zeros((n_receipts, MAX_LINES_PER_RECEIPT, NUM_TEXT_FEATURES), dtype=np.float32)
    all_pos_feats = np.zeros((n_receipts, MAX_LINES_PER_RECEIPT, NUM_POS_FEATURES), dtype=np.float32)
    all_masks = np.zeros((n_receipts, MAX_LINES_PER_RECEIPT), dtype=np.float32)
    all_labels = np.zeros((n_receipts, MAX_LINES_PER_RECEIPT, NUM_CLASSES), dtype=np.float32)
    
    for r_idx, receipt in enumerate(receipts):
        n_lines = min(len(receipt['lines']), MAX_LINES_PER_RECEIPT)
        
        for i in range(n_lines):
            line = receipt['lines'][i]
            text = line.get('text', '')
            
            all_char_ids[r_idx, i] = text_to_char_ids(text)
            all_text_feats[r_idx, i] = extract_text_features(text)
            all_pos_feats[r_idx, i] = extract_position_features(line, n_lines, i)
            all_masks[r_idx, i] = 1.0
            
            label_idx = receipt['labels'][i]
            all_labels[r_idx, i, label_idx] = 1.0
    
    # Split train/val (80/20)
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(n_receipts)
    split_idx = int(0.8 * n_receipts)
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    train_data = (
        (all_char_ids[train_idx], all_text_feats[train_idx], 
         all_pos_feats[train_idx], all_masks[train_idx]),
        all_labels[train_idx],
        all_masks[train_idx],
    )
    
    val_data = (
        (all_char_ids[val_idx], all_text_feats[val_idx],
         all_pos_feats[val_idx], all_masks[val_idx]),
        all_labels[val_idx],
        all_masks[val_idx],
    )
    
    print(f"[Data V2] Train: {len(train_idx)} receipts, Val: {len(val_idx)} receipts")
    
    return train_data, val_data


# ==============================================================================
# Also support loading from JSON (auto_labeled_receipts.json)
# ==============================================================================

def load_training_data_from_json(json_path: Path = None) -> Tuple:
    """Load from auto-labeled JSON and prepare training data."""
    if json_path is None:
        json_path = ROOT_DIR / "data" / "auto_labeled_receipts.json"
    
    # Also try progress file
    if not json_path.exists():
        json_path = ROOT_DIR / "data" / "auto_label_progress.json"
    
    print(f"[Data V2] Loading from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', data)  # Handle both formats
    
    label_to_idx = {v: k for k, v in LINE_CLASSES.items()}
    
    receipts = []
    for filename, receipt_data in results.items():
        lines = receipt_data.get('lines', [])
        labels = receipt_data.get('labels', [])
        
        if not lines or not labels:
            continue
        
        label_indices = []
        for label_str in labels:
            label_indices.append(label_to_idx.get(label_str, 6))
        
        receipts.append({
            'filename': filename,
            'lines': lines,
            'labels': label_indices,
        })
    
    print(f"[Data V2] Loaded {len(receipts)} receipts from JSON")
    
    # Same preparation as above
    n_receipts = len(receipts)
    
    all_char_ids = np.zeros((n_receipts, MAX_LINES_PER_RECEIPT, MAX_TEXT_LENGTH), dtype=np.int32)
    all_text_feats = np.zeros((n_receipts, MAX_LINES_PER_RECEIPT, NUM_TEXT_FEATURES), dtype=np.float32)
    all_pos_feats = np.zeros((n_receipts, MAX_LINES_PER_RECEIPT, NUM_POS_FEATURES), dtype=np.float32)
    all_masks = np.zeros((n_receipts, MAX_LINES_PER_RECEIPT), dtype=np.float32)
    all_labels_arr = np.zeros((n_receipts, MAX_LINES_PER_RECEIPT, NUM_CLASSES), dtype=np.float32)
    
    for r_idx, receipt in enumerate(receipts):
        n_lines = min(len(receipt['lines']), MAX_LINES_PER_RECEIPT)
        
        for i in range(n_lines):
            line = receipt['lines'][i]
            text = line.get('text', '')
            
            all_char_ids[r_idx, i] = text_to_char_ids(text)
            all_text_feats[r_idx, i] = extract_text_features(text)
            all_pos_feats[r_idx, i] = extract_position_features(line, n_lines, i)
            all_masks[r_idx, i] = 1.0
            
            label_idx = receipt['labels'][i]
            all_labels_arr[r_idx, i, label_idx] = 1.0
    
    # Split
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(n_receipts)
    split_idx = int(0.8 * n_receipts)
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    train_data = (
        (all_char_ids[train_idx], all_text_feats[train_idx],
         all_pos_feats[train_idx], all_masks[train_idx]),
        all_labels_arr[train_idx],
        all_masks[train_idx],
    )
    
    val_data = (
        (all_char_ids[val_idx], all_text_feats[val_idx],
         all_pos_feats[val_idx], all_masks[val_idx]),
        all_labels_arr[val_idx],
        all_masks[val_idx],
    )
    
    print(f"[Data V2] Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    return train_data, val_data


import json
