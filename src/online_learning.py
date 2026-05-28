"""
OCR FinSight - Online Learning System
Model learns from user corrections in real-time.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import tensorflow as tf

from src.config import ROOT_DIR, LINE_CLASSES, MAX_TEXT_LENGTH


def text_to_char_ids(text: str, max_length: int = MAX_TEXT_LENGTH) -> np.ndarray:
    """Convert text to array of ASCII character IDs."""
    ids = [min(ord(c), 127) for c in text[:max_length]]
    ids = ids + [0] * (max_length - len(ids))
    return np.array(ids, dtype=np.int32)


def extract_features_for_line(line: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for a single line (compatible with model).
    
    Args:
        line: Line dict with 'text', 'x_min', 'y_min', etc.
        
    Returns:
        Tuple of (char_ids, text_features, position_features)
    """
    text = line.get('text', '')
    
    # Char IDs
    char_ids = text_to_char_ids(text)
    
    # Text features (10 features)
    text_features = np.array([
        len(text),  # length
        sum(c.isupper() for c in text),  # uppercase count
        sum(c.islower() for c in text),  # lowercase count
        sum(c.isdigit() for c in text),  # digit count
        sum(c.isspace() for c in text),  # space count
        sum(c in '.,;:!?' for c in text),  # punctuation count
        1 if any(c.isdigit() for c in text) else 0,  # has digit
        1 if any(c.isupper() for c in text) else 0,  # has uppercase
        1 if any(c in 'Rp$€£¥' for c in text) else 0,  # has currency
        1 if any(c in '/-.' for c in text) else 0,  # has date separator
    ], dtype=np.float32)
    
    # Position features (5 features)
    position_features = np.array([
        line.get('y_center', 0.5),  # y position (normalized)
        line.get('x_center', 0.5),  # x position (normalized)
        line.get('width', 0.1),  # width (normalized)
        line.get('height', 0.05),  # height (normalized)
        line.get('line_index', 0),  # line index
    ], dtype=np.float32)
    
    return char_ids, text_features, position_features


class OnlineLearningModel:
    """Online learning wrapper for classification model.
    
    Model learns from user corrections and improves continuously.
    """
    
    def __init__(self, base_model_path: Path = None, buffer_size: int = 50):
        """Initialize online learning model.
        
        Args:
            base_model_path: Path to pre-trained model weights
            buffer_size: Number of corrections before retraining (default: 50 for stability)
        """
        if base_model_path is None:
            base_model_path = ROOT_DIR / "models" / "best_weights.weights.h5"
        
        # Load base model - need to build architecture first
        print(f"[OnlineLearning] Building model architecture...")
        from src.model import ReceiptLineClassifier
        
        self.base_model = ReceiptLineClassifier()
        
        # Build model by calling it once with dummy data
        dummy_char_ids = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
        dummy_text_feats = np.zeros((1, 10), dtype=np.float32)
        dummy_pos_feats = np.zeros((1, 5), dtype=np.float32)
        _ = self.base_model([dummy_char_ids, dummy_text_feats, dummy_pos_feats], training=False)
        
        # Now load weights
        print(f"[OnlineLearning] Loading weights from {base_model_path}")
        self.base_model.load_weights(base_model_path)
        
        # Label mapping
        self.label_to_idx = {v: k for k, v in LINE_CLASSES.items()}
        self.idx_to_label = LINE_CLASSES
        
        # Online learning settings
        self.correction_buffer = []
        self.buffer_size = buffer_size
        self.learning_rate = 0.0005  # Sweet spot: not too high, not too low
        self.is_retraining = False  # Lock to prevent concurrent retraining
        
        # Recompile with optimized learning rate
        self.base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"[OnlineLearning] Model loaded")
        print(f"[OnlineLearning] Buffer size: {buffer_size}")
        print(f"[OnlineLearning] Learning rate: {self.learning_rate}")
        
        # Online model save path
        self.online_model_path = ROOT_DIR / "models" / "online_model.h5"
        
        print(f"[OnlineLearning] Model loaded")
        print(f"[OnlineLearning] Buffer size: {self.buffer_size}")
        print(f"[OnlineLearning] Learning rate: {self.learning_rate}")
    
    def predict(self, lines: List[Dict]) -> Tuple[List[str], List[float]]:
        """Predict labels for lines with post-processing.
        
        Args:
            lines: List of line dicts
            
        Returns:
            Tuple of (predicted_labels, confidences)
        """
        if not lines:
            return [], []
        
        # Extract features
        char_ids_list = []
        text_features_list = []
        position_features_list = []
        
        for line in lines:
            char_ids, text_feats, pos_feats = extract_features_for_line(line)
            char_ids_list.append(char_ids)
            text_features_list.append(text_feats)
            position_features_list.append(pos_feats)
        
        # Convert to arrays
        char_ids_array = np.array(char_ids_list)
        text_features_array = np.array(text_features_list)
        position_features_array = np.array(position_features_list)
        
        # Predict
        predictions = self.base_model.predict(
            [char_ids_array, text_features_array, position_features_array],
            verbose=0
        )
        
        # Convert to labels and confidences
        predicted_labels = []
        confidences = []
        
        for pred in predictions:
            idx = np.argmax(pred)
            confidence = float(pred[idx])
            label = self.idx_to_label[idx]
            
            predicted_labels.append(label)
            confidences.append(confidence)
        
        # Apply post-processing rules to fix obvious mistakes
        predicted_labels = self._post_process_labels(lines, predicted_labels)
        
        return predicted_labels, confidences
    
    def _post_process_labels(self, lines: List[Dict], predicted_labels: List[str]) -> List[str]:
        """Apply post-processing rules to fix obvious mistakes.
        
        Enhanced rules based on common labeling errors:
        - Split tokens (RM + number)
        - Zone-based corrections (top=store/address, middle=items, bottom=totals)
        - Keyword-based overrides
        - Context propagation from neighbors
        
        Args:
            lines: List of line dicts
            predicted_labels: Initial predictions from model
            
        Returns:
            Corrected labels
        """
        import re
        
        corrected_labels = predicted_labels.copy()
        total_lines = len(lines)
        
        if total_lines == 0:
            return corrected_labels
        
        # --- Find total zone start ---
        total_start_idx = total_lines
        for i, line in enumerate(lines):
            text_lower = line.get('text', '').lower().strip()
            if re.search(r'\b(sub\s*total|total|grand\s*total)\b', text_lower):
                total_start_idx = i
                break
        
        for i, (line, label) in enumerate(zip(lines, predicted_labels)):
            text = line.get('text', '')
            text_lower = text.lower().strip()
            text_stripped = text.strip()
            y_pos = line.get('y_center', 0.5)
            relative_pos = i / max(total_lines, 1)
            
            # === Rule 1: DATE patterns always win ===
            # Strict date patterns (DD/MM/YY etc) — must match a real date format
            date_patterns = [
                r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',   # DD/MM/YYYY, DD-MM-YY
                r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}',      # YYYY-MM-DD
                r'\d{1,2}\s+\d{1,2}\s+\d{2,4}',             # DD MM YYYY (spasi)
            ]
            # Strong date keywords — these alone trigger DATE label
            strong_date_keywords = [
                'tanggal', 'tgl', 'tarikh',
                'bill date', 'invoice date',
                'bill start', 'bill end', 'closed bill',
            ]
            # Time-only keywords (jam, time) — only count as DATE if line ALSO has a date pattern
            # Avoid false-positive on standalone time like "10:30:45" or "Jam: 10:30"
            time_only_keywords = ['masa', 'waktu', 'time', 'jam']
            
            has_date_pattern = any(re.search(p, text) for p in date_patterns)
            has_strong_kw = any(kw in text_lower for kw in strong_date_keywords)
            has_time_kw = any(kw in text_lower for kw in time_only_keywords)
            # 'date' keyword is treated as strong, but exclude false positives like "update", "candidate"
            has_date_word = bool(re.search(r'\bdate\b', text_lower))
            
            if has_date_pattern or has_strong_kw or has_date_word:
                # Exclude: pure transaction IDs (>7 digits no separator)
                # Exclude: phone numbers (starts with 0 and >8 digits)
                is_transaction_id = re.match(r'^\d{7,}$', text_stripped)
                is_phone = re.match(r'^0\d{8,}$', text_stripped.replace(' ', '').replace('-', ''))
                if not is_transaction_id and not is_phone:
                    corrected_labels[i] = 'DATE'
                    continue
            # Time keyword alone (no date pattern) → leave as is (don't force DATE)
            # This avoids misclassifying "Jam: 10:30" as DATE
            
            # === Rule 2: TOTAL keywords → TOTAL_PAYMENT ===
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
            cash_keywords = ['cash', 'tunai']
            
            is_total_kw = any(kw in text_lower for kw in total_keywords)
            is_cash_kw = any(kw in text_lower for kw in cash_keywords)
            
            if is_total_kw or (is_cash_kw and (relative_pos > 0.5 or i >= total_start_idx)):
                if 'cashier' not in text_lower and 'kasir' not in text_lower:
                    corrected_labels[i] = 'TOTAL_PAYMENT'
                    continue
            
            # === Rule 3: In total zone, numbers/RM → TOTAL_PAYMENT ===
            if i >= total_start_idx:
                # DATE always wins even in total zone — check first
                # Strict: only date patterns or strong keywords (not time-only)
                date_patterns_total = [
                    r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',
                    r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}',
                    r'\d{1,2}\s+\d{1,2}\s+\d{2,4}',
                ]
                strong_date_kw_total = [
                    'tanggal', 'tgl', 'tarikh',
                    'bill date', 'invoice date',
                    'bill start', 'bill end', 'closed bill',
                ]
                has_date_in_total = any(re.search(p, text) for p in date_patterns_total)
                has_strong_kw_total = any(kw in text_lower for kw in strong_date_kw_total)
                has_date_word_total = bool(re.search(r'\bdate\b', text_lower))
                if has_date_in_total or has_strong_kw_total or has_date_word_total:
                    is_transaction_id = re.match(r'^\d{7,}$', text_stripped)
                    is_phone = re.match(r'^0\d{8,}$', text_stripped.replace(' ', '').replace('-', ''))
                    if not is_transaction_id and not is_phone:
                        corrected_labels[i] = 'DATE'
                        continue

                # Standalone currency or number
                if text_stripped.upper() in ['RM', 'RP', 'RI', 'RH', 'R']:
                    corrected_labels[i] = 'TOTAL_PAYMENT'
                    continue
                # Pure number in total zone
                if re.match(r'^[\d.,\s]+$', text_stripped) and len(text_stripped) >= 2:
                    corrected_labels[i] = 'TOTAL_PAYMENT'
                    continue
                # RM/Rp + number
                if re.search(r'^r[mhp]\.?\s*[\d.,]+', text_lower):
                    corrected_labels[i] = 'TOTAL_PAYMENT'
                    continue
            
            # === Rule 4: Top section STORE detection ===
            company_suffixes = ['sdn bhd', 'sdn. bhd', 'sdn', 'bhd', 'pt.', 'pt ', 
                              'cv.', 'cv ', 'ltd', 'inc', 'corp', 'co.', 
                              'enterprise', 'trading', 'marketing']
            
            if relative_pos < 0.15:
                # Top 3 lines with mostly uppercase = STORE
                if i <= 2:
                    alpha_count = sum(c.isalpha() for c in text)
                    upper_count = sum(c.isupper() for c in text)
                    if alpha_count > 2 and upper_count / max(alpha_count, 1) > 0.5:
                        if label in ['OTHER', 'ITEM_DESC', 'ADDRESS_CONTACT']:
                            corrected_labels[i] = 'STORE'
                            continue
                
                # Company suffix in top section
                if any(s in text_lower for s in company_suffixes):
                    if label in ['OTHER', 'ITEM_DESC', 'ADDRESS_CONTACT']:
                        corrected_labels[i] = 'STORE'
                        continue
            
            # === Rule 5: ADDRESS_CONTACT detection ===
            address_keywords = [
                'jl.', 'jl ', 'jalan', 'lorong', 'taman', 'bandar',
                'kec.', 'kec ', 'kel.', 'kel ', 'desa',
                'telp', 'tel:', 'tel ', 'phone', 'fax',
                'email', 'website', 'www.',
                'npwp', 'gst no', 'gst reg', 'roc no', 'roc nu',
                'plaza', 'heights', 'damansara', 'selangor', 'johor',
                'kuala lumpur', 'malaysia', 'indonesia',
            ]
            is_postal = re.match(r'^\d{5}$', text_stripped)
            
            if (any(kw in text_lower for kw in address_keywords) or is_postal) and relative_pos < 0.35:
                if label in ['OTHER', 'ITEM_DESC', 'STORE', 'ITEM_PRICE/QTY']:
                    corrected_labels[i] = 'ADDRESS_CONTACT'
                    continue
            
            # === Rule 6: ITEM_PRICE/QTY in item zone ===
            if i < total_start_idx and relative_pos > 0.15:
                # Skip transaction IDs (long digit strings with x)
                is_transaction_id = re.match(r'^[0-9]{5,}[xX][0-9]+$', text_stripped)
                if is_transaction_id:
                    continue
                    
                qty_patterns = [
                    r'\d+\s*[xX\*@]\s*[\d.,]+',
                    r'[xX\*@]\s*[\d.,]+',
                    r'=\s*r[mh]\s*[\d.,]+',
                    r'\d+[.,]\d+\s*(uni|unit|pcs)',
                ]
                if any(re.search(p, text_lower) for p in qty_patterns):
                    if label in ['TOTAL_PAYMENT', 'OTHER']:
                        corrected_labels[i] = 'ITEM_PRICE/QTY'
                        continue
                
                # RM + number in item zone = ITEM_PRICE/QTY
                if re.search(r'^r[mhp]\.?\s*[\d.,]+', text_lower):
                    if label == 'TOTAL_PAYMENT':
                        corrected_labels[i] = 'ITEM_PRICE/QTY'
                        continue
            
            # === Rule 7: Footer detection → OTHER ===
            footer_keywords = ['thank', 'terima kasih', 'forward', 'visit', 'welcome',
                             'goods sold', 'no refund', 'no exchange', 'copy', 'duplicate']
            if any(kw in text_lower for kw in footer_keywords):
                corrected_labels[i] = 'OTHER'
                continue
        
        # === Phase 2: Context propagation for split tokens ===
        for i in range(total_lines - 1):
            text_i = lines[i].get('text', '').strip().upper()
            text_next = lines[i + 1].get('text', '').strip()
            
            # "RM" followed by number → both get same label
            if text_i in ['RM', 'RP', 'RI', 'RH'] and re.match(r'^[\d.,\s]+$', text_next):
                if i >= total_start_idx:
                    corrected_labels[i] = 'TOTAL_PAYMENT'
                    corrected_labels[i + 1] = 'TOTAL_PAYMENT'
                elif corrected_labels[i] != 'TOTAL_PAYMENT':
                    corrected_labels[i] = 'ITEM_PRICE/QTY'
                    corrected_labels[i + 1] = 'ITEM_PRICE/QTY'
        
        # === Phase 3: Neighbor smoothing ===
        for i in range(1, total_lines - 1):
            if corrected_labels[i] == 'OTHER':
                relative_pos = i / max(total_lines, 1)
                # In address zone: OTHER between ADDRESS = ADDRESS
                if relative_pos < 0.25:
                    if corrected_labels[i-1] == 'ADDRESS_CONTACT' and corrected_labels[i+1] == 'ADDRESS_CONTACT':
                        corrected_labels[i] = 'ADDRESS_CONTACT'
                # In top zone: OTHER between STORE = STORE
                if relative_pos < 0.1:
                    if corrected_labels[i-1] == 'STORE' and corrected_labels[i+1] == 'STORE':
                        corrected_labels[i] = 'STORE'
        
        return corrected_labels
    
    def add_correction(self, line: Dict, true_label: str):
        """Add user correction to buffer.
        
        Args:
            line: Line dict with text and features
            true_label: Correct label string
        """
        # Extract features
        char_ids, text_feats, pos_feats = extract_features_for_line(line)
        
        # Convert label to one-hot
        label_idx = self.label_to_idx[true_label]
        true_label_onehot = np.zeros(len(self.idx_to_label))
        true_label_onehot[label_idx] = 1
        
        # Add to buffer
        self.correction_buffer.append({
            'char_ids': char_ids,
            'text_features': text_feats,
            'position_features': pos_feats,
            'label': true_label_onehot
        })
        
        print(f"[OnlineLearning] Added correction (buffer: {len(self.correction_buffer)}/{self.buffer_size})")
        
        # Auto-retrain if buffer full
        if len(self.correction_buffer) >= self.buffer_size:
            self.retrain()
    
    def retrain(self):
        """Fine-tune model on corrections in buffer."""
        # Check if already retraining (prevent concurrent calls)
        if self.is_retraining:
            print("[OnlineLearning] ⚠️  Already retraining, skipping...")
            return
        
        if not self.correction_buffer:
            print("[OnlineLearning] No corrections to learn from")
            return
        
        # Set lock
        self.is_retraining = True
        
        try:
            print(f"[OnlineLearning] 🔄 Retraining on {len(self.correction_buffer)} corrections...")
            
            # Prepare data
            char_ids = np.array([c['char_ids'] for c in self.correction_buffer])
            text_features = np.array([c['text_features'] for c in self.correction_buffer])
            position_features = np.array([c['position_features'] for c in self.correction_buffer])
            labels = np.array([c['label'] for c in self.correction_buffer])
        
            # Fine-tune with class weights and higher learning rate
            # Compile with new learning rate
            import tensorflow as tf
            self.base_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Class weights to focus on confused labels (very conservative for stability)
            # Ultra-low weights (1.1-1.2x) to prevent catastrophic forgetting and overfitting
            # Reduced from 1.2-1.5x due to high training variance and class imbalance
            class_weights = {
                0: 1.0,   # OTHER (baseline)
                1: 1.1,   # STORE (was 1.2, reduced for stability)
                2: 1.1,   # ADDRESS_CONTACT (was 1.3, reduced for stability)
                3: 1.2,   # ITEM_DESC (was 1.5, reduced for stability)
                4: 1.2,   # ITEM_PRICE/QTY (was 1.5, reduced for stability)
                5: 1.2,   # TOTAL_PAYMENT (was 1.5, reduced for stability)
                6: 1.1,   # DATE (was 1.2, reduced for stability)
            }
            
            history = self.base_model.fit(
                [char_ids, text_features, position_features],
                labels,
                epochs=5,  # Increased from 3 to 5
                batch_size=16,
                verbose=0,
                validation_split=0.2,
                class_weight=class_weights  # Add class weights!
            )
            
            # Save updated model
            self.online_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.base_model.save(self.online_model_path)
            
            final_loss = history.history['loss'][-1]
            final_acc = history.history['accuracy'][-1]
            
            print(f"[OnlineLearning] ✅ Model updated!")
            print(f"[OnlineLearning]    Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")
            
            # Auto-evaluation
            from src.auto_evaluation import get_auto_evaluator
            evaluator = get_auto_evaluator()
            
            # Track retrain count
            if not hasattr(self, 'retrain_count'):
                self.retrain_count = 0
            self.retrain_count += 1
            
            # Evaluate
            evaluation = evaluator.evaluate_retrain(
                retrain_id=self.retrain_count,
                loss=final_loss,
                accuracy=final_acc,
                corrections_count=len(self.correction_buffer),
                buffer_size=self.buffer_size
            )
            
            # Check for alerts
            alerts = evaluator.get_alerts()
            for alert in alerts:
                if alert['severity'] in ['high', 'medium']:
                    print(f"[AutoEval] ⚠️  {alert['message']}")
                elif alert['type'] == 'success':
                    print(f"[AutoEval] ✅ {alert['message']}")
            
            # Clear buffer
            self.correction_buffer = []
            
        except Exception as e:
            print(f"[OnlineLearning] ❌ Retrain failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always release lock
            self.is_retraining = False
    
    def force_retrain(self):
        """Force retrain even if buffer not full."""
        if self.correction_buffer:
            self.retrain()
        else:
            print("[OnlineLearning] No corrections in buffer")
    
    def get_buffer_status(self) -> Dict:
        """Get current buffer status.
        
        Returns:
            Dict with buffer info
        """
        return {
            'buffer_size': len(self.correction_buffer),
            'buffer_capacity': self.buffer_size,
            'buffer_full': len(self.correction_buffer) >= self.buffer_size,
            'progress': len(self.correction_buffer) / self.buffer_size * 100
        }


class CorrectionTracker:
    """Track user corrections and analyze patterns."""
    
    def __init__(self, save_path: Path = None):
        """Initialize correction tracker.
        
        Args:
            save_path: Path to save corrections log
        """
        if save_path is None:
            save_path = ROOT_DIR / "data" / "corrections_log.json"
        
        self.save_path = save_path
        self.corrections = []
        
        # Load existing corrections
        if self.save_path.exists():
            with open(self.save_path, 'r', encoding='utf-8') as f:
                self.corrections = json.load(f)
            print(f"[CorrectionTracker] Loaded {len(self.corrections)} corrections")
        else:
            print(f"[CorrectionTracker] Starting fresh")
    
    def add_correction(self, correction: Dict):
        """Add correction to tracker.
        
        Args:
            correction: Dict with correction info
        """
        correction['timestamp'] = datetime.now().isoformat()
        self.corrections.append(correction)
        
        # Auto-save
        self.save()
    
    def save(self):
        """Save corrections to file."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.corrections, f, indent=2, ensure_ascii=False)
    
    def get_stats(self) -> Dict:
        """Get correction statistics.
        
        Returns:
            Dict with stats
        """
        if not self.corrections:
            return {
                'total_corrections': 0,
                'common_errors': [],
                'unique_error_types': 0
            }
        
        # Count by error type
        error_counts = {}
        for c in self.corrections:
            pred = c.get('predicted', 'UNKNOWN')
            correct = c.get('correct', 'UNKNOWN')
            key = f"{pred} → {correct}"
            error_counts[key] = error_counts.get(key, 0) + 1
        
        # Sort by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_corrections': len(self.corrections),
            'common_errors': sorted_errors[:10],
            'unique_error_types': len(error_counts)
        }
    
    def get_recent_corrections(self, n: int = 10) -> List[Dict]:
        """Get N most recent corrections.
        
        Args:
            n: Number of corrections to return
            
        Returns:
            List of recent corrections
        """
        return self.corrections[-n:]
    
    def get_accuracy_trend(self, window_size: int = 50) -> List[float]:
        """Get accuracy trend over time.
        
        Args:
            window_size: Window size for moving average
            
        Returns:
            List of accuracy values
        """
        if len(self.corrections) < window_size:
            return []
        
        accuracies = []
        for i in range(window_size, len(self.corrections) + 1):
            window = self.corrections[i-window_size:i]
            # Accuracy = 1 - (corrections / total)
            # Assuming each correction is an error
            accuracy = 1.0 - (len(window) / window_size)
            accuracies.append(accuracy)
        
        return accuracies


class ActiveLearningStrategy:
    """Strategy for deciding which examples to learn from."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize active learning strategy.
        
        Args:
            confidence_threshold: Confidence threshold for learning
        """
        self.confidence_threshold = confidence_threshold
    
    def should_learn(self, predicted: str, corrected: str, confidence: float) -> bool:
        """Decide if this example should be learned.
        
        Args:
            predicted: Predicted label
            corrected: Corrected label
            confidence: Prediction confidence
            
        Returns:
            True if should learn, False otherwise
        """
        # Always learn from corrections
        if predicted != corrected:
            return True
        
        # Learn from low confidence predictions (even if correct)
        if confidence < self.confidence_threshold:
            return True
        
        # Skip high confidence correct predictions
        return False
    
    def prioritize_corrections(self, corrections: List[Dict]) -> List[Dict]:
        """Prioritize corrections for learning.
        
        Args:
            corrections: List of corrections
            
        Returns:
            Prioritized list of corrections
        """
        # Priority 1: Low confidence errors (< 0.5)
        priority_1 = [c for c in corrections if c.get('confidence', 0.5) < 0.5]
        
        # Priority 2: Medium confidence errors (0.5-0.7)
        priority_2 = [c for c in corrections if 0.5 <= c.get('confidence', 0.5) < 0.7]
        
        # Priority 3: High confidence errors (>= 0.7) - surprising!
        priority_3 = [c for c in corrections if c.get('confidence', 0.5) >= 0.7]
        
        return priority_1 + priority_2 + priority_3


# Singleton instances
_online_model = None
_correction_tracker = None
_active_learning = None


def get_online_model() -> OnlineLearningModel:
    """Get singleton online learning model."""
    global _online_model
    if _online_model is None:
        _online_model = OnlineLearningModel()
    return _online_model


def get_correction_tracker() -> CorrectionTracker:
    """Get singleton correction tracker."""
    global _correction_tracker
    if _correction_tracker is None:
        _correction_tracker = CorrectionTracker()
    return _correction_tracker


def get_active_learning() -> ActiveLearningStrategy:
    """Get singleton active learning strategy."""
    global _active_learning
    if _active_learning is None:
        _active_learning = ActiveLearningStrategy()
    return _active_learning
