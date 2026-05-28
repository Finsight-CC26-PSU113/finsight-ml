"""
OCR FinSight - Simple Web App
Uses existing models: online_model.h5 + fine-tuned EasyOCR
"""

import sys
import os
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import cv2
import base64

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ROOT_DIR
from src.preprocessing import load_image, deskew
from src.ocr_engine import OCREngine
from src.extractor import ReceiptExtractor

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global objects
ocr_engine = None
classifier = None
extractor = None

def init_ocr():
    """Initialize OCR engine."""
    global ocr_engine
    if ocr_engine is None:
        print("Loading EasyOCR...")
        ocr_engine = OCREngine()
        print("✅ EasyOCR ready")

def init_classifier():
    """Initialize classifier — try V2 (new) first, fallback to online_learning."""
    global classifier
    if classifier is None:
        # Try V2 first (better accuracy on DATE/STORE/ITEMS)
        try:
            from src.model import ReceiptLineClassifier
            from src.config import NUM_CLASSES, MAX_TEXT_LENGTH
            import tensorflow as tf
            
            v2_weights = ROOT_DIR / "models" / "classifier_v2" / "best_weights.weights.h5"
            if not v2_weights.exists():
                raise FileNotFoundError(f"V2 weights not found: {v2_weights}")
            
            print("Loading classifier V2...")
            model = ReceiptLineClassifier()
            
            # Build with dummy input
            dummy_chars = tf.zeros((1, MAX_TEXT_LENGTH), dtype=tf.int32)
            dummy_tf = tf.zeros((1, 10), dtype=tf.float32)
            dummy_pf = tf.zeros((1, 5), dtype=tf.float32)
            _ = model((dummy_chars, dummy_tf, dummy_pf), training=False)
            
            model.load_weights(str(v2_weights))
            classifier = model
            classifier_meta = {'version': 'v2', 'is_v2': True}
            print(f"✅ Classifier V2 loaded ({v2_weights.name}) — 85% acc, DATE recall 96%")
            
            # Store version info on the classifier
            classifier._is_v2 = True
            return
        except Exception as e:
            print(f"⚠️  V2 model failed: {e}, falling back to online learning model...")
        
        # Fallback to old online learning model
        try:
            from src.online_learning import OnlineLearningModel
            classifier = OnlineLearningModel()
            classifier._is_v2 = False
            print("✅ Online Learning Model loaded (fallback)")
        except Exception as e:
            print(f"❌ All classifier loading failed: {e}")
            classifier = None

def init_extractor():
    """Initialize extractor."""
    global extractor
    if extractor is None:
        extractor = ReceiptExtractor()
        print("✅ Extractor ready")


def predict_with_v2(model, lines):
    """Predict using classifier V2.
    
    Uses feature extraction from train_classifier_v2.py.
    """
    import re
    from src.config import LINE_CLASSES, MAX_TEXT_LENGTH
    import tensorflow as tf
    
    # Feature regex (same as train_classifier_v2.py)
    DATE_REGEX = re.compile(
        r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})|'
        r'(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})|'
        r'(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|'
        r'januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)'
        r'\s+\d{2,4})|'
        r'((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}[\s,]+\d{2,4})',
        re.IGNORECASE
    )
    TIME_REGEX = re.compile(r'\d{1,2}:\d{2}(:\d{2})?')
    QTY_REGEX = re.compile(r'\d+\s*[xX\*@]\s*[\d.,]+|@\s*[\d.,]+')
    CURRENCY_REGEX = re.compile(r'(rm|rp|idr|usd|sgd|sr|\$|€)\s*[\d.,]+', re.IGNORECASE)
    
    def text_features(text):
        if not text:
            return [0.0] * 10
        text = str(text)
        n = len(text)
        digits = sum(c.isdigit() for c in text)
        alphas = sum(c.isalpha() for c in text)
        uppers = sum(c.isupper() for c in text)
        spaces = sum(c.isspace() for c in text)
        specials = n - digits - alphas - spaces
        return [
            digits / max(n, 1),
            alphas / max(n, 1),
            uppers / max(alphas, 1),
            spaces / max(n, 1),
            specials / max(n, 1),
            n / 100.0,
            len(text.split()) / 20.0,
            1.0 if CURRENCY_REGEX.search(text) else 0.0,
            1.0 if (DATE_REGEX.search(text) or TIME_REGEX.search(text)) else 0.0,
            1.0 if QTY_REGEX.search(text) else 0.0,
        ]
    
    def char_ids(text, max_length=MAX_TEXT_LENGTH):
        if not text:
            text = ""
        text = str(text)
        ids = [min(ord(c), 127) for c in text[:max_length]]
        ids = ids + [0] * (max_length - len(ids))
        return ids
    
    n = len(lines)
    char_ids_arr = np.array([char_ids(l.get('text', '')) for l in lines], dtype=np.int32)
    text_feats_arr = np.array([text_features(l.get('text', '')) for l in lines], dtype=np.float32)
    pos_feats_arr = np.array([
        [
            float(l.get('y_center', 0.5)),
            float(l.get('x_center', 0.5)),
            float(l.get('width', 0.1)),
            float(l.get('height', 0.05)),
            float(i) / max(n, 1),
        ] for i, l in enumerate(lines)
    ], dtype=np.float32)
    
    predictions = model((char_ids_arr, text_feats_arr, pos_feats_arr), training=False).numpy()
    pred_idx = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    labels = [LINE_CLASSES[int(idx)] for idx in pred_idx]
    return labels, [float(c) for c in confidences]


def process_receipt(image_file):
    """Process receipt image through full pipeline."""
    # Read image
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image format")
    
    # Preprocess
    img = deskew(img)
    
    # OCR
    lines, img_info = ocr_engine.read_receipt_with_image_info(img)
    
    if not lines:
        return {
            'error': 'No text detected in image',
            'lines': [],
            'extracted': {}
        }
    
    # Classify lines
    classified_lines = []
    if classifier:
        try:
            is_v2 = getattr(classifier, '_is_v2', False)
            
            if is_v2:
                # V2: Direct TF model with new feature extraction
                predicted_labels, confidences = predict_with_v2(classifier, lines)
            else:
                # Legacy: OnlineLearningModel with built-in predict()
                predicted_labels, confidences = classifier.predict(lines)
            
            for line, label, conf in zip(lines, predicted_labels, confidences):
                line_copy = dict(line)
                line_copy['predicted_class'] = label
                line_copy['class_confidence'] = conf
                classified_lines.append(line_copy)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Classification error: {e}")
            # Fallback: mark all as OTHER
            for line in lines:
                line_copy = dict(line)
                line_copy['predicted_class'] = 'OTHER'
                line_copy['class_confidence'] = 0.5
                classified_lines.append(line_copy)
    else:
        # No classifier: mark all as OTHER
        for line in lines:
            line_copy = dict(line)
            line_copy['predicted_class'] = 'OTHER'
            line_copy['class_confidence'] = 0.5
            classified_lines.append(line_copy)
    
    # Extract structured data
    extracted = extractor.extract(classified_lines)
    
    # Format for display
    result = {
        'success': True,
        'lines': [
            {
                'text': line['text'],
                'label': line['predicted_class'],
                'confidence': round(line['class_confidence'], 3),
                'ocr_confidence': round(line['confidence'], 3),
                'bbox': {
                    'x_min': line['x_min'],
                    'y_min': line['y_min'],
                    'x_max': line['x_max'],
                    'y_max': line['y_max']
                }
            }
            for line in classified_lines
        ],
        'extracted': {
            'store': extracted['store'],
            'date': extracted['date'],
            'items': [f"{item['name']} - Qty: {item['qty']} - Price: {item['price']}" 
                     for item in extracted['items']],
            'total': f"Rp {extracted['total']:,.0f}".replace(',', '.') if extracted['total'] > 0 else 'N/A',
            'address': extracted['address']
        },
        'stats': {
            'total_lines': len(classified_lines),
            'avg_confidence': round(np.mean([line['class_confidence'] for line in classified_lines]) * 100, 1)
        }
    }
    
    return result

@app.route('/')
def index():
    """Main page."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCR FinSight - Receipt Scanner</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
            .upload-area:hover { border-color: #007bff; }
            input[type="file"] { margin: 10px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 5px; }
            .line { margin: 5px 0; padding: 8px; border-left: 4px solid #007bff; background: white; }
            .extracted { background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .error { color: red; background: #ffe6e6; padding: 10px; border-radius: 5px; }
            .loading { color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧾 OCR FinSight - Receipt Scanner</h1>
            <p style="text-align: center; color: #666;">Upload gambar struk untuk ekstraksi data otomatis</p>
            
            <div class="upload-area">
                <p>📁 Pilih gambar struk (JPG, PNG)</p>
                <input type="file" id="imageInput" accept="image/*">
                <br>
                <button onclick="processImage()">🔍 Scan Receipt</button>
            </div>
            
            <div id="result"></div>
        </div>

        <script>
            function processImage() {
                const input = document.getElementById('imageInput');
                const resultDiv = document.getElementById('result');
                
                if (!input.files[0]) {
                    alert('Pilih gambar terlebih dahulu');
                    return;
                }
                
                resultDiv.innerHTML = '<div class="loading">⏳ Processing image...</div>';
                
                const formData = new FormData();
                formData.append('image', input.files[0]);
                
                fetch('/api/process', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayResult(data);
                    } else {
                        resultDiv.innerHTML = `<div class="error">❌ Error: ${data.error}</div>`;
                    }
                })
                .catch(error => {
                    resultDiv.innerHTML = `<div class="error">❌ Network error: ${error}</div>`;
                });
            }
            
            function displayResult(data) {
                const resultDiv = document.getElementById('result');
                
                let html = '<div class="result">';
                html += '<h3>📊 Extraction Results</h3>';
                
                // Extracted data
                html += '<div class="extracted">';
                html += '<h4>🏪 Structured Data:</h4>';
                html += `<p><strong>Store:</strong> ${data.extracted.store}</p>`;
                html += `<p><strong>Date:</strong> ${data.extracted.date}</p>`;
                html += `<p><strong>Total:</strong> ${data.extracted.total}</p>`;
                html += `<p><strong>Items:</strong> ${data.extracted.items.length} detected</p>`;
                if (data.extracted.items.length > 0) {
                    html += '<ul>';
                    data.extracted.items.forEach(item => {
                        html += `<li>${item}</li>`;
                    });
                    html += '</ul>';
                }
                html += '</div>';
                
                // Stats
                html += `<p><strong>📈 Stats:</strong> ${data.stats.total_lines} lines detected, ${data.stats.avg_confidence}% avg confidence</p>`;
                
                // Lines
                html += '<h4>📝 Detected Lines:</h4>';
                data.lines.forEach((line, i) => {
                    const labelColor = getLabelColor(line.label);
                    html += `<div class="line" style="border-left-color: ${labelColor}">`;
                    html += `<strong>${line.label}</strong> (${(line.confidence * 100).toFixed(1)}%) - ${line.text}`;
                    html += '</div>';
                });
                
                html += '</div>';
                resultDiv.innerHTML = html;
            }
            
            function getLabelColor(label) {
                const colors = {
                    'STORE': '#ff6b6b',
                    'DATE': '#4ecdc4',
                    'TOTAL_PAYMENT': '#45b7d1',
                    'ITEM_DESC': '#96ceb4',
                    'ITEM_PRICE/QTY': '#feca57',
                    'ADDRESS_CONTACT': '#ff9ff3',
                    'OTHER': '#95a5a6'
                };
                return colors[label] || '#95a5a6';
            }
        </script>
    </body>
    </html>
    '''

@app.route('/api/process', methods=['POST'])
def api_process():
    """Process uploaded image."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    try:
        result = process_receipt(file)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health')
def health():
    """Health check."""
    return jsonify({
        'status': 'ok',
        'ocr': ocr_engine is not None,
        'classifier': classifier is not None,
        'extractor': extractor is not None
    })

if __name__ == '__main__':
    print("=" * 50)
    print("🧾 OCR FinSight - Simple Receipt Scanner")
    print("=" * 50)
    
    # Initialize components
    init_ocr()
    init_classifier()
    init_extractor()
    
    print("\n🌐 Open: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=False, host='0.0.0.0', port=5000)