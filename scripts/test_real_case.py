"""
Real Case Test: OCR Fine-tuned + TF Classifier
Test pipeline lengkap pada gambar struk nyata.
Bandingkan: Original EasyOCR vs Fine-tuned EasyOCR + Classifier

Usage:
    wsl python3 scripts/test_real_case.py
    wsl python3 scripts/test_real_case.py --images 000.jpg 050.jpg 100.jpg
"""

import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ROOT_DIR
from src.preprocessing import load_image, deskew
from src.ocr_engine import OCREngine

IMAGES_DIR = ROOT_DIR / "fullDataset" / "images"
FINETUNED_MODEL = ROOT_DIR / "models" / "finetuned_easyocr" / "best_model.pth"
TF_MODEL = ROOT_DIR / "models" / "online_model.h5"
TF_BASE_MODEL = ROOT_DIR / "models" / "best_weights.weights.h5"

# Label colors for display
LABEL_COLORS = {
    'STORE':          '🏪',
    'ADDRESS_CONTACT':'📍',
    'DATE':           '📅',
    'ITEM_DESC':      '🛒',
    'ITEM_PRICE/QTY': '💰',
    'TOTAL_PAYMENT':  '💳',
    'OTHER':          '⬜',
}

def load_tf_classifier():
    """Load TF BiLSTM classifier."""
    try:
        import tensorflow as tf
        from src.model import ReceiptLineClassifier
        from src.config import NUM_CLASSES, EMBEDDING_DIM, LSTM_UNITS, DENSE_UNITS, DROPOUT_RATE

        model = ReceiptLineClassifier(
            num_classes=NUM_CLASSES,
            embedding_dim=EMBEDDING_DIM,
            lstm_units=LSTM_UNITS,
            dense_units=DENSE_UNITS,
            dropout_rate=DROPOUT_RATE
        )

        # Try online model first, fallback to base
        if TF_MODEL.exists():
            model.load_weights(str(TF_MODEL))
            print(f"  ✅ TF Classifier loaded (online model)")
        elif TF_BASE_MODEL.exists():
            model.load_weights(str(TF_BASE_MODEL))
            print(f"  ✅ TF Classifier loaded (base model)")
        else:
            print(f"  ⚠️  No TF model found")
            return None

        return model
    except Exception as e:
        print(f"  ❌ TF Classifier error: {e}")
        return None


def load_finetuned_ocr():
    """Load fine-tuned EasyOCR recognition model."""
    try:
        import torch
        import easyocr
        easyocr_dir = Path(easyocr.__file__).parent
        sys.path.insert(0, str(easyocr_dir))
        from model.vgg_model import Model as EasyOCRModel

        # Import charset
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from finetune_easyocr import CHARSET, CHAR_TO_IDX, IDX_TO_CHAR, NUM_CLASSES as OCR_CLASSES, ctc_decode

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = EasyOCRModel(
            input_channel=1,
            output_channel=256,
            hidden_size=256,
            num_class=OCR_CLASSES
        )
        model.load_state_dict(torch.load(str(FINETUNED_MODEL), map_location=device))
        model.to(device)
        model.eval()

        print(f"  ✅ Fine-tuned OCR loaded ({device})")
        return model, device, ctc_decode
    except Exception as e:
        print(f"  ❌ Fine-tuned OCR error: {e}")
        return None, None, None


def recognize_with_finetuned(model, device, ctc_decode_fn, img, line):
    """Recognize text using fine-tuned model."""
    import torch
    import cv2

    h, w = img.shape[:2]
    x_min = max(0, int(line['x_min'] * w) - 4)
    y_min = max(0, int(line['y_min'] * h) - 2)
    x_max = min(w, int(line['x_max'] * w) + 4)
    y_max = min(h, int(line['y_max'] * h) + 2)

    if len(img.shape) == 3:
        crop = img[y_min:y_max, x_min:x_max]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop = img[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return line.get('text', '')

    ch, cw = crop.shape
    new_w = max(int(cw * 32 / ch), 1)
    crop = cv2.resize(crop, (new_w, 32), interpolation=cv2.INTER_CUBIC)

    if crop.shape[1] < 256:
        import numpy as np
        pad = np.ones((32, 256 - crop.shape[1]), dtype=np.uint8) * 255
        crop = np.concatenate([crop, pad], axis=1)
    else:
        crop = crop[:, :256]

    crop_tensor = torch.FloatTensor(crop).unsqueeze(0).unsqueeze(0)
    crop_tensor = (crop_tensor / 255.0 - 0.5) / 0.5
    crop_tensor = crop_tensor.to(device)

    with torch.no_grad():
        output = model(crop_tensor, None)
        output = output.permute(1, 0, 2).log_softmax(2)
        decoded = ctc_decode_fn(output)

    return decoded[0] if decoded else line.get('text', '')


def classify_with_tf(tf_model, lines, img_info):
    """Classify lines using TF BiLSTM."""
    try:
        from src.extractor import ReceiptExtractor
        from src.online_learning import OnlineLearningModel

        ol = OnlineLearningModel()
        results = []
        for line in lines:
            pred_class, confidence = ol.predict_line(line, img_info)
            results.append((pred_class, confidence))
        return results
    except Exception as e:
        return [('OTHER', 0.0)] * len(lines)


def test_image(image_path, ocr_engine, ft_model, ft_device, ctc_fn, tf_model):
    """Run full pipeline on one image."""
    print(f"\n{'='*70}")
    print(f"📷 Image: {Path(image_path).name}")
    print(f"{'='*70}")

    # Load image
    img = load_image(image_path)
    img = deskew(img)

    # OCR
    t0 = time.time()
    lines, img_info = ocr_engine.read_receipt_with_image_info(img)
    ocr_time = time.time() - t0

    if not lines:
        print("  ⚠️  No lines detected")
        return

    print(f"\n📊 Detected {len(lines)} lines | OCR time: {ocr_time:.2f}s")
    print(f"\n{'─'*70}")
    print(f"{'#':>3} | {'Conf':>5} | {'Label':>15} | {'Orig EasyOCR':<30} | {'Fine-tuned OCR':<30}")
    print(f"{'─'*70}")

    # Process each line
    store_lines = []
    date_lines = []
    item_lines = []
    total_lines = []

    for i, line in enumerate(lines):
        orig_text = line.get('text', '').strip()
        conf = line.get('confidence', 0.0)

        # Fine-tuned OCR
        if ft_model is not None:
            ft_text = recognize_with_finetuned(ft_model, ft_device, ctc_fn, img, line)
        else:
            ft_text = orig_text

        # TF Classifier (use fine-tuned text)
        line_copy = dict(line)
        line_copy['text'] = ft_text

        try:
            from src.online_learning import OnlineLearningModel
            ol = OnlineLearningModel()
            pred_class, class_conf = ol.predict_line(line_copy, img_info)
        except:
            pred_class, class_conf = 'OTHER', 0.0

        # Track for extraction
        if pred_class == 'STORE':
            store_lines.append(ft_text)
        elif pred_class == 'DATE':
            date_lines.append(ft_text)
        elif pred_class in ['ITEM_DESC', 'ITEM_PRICE/QTY']:
            item_lines.append((pred_class, ft_text))
        elif pred_class == 'TOTAL_PAYMENT':
            total_lines.append(ft_text)

        # Display
        icon = LABEL_COLORS.get(pred_class, '⬜')
        conf_icon = "✅" if conf > 0.8 else "⚠️" if conf > 0.5 else "❌"
        changed = "→" if orig_text != ft_text else " "

        print(f"{i:>3} | {conf:>5.2f} | {icon} {pred_class:>13} | {orig_text[:28]:<30} | {changed} {ft_text[:28]:<28}")

    # Extraction summary
    print(f"\n{'─'*70}")
    print(f"📋 EXTRACTION RESULT")
    print(f"{'─'*70}")
    print(f"  🏪 Store    : {' '.join(store_lines[:2]) if store_lines else '(not found)'}")
    print(f"  📅 Date     : {' '.join(date_lines[:1]) if date_lines else '(not found)'}")
    print(f"  🛒 Items    : {len([x for x in item_lines if x[0]=='ITEM_DESC'])} items detected")
    print(f"  💳 Total    : {' | '.join(total_lines[:3]) if total_lines else '(not found)'}")

    # Count changes
    changes = sum(1 for line in lines
                  if ft_model and recognize_with_finetuned(ft_model, ft_device, ctc_fn, img, line) != line.get('text', ''))
    print(f"\n  🔄 OCR corrections: {changes}/{len(lines)} lines improved by fine-tuning")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', default=['050.jpg', '100.jpg', '200.jpg', '300.jpg', '400.jpg'],
                        help='Image filenames to test')
    parser.add_argument('--dir', type=str, default=None, help='Custom images directory')
    args = parser.parse_args()

    images_dir = Path(args.dir) if args.dir else IMAGES_DIR

    print("=" * 70)
    print("🧪 OCR FinSight 2.0 - Real Case Test")
    print("   Fine-tuned EasyOCR + BiLSTM Classifier")
    print("=" * 70)

    # Load models
    print("\n📦 Loading models...")
    ocr_engine = OCREngine()
    ft_model, ft_device, ctc_fn = load_finetuned_ocr()
    tf_model = load_tf_classifier()

    if ft_model is None:
        print("  ⚠️  Running with original EasyOCR only")

    # Test images
    for img_name in args.images:
        img_path = images_dir / img_name
        if not img_path.exists():
            # Try without extension
            for ext in ['.jpg', '.png', '.jpeg']:
                candidate = images_dir / (img_name.replace('.jpg','').replace('.png','') + ext)
                if candidate.exists():
                    img_path = candidate
                    break

        if img_path.exists():
            test_image(str(img_path), ocr_engine, ft_model, ft_device, ctc_fn, tf_model)
        else:
            print(f"\n⚠️  Image not found: {img_name}")

    print(f"\n{'='*70}")
    print("✅ Test complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
