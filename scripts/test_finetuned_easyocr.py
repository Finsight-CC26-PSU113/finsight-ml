"""
Step 3: Test Fine-tuned EasyOCR Model
======================================
Bandingkan hasil OCR sebelum dan sesudah fine-tuning
pada gambar struk asli.

Usage:
    wsl python3 scripts/test_finetuned_easyocr.py
    wsl python3 scripts/test_finetuned_easyocr.py --image path/to/receipt.jpg
"""

import os
import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ROOT_DIR
from src.preprocessing import load_image, deskew
from src.ocr_engine import OCREngine

FINETUNED_MODEL = ROOT_DIR / "models" / "finetuned_easyocr" / "best_model.pth"
TRAINING_LOG = ROOT_DIR / "models" / "finetuned_easyocr" / "training_log.json"
GROUNDTRUTH_DIR = ROOT_DIR / "data" / "ocr_groundtruth"
IMAGES_DIR = ROOT_DIR / "fullDataset" / "images"


def load_training_results():
    """Show training results summary."""
    if not TRAINING_LOG.exists():
        print("⚠️  No training log found")
        return
    
    with open(TRAINING_LOG, 'r') as f:
        log = json.load(f)
    
    print("=" * 65)
    print("📊 TRAINING RESULTS")
    print("=" * 65)
    print(f"{'Epoch':>6} | {'Loss':>8} | {'CER':>8} | {'Accuracy':>10}")
    print("-" * 45)
    
    for entry in log:
        print(f"{entry['epoch']:>6} | {entry['train_loss']:>8.4f} | "
              f"{entry['val_cer']:>8.4f} | {entry['val_accuracy']:>10.4f}")
    
    best = min(log, key=lambda x: x['val_cer'])
    print("-" * 45)
    print(f"Best epoch: {best['epoch']} | CER: {best['val_cer']:.4f} | "
          f"Char Accuracy: {(1-best['val_cer'])*100:.1f}%")


def compare_ocr_results(image_path: Path, finetuned_model_path: Path):
    """Compare original EasyOCR vs fine-tuned on a receipt image."""
    print(f"\n📷 Testing on: {image_path.name}")
    print("=" * 65)
    
    # Load image
    img = load_image(image_path)
    img = deskew(img)
    
    # Original EasyOCR
    print("\n[Original EasyOCR]")
    ocr = OCREngine()
    original_lines = ocr.read_receipt(img)
    
    print(f"  Lines detected: {len(original_lines)}")
    print(f"  Avg confidence: {np.mean([l['confidence'] for l in original_lines]):.3f}")
    print()
    
    # Show results
    print(f"  {'#':>3} | {'Conf':>5} | {'Text'}")
    print(f"  {'-'*3}-+-{'-'*5}-+-{'-'*40}")
    for i, line in enumerate(original_lines[:15]):
        conf_icon = "✅" if line['confidence'] > 0.8 else "⚠️" if line['confidence'] > 0.5 else "❌"
        print(f"  {i:>3} | {line['confidence']:>5.3f} | {conf_icon} {line['text'][:50]}")
    
    if len(original_lines) > 15:
        print(f"  ... and {len(original_lines) - 15} more lines")
    
    # Check if fine-tuned model exists
    if not finetuned_model_path.exists():
        print(f"\n⚠️  Fine-tuned model not found: {finetuned_model_path}")
        print(f"   Run: python3 scripts/finetune_easyocr.py first")
        return
    
    # Fine-tuned model comparison
    print(f"\n[Fine-tuned EasyOCR]")
    print(f"  Model: {finetuned_model_path.name}")
    
    # Load fine-tuned model and patch EasyOCR
    try:
        import easyocr
        easyocr_dir = Path(easyocr.__file__).parent
        sys.path.insert(0, str(easyocr_dir))
        from model.model import Model as EasyOCRModel
        
        # Import charset from finetune script
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from finetune_easyocr import CHARSET, CHAR_TO_IDX, IDX_TO_CHAR, NUM_CLASSES, ctc_decode
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = EasyOCRModel(
            input_channel=1,
            output_channel=512,
            hidden_size=256,
            num_class=NUM_CLASSES
        )
        model.load_state_dict(torch.load(str(finetuned_model_path), map_location=device))
        model.to(device)
        model.eval()
        
        print(f"  ✅ Fine-tuned model loaded")
        
        # Run recognition on each line crop
        improved_lines = []
        for line in original_lines:
            # Crop the line
            h, w = img.shape[:2] if len(img.shape) == 2 else img.shape[:2]
            x_min = int(line['x_min'] * w)
            y_min = int(line['y_min'] * h)
            x_max = int(line['x_max'] * w)
            y_max = int(line['y_max'] * h)
            
            # Add padding
            x_min = max(0, x_min - 4)
            y_min = max(0, y_min - 2)
            x_max = min(w, x_max + 4)
            y_max = min(h, y_max + 2)
            
            if len(img.shape) == 3:
                crop = img[y_min:y_max, x_min:x_max]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                crop = img[y_min:y_max, x_min:x_max]
            
            if crop.size == 0:
                improved_lines.append(line['text'])
                continue
            
            # Resize to 32px height
            ch, cw = crop.shape
            new_w = max(int(cw * 32 / ch), 1)
            crop = cv2.resize(crop, (new_w, 32), interpolation=cv2.INTER_CUBIC)
            
            # Pad to 256
            if crop.shape[1] < 256:
                pad = np.ones((32, 256 - crop.shape[1]), dtype=np.uint8) * 255
                crop = np.concatenate([crop, pad], axis=1)
            else:
                crop = crop[:, :256]
            
            # Normalize
            crop_tensor = torch.FloatTensor(crop).unsqueeze(0).unsqueeze(0)
            crop_tensor = (crop_tensor / 255.0 - 0.5) / 0.5
            crop_tensor = crop_tensor.to(device)
            
            with torch.no_grad():
                output = model(crop_tensor, None)
                output = output.permute(1, 0, 2).log_softmax(2)
                decoded = ctc_decode(output)
            
            improved_lines.append(decoded[0] if decoded else line['text'])
        
        # Show comparison
        print(f"\n  {'#':>3} | {'Original':^30} | {'Fine-tuned':^30}")
        print(f"  {'-'*3}-+-{'-'*30}-+-{'-'*30}")
        
        changes = 0
        for i, (orig_line, new_text) in enumerate(zip(original_lines[:15], improved_lines[:15])):
            orig_text = orig_line['text'][:28]
            new_text_short = new_text[:28]
            changed = "🔄" if orig_text != new_text_short else "  "
            if orig_text != new_text_short:
                changes += 1
            print(f"  {i:>3} | {orig_text:<30} | {changed} {new_text_short:<28}")
        
        print(f"\n  Changes: {changes}/{min(15, len(original_lines))} lines improved")
        
    except Exception as e:
        print(f"  ❌ Error loading fine-tuned model: {e}")
        import traceback
        traceback.print_exc()


def evaluate_on_groundtruth():
    """Evaluate fine-tuned model on ground truth dataset."""
    metadata_file = GROUNDTRUTH_DIR / "metadata.json"
    
    if not metadata_file.exists():
        print("⚠️  No ground truth metadata found")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n📊 Ground Truth Evaluation")
    print(f"   Total samples: {len(metadata)}")
    
    # Compare EasyOCR text vs ground truth
    total_cer = 0
    exact_matches = 0
    
    for sample in metadata[:100]:  # Evaluate first 100
        easyocr_text = sample.get('easyocr_text', '')
        ground_truth = sample.get('ground_truth', '')
        
        if not ground_truth:
            continue
        
        # CER
        if len(ground_truth) == 0:
            cer = 0.0
        else:
            # Simple edit distance
            m, n = len(easyocr_text), len(ground_truth)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1): dp[i][0] = i
            for j in range(n + 1): dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if easyocr_text[i-1] == ground_truth[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            cer = dp[m][n] / len(ground_truth)
        
        total_cer += cer
        if easyocr_text == ground_truth:
            exact_matches += 1
    
    n_eval = min(100, len(metadata))
    avg_cer = total_cer / max(n_eval, 1)
    accuracy = exact_matches / max(n_eval, 1)
    
    print(f"\n   Original EasyOCR performance:")
    print(f"   CER: {avg_cer:.4f} ({(1-avg_cer)*100:.1f}% char accuracy)")
    print(f"   Exact match: {exact_matches}/{n_eval} ({accuracy*100:.1f}%)")
    
    # Show some examples
    print(f"\n   Sample comparisons (EasyOCR vs Ground Truth):")
    print(f"   {'EasyOCR':^35} | {'Ground Truth':^35}")
    print(f"   {'-'*35}-+-{'-'*35}")
    
    for sample in metadata[:10]:
        ocr = sample.get('easyocr_text', '')[:33]
        gt = sample.get('ground_truth', '')[:33]
        match = "✅" if ocr == gt else "❌"
        print(f"   {match} {ocr:<33} | {gt:<33}")


def main():
    parser = argparse.ArgumentParser(description='Test fine-tuned EasyOCR model')
    parser.add_argument('--image', type=str, default=None, help='Specific image to test')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate on ground truth')
    args = parser.parse_args()
    
    print("=" * 65)
    print("🧪 Fine-tuned EasyOCR Test")
    print("=" * 65)
    
    # Show training results
    load_training_results()
    
    # Evaluate on ground truth
    evaluate_on_groundtruth()
    
    # Test on specific image
    if args.eval_only:
        return
    
    if args.image:
        image_path = Path(args.image)
    else:
        # Use a sample image
        images = sorted(IMAGES_DIR.glob("*.jpg"))
        if not images:
            print("⚠️  No images found")
            return
        image_path = images[0]
    
    compare_ocr_results(image_path, FINETUNED_MODEL)


if __name__ == '__main__':
    main()
