"""
Step 2: Fine-tune EasyOCR Recognition Model (CRNN)
====================================================
Script ini fine-tune model recognition EasyOCR (latin_g2.pth)
menggunakan dataset ground truth yang dibuat oleh Gemini.

Architecture EasyOCR Recognition:
    Input (grayscale 32xW) 
    → ResNet Feature Extractor
    → BiLSTM Sequence Modeling  
    → CTC Loss (Connectionist Temporal Classification)

Usage:
    wsl python3 scripts/finetune_easyocr.py
    wsl python3 scripts/finetune_easyocr.py --epochs 20 --batch-size 32

Output:
    models/finetuned_easyocr/
        best_model.pth       ← best checkpoint
        last_model.pth       ← last epoch
        training_log.json    ← loss/accuracy history
"""

import os
import sys
import json
import time
import argparse
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ROOT_DIR

# ============================================================
# Paths
# ============================================================
GROUNDTRUTH_DIR = ROOT_DIR / "data" / "ocr_groundtruth"
LABELS_FILE = GROUNDTRUTH_DIR / "labels.txt"
CROPS_DIR = GROUNDTRUTH_DIR / "images"
OUTPUT_DIR = ROOT_DIR / "models" / "finetuned_easyocr"

# EasyOCR pretrained model
EASYOCR_MODEL_DIR = Path.home() / ".EasyOCR" / "model"
PRETRAINED_MODEL = EASYOCR_MODEL_DIR / "latin_g2.pth"

# ============================================================
# Character Set (EasyOCR latin_g2 charset)
# ============================================================
# Standard printable ASCII + common receipt characters
CHARSET = (
    string.ascii_letters +      # a-z A-Z
    string.digits +             # 0-9
    string.punctuation +        # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    " "                         # space
)
# Add special chars common in Malaysian/Indonesian receipts
CHARSET += "RM"  # already in uppercase but ensure
CHARSET = "".join(sorted(set(CHARSET)))  # deduplicate

# CTC blank token index = 0
BLANK_IDX = 0
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank

print(f"Charset size: {len(CHARSET)} chars → {NUM_CLASSES} classes (with blank)")


# ============================================================
# Dataset
# ============================================================

class ReceiptCropDataset(data.Dataset):
    """Dataset of cropped receipt text lines with ground truth."""
    
    def __init__(self, labels_file: Path, crops_dir: Path, 
                 img_height: int = 32, img_width: int = 256,
                 augment: bool = False):
        self.crops_dir = crops_dir
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        
        # Load labels
        self.samples = []
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' not in line:
                    continue
                img_rel_path, text = line.split('\t', 1)
                img_path = crops_dir.parent / img_rel_path
                
                # Filter: skip if text has chars not in charset
                filtered_text = ''.join(c for c in text if c in CHAR_TO_IDX)
                if len(filtered_text) < 1:
                    continue
                
                if img_path.exists():
                    self.samples.append((img_path, filtered_text))
        
        print(f"  Loaded {len(self.samples)} samples from {labels_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        
        # Load grayscale crop
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return blank if image missing
            img = np.ones((self.img_height, self.img_width), dtype=np.uint8) * 255
        
        # Resize to fixed height, variable width (pad to max_width)
        h, w = img.shape
        new_w = max(int(w * self.img_height / h), 1)
        img = cv2.resize(img, (new_w, self.img_height), interpolation=cv2.INTER_CUBIC)
        
        # Augmentation (training only)
        if self.augment:
            img = self._augment(img)
        
        # Pad or crop to fixed width
        if img.shape[1] < self.img_width:
            pad = np.ones((self.img_height, self.img_width - img.shape[1]), dtype=np.uint8) * 255
            img = np.concatenate([img, pad], axis=1)
        else:
            img = img[:, :self.img_width]
        
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.FloatTensor(img).unsqueeze(0)  # [1, H, W]
        
        # Encode text to indices
        label = torch.LongTensor([CHAR_TO_IDX.get(c, 1) for c in text])
        
        return img, label, text
    
    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Light augmentation for training."""
        # Random brightness
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img.astype(float) * factor, 0, 255).astype(np.uint8)
        
        # Random noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Random blur
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img


def collate_fn(batch):
    """Custom collate for variable-length labels."""
    images, labels, texts = zip(*batch)
    images = torch.stack(images, 0)
    label_lengths = torch.LongTensor([len(l) for l in labels])
    labels_concat = torch.cat(labels, 0)
    return images, labels_concat, label_lengths, texts


# ============================================================
# Model (EasyOCR CRNN Architecture)
# ============================================================

def load_easyocr_model(pretrained_path: Path, num_classes: int, device: torch.device,
                        load_pretrained_charset: bool = True):
    """Load EasyOCR latin_g2 recognition model architecture and pretrained weights.
    
    EasyOCR latin_g2 architecture:
    - Feature: VGG (NOT ResNet) - output_channel=256
    - Sequence: 2x BidirectionalLSTM, hidden_size=256
    - Prediction: Linear(256, 352) for original charset
    
    Strategy: Load pretrained CNN+LSTM, replace final layer for our charset.
    """
    try:
        import easyocr
        easyocr_dir = Path(easyocr.__file__).parent
        sys.path.insert(0, str(easyocr_dir))
        # Use VGG model (matches latin_g2.pth weights)
        from model.vgg_model import Model as EasyOCRModel
        
        # CORRECT config matching latin_g2.pth
        # VGG output 256, hidden 256, original num_class 352
        ORIGINAL_NUM_CLASSES = 352
        
        # Build with original num_class first to load pretrained
        model = EasyOCRModel(
            input_channel=1,
            output_channel=256,  # VGG outputs 256 channels
            hidden_size=256,
            num_class=ORIGINAL_NUM_CLASSES if load_pretrained_charset else num_classes
        )
        
        # Load pretrained weights
        if pretrained_path.exists():
            print(f"  Loading pretrained: {pretrained_path.name}")
            state_dict = torch.load(str(pretrained_path), map_location=device)
            
            # Handle DataParallel wrapper (module. prefix)
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            
            if not missing and not unexpected:
                print(f"  ✅ Pretrained loaded perfectly")
            else:
                print(f"  ⚠️  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
            # Now REPLACE the final Prediction layer for our smaller charset
            if load_pretrained_charset:
                old_prediction = model.Prediction
                # Initialize new layer with random weights for our num_classes
                model.Prediction = torch.nn.Linear(256, num_classes)
                # Optionally: copy weights for chars that exist in both charsets
                # (For simplicity, we just train from scratch on the prediction layer)
                print(f"  🔄 Replaced final layer: {ORIGINAL_NUM_CLASSES} → {num_classes} classes")
        else:
            print(f"  ⚠️  Pretrained not found, training from scratch")
        
        return model.to(device)
        
    except Exception as e:
        print(f"  ⚠️  Cannot load EasyOCR model: {e}")
        import traceback
        traceback.print_exc()
        print(f"  ℹ️  Using simplified CRNN")
        return build_simple_crnn(num_classes).to(device)


def build_simple_crnn(num_classes: int) -> nn.Module:
    """Fallback: Simple CRNN if EasyOCR model import fails."""
    
    class SimpleCRNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            # CNN feature extractor
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1)),
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1)),
                nn.Conv2d(512, 512, 2), nn.ReLU(),
            )
            # RNN sequence modeling
            self.rnn = nn.Sequential(
                nn.LSTM(512, 256, bidirectional=True, batch_first=True),
            )
            self.fc = nn.Linear(512, num_classes)
        
        def forward(self, x, text=None):
            # CNN
            features = self.cnn(x)  # [B, C, H, W]
            b, c, h, w = features.size()
            features = features.squeeze(2)  # [B, C, W]
            features = features.permute(0, 2, 1)  # [B, W, C]
            # RNN
            rnn_out, _ = self.rnn[0](features)
            # FC
            output = self.fc(rnn_out)  # [B, W, num_classes]
            return output
    
    return SimpleCRNN(num_classes)


# ============================================================
# CTC Decoder
# ============================================================

def ctc_decode(output: torch.Tensor) -> list[str]:
    """Greedy CTC decode.
    
    Args:
        output: [T, B, num_classes] log probabilities
        
    Returns:
        List of decoded strings
    """
    # Get best path
    _, max_indices = output.max(2)  # [T, B]
    max_indices = max_indices.transpose(0, 1)  # [B, T]
    
    decoded = []
    for seq in max_indices:
        chars = []
        prev = BLANK_IDX
        for idx in seq.tolist():
            if idx != BLANK_IDX and idx != prev:
                char = IDX_TO_CHAR.get(idx, '')
                if char:
                    chars.append(char)
            prev = idx
        decoded.append(''.join(chars))
    
    return decoded


def compute_cer(pred: str, target: str) -> float:
    """Character Error Rate."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    
    # Levenshtein distance
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / len(target)


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    """Train one epoch with live progress bar."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", 
                ncols=110, leave=False, dynamic_ncols=False)
    
    for batch_idx, (images, labels, label_lengths, texts) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        try:
            output = model(images, labels)  # [B, T, num_classes]
        except TypeError:
            output = model(images)
        
        # CTC Loss expects [T, B, C]
        output = output.permute(1, 0, 2)
        output = output.log_softmax(2)
        
        T = output.size(0)
        B = output.size(1)
        input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
        
        loss = criterion(output, labels, input_lengths, label_lengths)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update live stats every batch
        avg_loss = total_loss / num_batches
        gpu_mem = torch.cuda.memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'avg': f'{avg_loss:.3f}',
            'gpu': f'{gpu_mem:.1f}GB'
        })
    
    pbar.close()
    return total_loss / max(num_batches, 1)


def evaluate(model, loader, device, max_samples=200, epoch=0, total_epochs=0):
    """Evaluate model on validation set with progress bar."""
    model.eval()
    total_cer = 0
    exact_matches = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]  ", 
                ncols=110, leave=False)
    
    with torch.no_grad():
        for images, labels, label_lengths, texts in pbar:
            images = images.to(device, non_blocking=True)
            
            try:
                output = model(images, labels.to(device))
            except TypeError:
                output = model(images)
            
            output = output.permute(1, 0, 2).log_softmax(2)
            preds = ctc_decode(output)
            
            for pred, target in zip(preds, texts):
                cer = compute_cer(pred, target)
                total_cer += cer
                if pred == target:
                    exact_matches += 1
                total_samples += 1
            
            avg_cer = total_cer / max(total_samples, 1)
            avg_acc = exact_matches / max(total_samples, 1)
            pbar.set_postfix({'cer': f'{avg_cer:.3f}', 'acc': f'{avg_acc:.3f}', 'n': total_samples})
            
            if total_samples >= max_samples:
                break
    
    pbar.close()
    avg_cer = total_cer / max(total_samples, 1)
    accuracy = exact_matches / max(total_samples, 1)
    
    return avg_cer, accuracy


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Fine-tune EasyOCR recognition model')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs (default: 15)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--img-width', type=int, default=256, help='Input image width (default: 256)')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split (default: 0.1)')
    parser.add_argument('--freeze-cnn', action='store_true', help='Freeze CNN layers, only train RNN+FC')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loader workers (default: 4)')
    args = parser.parse_args()
    
    print("=" * 65)
    print("🔧 EasyOCR Fine-tuning")
    print("=" * 65)
    
    # Check dataset
    if not LABELS_FILE.exists():
        print(f"❌ Labels file not found: {LABELS_FILE}")
        print(f"   Run first: python3 scripts/generate_ocr_groundtruth.py --api-key YOUR_KEY")
        return
    
    # Count samples
    with open(LABELS_FILE, 'r') as f:
        total_samples = sum(1 for line in f if '\t' in line)
    
    print(f"\n📊 Dataset: {total_samples} samples")
    
    if total_samples < 50:
        print(f"⚠️  Very few samples ({total_samples}). Need at least 50 for fine-tuning.")
        print(f"   Run generate_ocr_groundtruth.py with more images first.")
        return
    
    # Device + GPU optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        # Speed up convolutions
        torch.backends.cudnn.benchmark = True
        # Mixed precision boost
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dataset split
    print(f"\n📦 Loading dataset...")
    full_dataset = ReceiptCropDataset(
        LABELS_FILE, CROPS_DIR,
        img_height=32, img_width=args.img_width,
        augment=False
    )
    
    val_size = max(int(len(full_dataset) * args.val_split), 10)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Enable augmentation for training subset
    train_dataset.dataset.augment = True
    
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )
    
    print(f"  Train: {train_size} | Val: {val_size}")
    
    # Load model
    print(f"\n🤖 Loading model...")
    import easyocr  # ensure imported for path detection
    model = load_easyocr_model(PRETRAINED_MODEL, NUM_CLASSES, device)
    
    # Optionally freeze CNN layers (transfer learning style)
    if args.freeze_cnn:
        print("  ❄️  Freezing CNN layers (only training RNN + FC)")
        for name, param in model.named_parameters():
            if 'FeatureExtraction' in name:
                param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    print(f"\n🚀 Training for {args.epochs} epochs...")
    print(f"   LR: {args.lr} | Batch: {args.batch_size} | Width: {args.img_width}")
    print()
    
    best_cer = float('inf')
    training_log = []
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train + Validate
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        val_cer, val_acc = evaluate(model, val_loader, device, max_samples=500,
                                      epoch=epoch, total_epochs=args.epochs)
        
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # One-line summary per epoch
        improvement = ""
        if val_cer < best_cer:
            improvement = " 🎯 NEW BEST!"
        
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Loss: {train_loss:6.4f} | "
              f"CER: {val_cer:.4f} ({(1-val_cer)*100:.1f}% acc) | "
              f"Exact: {val_acc*100:.1f}% | "
              f"LR: {current_lr:.1e} | "
              f"Time: {elapsed:.0f}s{improvement}")
        
        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_cer': round(val_cer, 4),
            'val_accuracy': round(val_acc, 4),
            'learning_rate': current_lr,
            'elapsed_sec': round(elapsed, 1),
        }
        training_log.append(log_entry)
        
        # Save training log
        with open(OUTPUT_DIR / 'training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)
        
        # Save best model
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pth')
        
        # Save last model
        torch.save(model.state_dict(), OUTPUT_DIR / 'last_model.pth')
        
        # LR scheduler
        scheduler.step(val_cer)
    
    # Final summary
    print("=" * 65)
    print("✅ TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Best CER    : {best_cer:.4f} ({(1-best_cer)*100:.1f}% char accuracy)")
    print(f"  Best model  : {OUTPUT_DIR / 'best_model.pth'}")
    print(f"  Training log: {OUTPUT_DIR / 'training_log.json'}")
    print()
    print("📋 Next step: Test the fine-tuned model")
    print(f"   python3 scripts/test_finetuned_easyocr.py")


if __name__ == '__main__':
    main()
