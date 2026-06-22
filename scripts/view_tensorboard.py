"""
View TensorBoard Logs
======================
Script untuk membuka TensorBoard dan visualize training metrics.

Usage:
    python scripts/view_tensorboard.py
    
Then open browser to: http://localhost:6006
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
log_dir = PROJECT_ROOT / "models" / "classifier_v5_indonesia" / "logs"

if not log_dir.exists():
    print(f"❌ TensorBoard logs not found at: {log_dir}")
    print(f"\n💡 Train the model first:")
    print(f"   python scripts/train_classifier_v5_indonesia.py")
    sys.exit(1)

print("=" * 70)
print("🔍 TENSORBOARD VIEWER")
print("=" * 70)
print(f"\n📂 Log directory: {log_dir}")
print(f"\n🌐 TensorBoard will start on: http://localhost:6006")
print(f"\n💡 Press Ctrl+C to stop TensorBoard\n")

import subprocess
subprocess.run([
    "tensorboard",
    f"--logdir={log_dir}",
    "--port=6006",
    "--host=0.0.0.0"
])
