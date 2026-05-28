"""
OCR FinSight - Cross-Platform Setup Script
============================================
Install dependencies dengan urutan yang benar untuk hindari NumPy 2.x conflict.

Usage:
    python setup.py            # Install for current Python env
    python setup.py --gpu      # Install with CUDA support (NVIDIA only)
    python setup.py --check    # Verify installation only

Compatible with:
    - Windows (native Python, no WSL needed)
    - Linux / WSL
    - macOS
    - Docker containers

Python: 3.10, 3.11, 3.12
"""

import sys
import subprocess
import argparse
import platform
from pathlib import Path


def run(cmd: list[str], check: bool = True) -> int:
    """Run subprocess, return exit code."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def pip_install(packages: list[str], extra_args: list[str] = None) -> bool:
    """Install packages via pip."""
    cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade'] + packages
    if extra_args:
        cmd.extend(extra_args)
    return run(cmd) == 0


def check_python_version():
    """Verify Python version compatibility."""
    v = sys.version_info
    if v.major != 3 or v.minor < 10:
        print(f"❌ Python 3.10+ required, you have {v.major}.{v.minor}")
        return False
    if v.minor > 12:
        print(f"⚠️  Python {v.major}.{v.minor} not officially tested. Try 3.10-3.12 if issues arise.")
    print(f"✅ Python {v.major}.{v.minor}.{v.micro}")
    return True


def install_core(use_gpu: bool):
    """Install dependencies in the correct order."""
    print("\n" + "=" * 60)
    print("📦 Step 1: Install NumPy 1.x (must be first)")
    print("=" * 60)
    if not pip_install(['numpy>=1.26.0,<2.0']):
        return False
    
    print("\n" + "=" * 60)
    print("📦 Step 2: Install OpenCV & Pillow")
    print("=" * 60)
    if not pip_install(['opencv-python-headless==4.10.0.84', 'Pillow>=10.0.0,<11.0']):
        return False
    
    print("\n" + "=" * 60)
    print("📦 Step 3: Install PyTorch (for EasyOCR)")
    print("=" * 60)
    if use_gpu:
        # CUDA 12.1 build
        torch_args = [
            '--index-url', 'https://download.pytorch.org/whl/cu121'
        ]
        ok = pip_install(['torch==2.1.2', 'torchvision==0.16.2'], extra_args=torch_args)
    else:
        # CPU build (works everywhere)
        torch_args = [
            '--index-url', 'https://download.pytorch.org/whl/cpu'
        ]
        ok = pip_install(['torch==2.1.2', 'torchvision==0.16.2'], extra_args=torch_args)
    if not ok:
        return False
    
    print("\n" + "=" * 60)
    print("📦 Step 4: Install TensorFlow")
    print("=" * 60)
    if not pip_install(['tensorflow==2.15.1']):
        return False
    
    print("\n" + "=" * 60)
    print("📦 Step 5: Install EasyOCR")
    print("=" * 60)
    if not pip_install(['easyocr==1.7.1']):
        return False
    
    print("\n" + "=" * 60)
    print("📦 Step 6: Install remaining dependencies")
    print("=" * 60)
    rest = [
        'pandas>=2.0.0,<2.3',
        'scikit-learn>=1.3.0,<1.6',
        'flask>=3.0.0,<4.0',
        'fastapi>=0.104.0,<0.120',
        'uvicorn[standard]>=0.24.0,<0.35',
        'python-multipart>=0.0.6',
        'jinja2>=3.1.0',
        'matplotlib>=3.7.0,<4.0',
        'tqdm>=4.65.0',
        'requests>=2.31.0',
    ]
    if not pip_install(rest):
        return False
    
    return True


def install_dev():
    """Install optional dev tools (for ground-truth generation)."""
    print("\n" + "=" * 60)
    print("📦 Optional: Install Gemini API (for labeling)")
    print("=" * 60)
    pip_install(['google-generativeai>=0.7.0'])


def verify_installation():
    """Test that all critical imports work."""
    print("\n" + "=" * 60)
    print("🔍 Verifying installation...")
    print("=" * 60)
    
    checks = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('torch', 'PyTorch'),
        ('tensorflow', 'TensorFlow'),
        ('easyocr', 'EasyOCR'),
        ('pandas', 'Pandas'),
        ('flask', 'Flask'),
    ]
    
    all_ok = True
    for module, name in checks:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', '?')
            print(f"  ✅ {name:<15} {version}")
        except ImportError as e:
            print(f"  ❌ {name:<15} FAILED: {e}")
            all_ok = False
        except Exception as e:
            print(f"  ⚠️  {name:<15} loaded with warning: {e}")
    
    # Specific compatibility check: numpy < 2.0
    try:
        import numpy
        v = numpy.__version__.split('.')
        if int(v[0]) >= 2:
            print(f"\n❌ NumPy {numpy.__version__} detected — must be < 2.0")
            print("   Run: pip install 'numpy<2.0' --force-reinstall")
            all_ok = False
    except Exception:
        pass
    
    return all_ok


def print_environment_info():
    """Print system info to help debugging."""
    print("\n" + "=" * 60)
    print("🖥️  Environment Info")
    print("=" * 60)
    print(f"  OS:        {platform.system()} {platform.release()}")
    print(f"  Arch:      {platform.machine()}")
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  Pip:       {subprocess.check_output([sys.executable, '-m', 'pip', '--version']).decode().strip()}")


def main():
    parser = argparse.ArgumentParser(description='OCR FinSight setup')
    parser.add_argument('--gpu', action='store_true', help='Install GPU support (CUDA 12.1)')
    parser.add_argument('--dev', action='store_true', help='Also install dev tools (Gemini API)')
    parser.add_argument('--check', action='store_true', help='Only verify existing installation')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧾 OCR FinSight - Setup")
    print("=" * 60)
    
    print_environment_info()
    
    if not check_python_version():
        sys.exit(1)
    
    if args.check:
        ok = verify_installation()
        sys.exit(0 if ok else 1)
    
    # Upgrade pip first
    print("\n📦 Upgrading pip...")
    pip_install(['pip', 'setuptools', 'wheel'])
    
    # Install in correct order
    mode = "GPU (CUDA 12.1)" if args.gpu else "CPU (cross-platform)"
    print(f"\n🚀 Installing dependencies — Mode: {mode}")
    
    if not install_core(use_gpu=args.gpu):
        print("\n❌ Installation failed")
        sys.exit(1)
    
    if args.dev:
        install_dev()
    
    # Verify
    if not verify_installation():
        print("\n⚠️  Some packages failed to load — check errors above")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run app:  python web/simple_app.py")
    print("  2. Open:     http://localhost:5000")
    print()


if __name__ == '__main__':
    main()
