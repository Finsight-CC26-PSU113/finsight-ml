"""
Download Pre-trained Models for OCR FinSight
=============================================
Automatically downloads classifier models from Google Drive.

Usage:
    python scripts/download_models.py

Requirements:
    pip install gdown
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Model configurations
MODELS = {
    "classifier_v5_indonesia": {
        "file_id": "1ubTfq-frSFCrQtbbyhsF6e_Ehrzf1JAF",  # Google Drive File ID
        "size": "1.06 MB",
        "description": "V5 Indonesia-only Context-Aware Classifier (86.26% acc)",
        "path": MODELS_DIR / "classifier_v5_indonesia" / "best_weights.weights.h5"
    }
}

def download_from_gdrive(file_id: str, dest: Path, description: str):
    """Download file from Google Drive with progress bar using gdown."""
    print(f"\n📦 Downloading {description}...")
    print(f"   Google Drive File ID: {file_id}")
    print(f"   Destination: {dest}")
    
    # Create parent directory
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if gdown is installed
    try:
        import gdown
    except ImportError:
        print("\n   ❌ gdown not installed!")
        print("   Install it with: pip install gdown")
        return False
    
    try:
        # Google Drive download URL
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # Download with gdown (has built-in progress bar)
        gdown.download(url, str(dest), quiet=False)
        
        # Verify file exists and has content
        if dest.exists() and dest.stat().st_size > 0:
            print(f"   ✅ Downloaded successfully! ({dest.stat().st_size / 1024 / 1024:.2f} MB)")
            return True
        else:
            print(f"   ❌ Download failed or file is empty")
            return False
            
    except Exception as e:
        print(f"\n   ❌ Failed: {e}")
        print("\n   💡 If you see 'Access denied' or 'Quota exceeded':")
        print("      1. Make sure file sharing is set to 'Anyone with the link'")
        print("      2. Try downloading manually from Google Drive")
        print("      3. Place the file in: {dest}")
        return False

def main():
    print("=" * 70)
    print("OCR FinSight - Model Downloader")
    print("=" * 70)
    
    # Check existing models
    existing = []
    missing = []
    
    for model_name, model_info in MODELS.items():
        if model_info["path"].exists():
            existing.append(model_name)
        else:
            missing.append(model_name)
    
    if existing:
        print(f"\n✅ Already installed ({len(existing)}):")
        for name in existing:
            print(f"   - {name}: {MODELS[name]['path']}")
    
    if not missing:
        print("\n🎉 All models are already downloaded!")
        print("\nYou can start the server now:")
        print("   python -m uvicorn web.api_v2:app --port 8000")
        return
    
    print(f"\n📥 Models to download ({len(missing)}):")
    for name in missing:
        info = MODELS[name]
        print(f"   - {name}")
        print(f"     {info['description']}")
        print(f"     Size: {info['size']}")
    
    # Confirm download
    response = input("\n❓ Download missing models? (y/n): ")
    if response.lower() != 'y':
        print("❌ Download cancelled.")
        return
    
    # Download each missing model
    success_count = 0
    for name in missing:
        info = MODELS[name]
        if download_from_gdrive(info["file_id"], info["path"], name):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"✅ Successful: {success_count}/{len(missing)}")
    print(f"❌ Failed: {len(missing) - success_count}/{len(missing)}")
    
    if success_count == len(missing):
        print("\n🎉 All models downloaded successfully!")
        print("\nNext steps:")
        print("   1. Verify installation:")
        print("      python scripts/test_pipeline.py --image test_receipts/sample1.jpg")
        print("   2. Start the server:")
        print("      python -m uvicorn web.api_v2:app --port 8000")
        print("   3. Open http://localhost:8000 in your browser")
    else:
        print("\n⚠️  Some downloads failed. Please try again or download manually from:")
        print("   Google Drive: https://drive.google.com/drive/folders/YOUR_FOLDER_ID")
        print("   Hugging Face: https://huggingface.co/yourusername/ocr-finsight-v5")

if __name__ == "__main__":
    main()
