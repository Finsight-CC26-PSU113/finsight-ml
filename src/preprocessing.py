"""
OCR FinSight - Image Preprocessing
OpenCV-based preprocessing pipeline untuk gambar struk.
Menghasilkan beberapa varian preprocessed image untuk OCR terbaik.
"""

import cv2
import numpy as np
from pathlib import Path
from src.config import MAX_IMAGE_SIZE


def load_image(image_path: str | Path) -> np.ndarray:
    """Load gambar dan resize jika terlalu besar.
    
    Args:
        image_path: Path ke file gambar.
        
    Returns:
        BGR image array.
        
    Raises:
        FileNotFoundError: Jika file tidak ditemukan.
        ValueError: Jika file bukan gambar valid.
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Resize jika terlalu besar (jaga aspect ratio)
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR ke grayscale.
    
    Jika sudah grayscale, return as-is.
    """
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def adaptive_threshold(img: np.ndarray) -> np.ndarray:
    """Adaptive thresholding untuk meningkatkan kontras teks.
    
    Cocok untuk struk dengan pencahayaan tidak merata.
    """
    gray = to_grayscale(img)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )


def denoise(img: np.ndarray) -> np.ndarray:
    """Denoise gambar menggunakan Non-Local Means.
    
    Efektif untuk struk foto yang noisy.
    """
    gray = to_grayscale(img)
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def sharpen(img: np.ndarray) -> np.ndarray:
    """Sharpen gambar untuk teks yang blur.
    
    Menggunakan unsharp masking.
    """
    gray = to_grayscale(img)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return sharpened


def deskew(img: np.ndarray) -> np.ndarray:
    """Koreksi kemiringan struk.
    
    Mendeteksi garis horizontal dominan dan merotasi gambar.
    """
    gray = to_grayscale(img)
    
    # Deteksi edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Deteksi garis dengan HoughLines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 4,  # min 25% width
        maxLineGap=10
    )
    
    if lines is None or len(lines) == 0:
        return img  # Tidak ada garis terdeteksi, return as-is
    
    # Hitung rata-rata sudut kemiringan
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Hanya ambil sudut kecil (< 15 derajat) — artinya garis horizontal
        if abs(angle) < 15:
            angles.append(angle)
    
    if not angles:
        return img
    
    median_angle = np.median(angles)
    
    # Skip rotasi jika sudah hampir lurus
    if abs(median_angle) < 0.5:
        return img
    
    # Rotasi gambar
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        img, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Meningkatkan kontras lokal — bagus untuk struk dengan background gelap.
    """
    gray = to_grayscale(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_pipeline(image_path: str | Path) -> dict[str, np.ndarray]:
    """Jalankan full preprocessing pipeline.
    
    Menghasilkan beberapa varian gambar yang sudah dipreprocess.
    EasyOCR nanti akan pilih hasil terbaik.
    
    Args:
        image_path: Path ke file gambar struk.
        
    Returns:
        Dict berisi varian preprocessed images:
        - 'original': gambar asli (resized + deskewed)
        - 'grayscale_thresh': grayscale + adaptive threshold
        - 'denoise': denoised + grayscale
        - 'sharpen': sharpened + grayscale
    """
    # Load dan deskew
    img = load_image(image_path)
    img = deskew(img)
    
    variants = {
        'original': img,
        'grayscale_thresh': adaptive_threshold(img),
        'denoise': denoise(img),
        'sharpen': sharpen(img),
    }
    
    return variants


def preprocess_for_ocr(image_path: str | Path) -> np.ndarray:
    """Preprocessing sederhana — satu output terbaik untuk OCR.
    
    Pipeline: load → deskew → denoise → enhance contrast
    
    Args:
        image_path: Path ke file gambar struk.
        
    Returns:
        Single preprocessed grayscale image.
    """
    img = load_image(image_path)
    img = deskew(img)
    img = denoise(img)
    img = enhance_contrast(img)
    return img
