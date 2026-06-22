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


def upscale(img: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Upscale gambar dengan interpolasi Lanczos (high quality).

    EasyOCR detector lebih akurat pada teks yang lebih besar (>32px tinggi).
    Struk yang difoto biasanya teks-nya kecil setelah resize ke 1080p.

    Args:
        img: BGR atau grayscale image.
        factor: Faktor pembesaran (default 2.0).
    """
    h, w = img.shape[:2]
    new_w = int(w * factor)
    new_h = int(h * factor)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def bilateral_smooth(img: np.ndarray) -> np.ndarray:
    """Bilateral filter — kurangi noise tapi pertahankan tepi karakter.

    Lebih baik dari Gaussian blur untuk OCR karena tidak melumerkan stroke huruf.
    """
    if len(img.shape) == 3:
        return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
    return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)


def enhance_contrast_color(img: np.ndarray) -> np.ndarray:
    """CLAHE diterapkan pada channel L dari LAB, lalu kembali ke BGR.

    Menjaga warna asli (untuk EasyOCR), tapi tetap menaikkan kontras lokal.
    """
    if len(img.shape) == 2:
        return enhance_contrast(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess_for_ocr(image_path: str | Path) -> np.ndarray:
    """Preprocessing sederhana untuk OCR berbasis path file (legacy).

    Pipeline: load → deskew → denoise → enhance contrast.
    Output: grayscale image. Untuk EasyOCR, gunakan `preprocess_for_easyocr` saja.
    """
    img = load_image(image_path)
    img = deskew(img)
    img = denoise(img)
    img = enhance_contrast(img)
    return img


def preprocess_for_easyocr(img: np.ndarray, *, upscale_factor: float = 2.0) -> np.ndarray:
    """Pipeline preprocessing yang dioptimalkan untuk EasyOCR.

    Berbeda dari `preprocess_for_ocr` (binary threshold), EasyOCR bekerja paling
    baik pada gambar BGR/grayscale dengan kontras yang baik. Threshold binary
    bisa membuat detector salah mengelompokkan box.

    Pipeline:
        1. Deskew (koreksi kemiringan)
        2. Bilateral smoothing (kurangi noise, pertahankan tepi)
        3. CLAHE pada channel L dari LAB (naikan kontras lokal tanpa rusak warna)
        4. Light sharpening (unsharp masking yang dilembutkan)
        5. Upscale 2× dengan Lanczos (teks kecil jadi lebih besar)

    Args:
        img: BGR image (numpy array) — input dari API.
        upscale_factor: Faktor pembesaran (default 2.0).
    """
    if img is None:
        raise ValueError("Image is None")

    # 1. Deskew
    img = deskew(img)

    # 2. Bilateral smoothing — preserve edges
    img = bilateral_smooth(img)

    # 3. CLAHE pada L channel (color preserving)
    img = enhance_contrast_color(img)

    # 4. Light sharpening pada channel grayscale lalu blend kembali ke BGR
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
        sharp_gray = cv2.addWeighted(gray, 1.4, blurred, -0.4, 0)
        # Blend sharpened grayscale ke BGR (jaga warna)
        sharp_bgr = cv2.cvtColor(sharp_gray, cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 0.6, sharp_bgr, 0.4, 0)

    # 5. Upscale untuk detector
    img = upscale(img, factor=upscale_factor)

    return img

