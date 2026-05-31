"""
OCR FinSight - Image Preprocessing
OpenCV pipeline for receipt images.
"""

import cv2
import numpy as np
from pathlib import Path
from app.config import MAX_IMAGE_SIZE


def load_image(image_path: str | Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def adaptive_threshold(img: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, blockSize=11, C=2)


def denoise(img: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img)
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def sharpen(img: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew(img: np.ndarray) -> np.ndarray:
    """Correct skew using dominant horizontal lines."""
    gray = to_grayscale(img)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=gray.shape[1] // 4, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return img
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 15:
            angles.append(angle)
    if not angles:
        return img
    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


def upscale(img: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Upscale with Lanczos interpolation. EasyOCR is more accurate on larger text."""
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_LANCZOS4)


def bilateral_smooth(img: np.ndarray) -> np.ndarray:
    """Bilateral filter — reduce noise while preserving character edges."""
    return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)


def enhance_contrast_color(img: np.ndarray) -> np.ndarray:
    """CLAHE on LAB L-channel — boosts contrast while keeping color for EasyOCR."""
    if len(img.shape) == 2:
        return enhance_contrast(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess_for_easyocr(img: np.ndarray, *, upscale_factor: float = 2.0) -> np.ndarray:
    """Optimised preprocessing pipeline for EasyOCR (color BGR input).

    Pipeline: deskew → bilateral smooth → CLAHE (L-channel) → light sharpen → upscale 2×
    Unlike binary-threshold approaches, this keeps the color image that EasyOCR prefers.
    """
    if img is None:
        raise ValueError("Image is None")

    img = deskew(img)
    img = bilateral_smooth(img)
    img = enhance_contrast_color(img)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
        sharp_gray = cv2.addWeighted(gray, 1.4, blurred, -0.4, 0)
        sharp_bgr = cv2.cvtColor(sharp_gray, cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 0.6, sharp_bgr, 0.4, 0)

    return upscale(img, factor=upscale_factor)


def preprocess_for_ocr(image_path: str | Path) -> np.ndarray:
    """Legacy path-based preprocessing (grayscale output)."""
    img = load_image(image_path)
    img = deskew(img)
    img = denoise(img)
    return enhance_contrast(img)
