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


def preprocess_for_ocr(image_path: str | Path) -> np.ndarray:
    """Single best-quality output: load → deskew → denoise → contrast."""
    img = load_image(image_path)
    img = deskew(img)
    img = denoise(img)
    return enhance_contrast(img)
