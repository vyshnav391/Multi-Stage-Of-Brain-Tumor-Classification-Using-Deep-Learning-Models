"""
feature_extractor.py
--------------------
Handcrafted feature engineering from MRI images.

Feature groups:
  (a) Statistical       — mean, std, skewness, kurtosis, entropy
  (b) Gradient/Texture  — Sobel mean & std
  (c) Edge              — Canny density, mean, std
  (d) Frequency         — FFT mean, std, high-frequency ratio
  (e) Morphological     — contour area, perimeter, circularity
  (f) Hu Moments        — 7 invariant shape descriptors
"""

import cv2
import numpy as np


def extract_handcrafted_features(img_path: str) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a single MRI image.

    Args:
        img_path: Path to the image file.

    Returns:
        1-D float32 numpy array of length ~23.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    features = {}

    # ── (a) Statistical ────────────────────────────────────────────
    features["mean"] = float(np.mean(gray))
    features["std"] = float(np.std(gray))
    normed = (gray - features["mean"]) / (features["std"] + 1e-8)
    features["skewness"] = float(np.mean(normed ** 3))
    features["kurtosis"] = float(np.mean(normed ** 4))
    hist, _ = np.histogram(gray, bins=256, density=True)
    features["entropy"] = float(-np.sum(hist * np.log2(hist + 1e-8)))

    # ── (b) Gradient / Texture ──────────────────────────────────────
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    features["gradient_mean"] = float(np.mean(magnitude))
    features["gradient_std"] = float(np.std(magnitude))

    # ── (c) Edge (Canny) ────────────────────────────────────────────
    edges = cv2.Canny(gray, 50, 150)
    features["edge_density"] = float(np.sum(edges > 0) / edges.size)
    features["edge_mean"] = float(np.mean(edges))
    features["edge_std"] = float(np.std(edges))

    # ── (d) Frequency (FFT) ─────────────────────────────────────────
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    features["fft_mean"] = float(np.mean(fft_mag))
    features["fft_std"] = float(np.std(fft_mag))
    features["fft_high_ratio"] = float(
        np.sum(fft_mag > np.percentile(fft_mag, 90)) / fft_mag.size
    )

    # ── (e) Morphological ───────────────────────────────────────────
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perim = cv2.arcLength(largest, True)
        circularity = (4 * np.pi * area) / (perim ** 2 + 1e-8)
    else:
        area, perim, circularity = 0.0, 0.0, 0.0
    features["contour_area"] = float(area)
    features["contour_perimeter"] = float(perim)
    features["circularity"] = float(circularity)

    # ── (f) Hu Moments ──────────────────────────────────────────────
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    for i, h in enumerate(hu):
        features[f"hu_{i}"] = float(-np.sign(h) * np.log10(abs(h) + 1e-8))

    return np.array(list(features.values()), dtype=np.float32)


# Friendly display names aligned with extract order
FEATURE_NAMES = [
    "Mean", "Std", "Skewness", "Kurtosis", "Entropy",
    "Gradient Mean", "Gradient Std",
    "Edge Density", "Edge Mean", "Edge Std",
    "FFT Mean", "FFT Std", "FFT High Ratio",
    "Contour Area", "Contour Perimeter", "Circularity",
    "Hu_0", "Hu_1", "Hu_2", "Hu_3", "Hu_4", "Hu_5", "Hu_6",
]
