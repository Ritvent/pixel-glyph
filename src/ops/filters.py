"""Exer 8 — Smoothing (mean, median).
Exer 9 — Sharpening (Laplacian, unsharp mask) + Gradient edges (Sobel, Prewitt).
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Exer 8 — Smoothing
# ---------------------------------------------------------------------------

def mean_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = max(1, int(ksize))
    return cv2.blur(img, (ksize, ksize))


def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)


def gaussian_filter(img: np.ndarray, ksize: int = 3, sigma: float = 0.0) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


# ---------------------------------------------------------------------------
# Exer 9 — Sharpening
# ---------------------------------------------------------------------------

def laplacian_sharpen(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """High-boost via Laplacian: out = img - amount * laplacian(img)."""
    f = img.astype(np.float32)
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)
    out = f - amount * lap
    return np.clip(out, 0, 255).astype(np.uint8)


def unsharp_mask(img: np.ndarray, amount: float = 1.0, radius: float = 1.0) -> np.ndarray:
    """Classic unsharp mask: img + amount * (img - gaussian_blur(img))."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=max(radius, 0.1))
    out = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return out


# ---------------------------------------------------------------------------
# Exer 9 — Gradient edges
# ---------------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _normalize_mag(mag: np.ndarray) -> np.ndarray:
    m = mag.max()
    if m <= 0:
        return np.zeros_like(mag, dtype=np.uint8)
    return np.clip(mag / m * 255.0, 0, 255).astype(np.uint8)


def sobel(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = _to_gray(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    return _normalize_mag(np.sqrt(gx * gx + gy * gy))


def prewitt(img: np.ndarray) -> np.ndarray:
    gray = _to_gray(img).astype(np.float32)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    return _normalize_mag(np.sqrt(gx * gx + gy * gy))


def roberts(img: np.ndarray) -> np.ndarray:
    gray = _to_gray(img).astype(np.float32)
    kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    return _normalize_mag(np.sqrt(gx * gx + gy * gy))
