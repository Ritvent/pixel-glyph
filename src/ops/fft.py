"""Exer 6 — FFT (1-D & 2-D) of an image."""
from __future__ import annotations

import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def fft_2d_magnitude(img: np.ndarray) -> np.ndarray:
    """log-magnitude of the centered 2D FFT, scaled to uint8 for display."""
    gray = _to_gray(img).astype(np.float32)
    f = np.fft.fft2(gray)
    f = np.fft.fftshift(f)
    mag = np.log1p(np.abs(f))
    m = mag.max()
    if m > 0:
        mag = mag / m * 255.0
    return mag.astype(np.uint8)


def fft_1d_row(img: np.ndarray, row_idx: int) -> np.ndarray:
    """Return |FFT(row)| for a single image row — array for plotting."""
    gray = _to_gray(img).astype(np.float32)
    r = int(row_idx) % gray.shape[0]
    return np.abs(np.fft.fft(gray[r, :]))


def fft_1d_col(img: np.ndarray, col_idx: int) -> np.ndarray:
    gray = _to_gray(img).astype(np.float32)
    c = int(col_idx) % gray.shape[1]
    return np.abs(np.fft.fft(gray[:, c]))
