"""Exer 4 — Contrast stretching, histogram, histogram equalization."""
from __future__ import annotations

import cv2
import numpy as np


def contrast_stretch(img: np.ndarray,
                     low_pct: float = 2.0,
                     high_pct: float = 98.0) -> np.ndarray:
    """Percentile-based linear contrast stretch.

    Maps the [low_pct, high_pct] percentile range to [0, 255]. Operates
    per-channel for color images so a strong color cast is not introduced.
    """
    f = img.astype(np.float32)
    if img.ndim == 2:
        lo, hi = np.percentile(f, [low_pct, high_pct])
        if hi <= lo:
            return img.copy()
        out = (f - lo) / (hi - lo) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    out = np.zeros_like(f)
    for c in range(f.shape[2]):
        lo, hi = np.percentile(f[:, :, c], [low_pct, high_pct])
        if hi <= lo:
            out[:, :, c] = f[:, :, c]
        else:
            out[:, :, c] = (f[:, :, c] - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def equalize(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization.

    Gray: cv2.equalizeHist directly.
    Color: equalize the Y channel of YCrCb to preserve hue.
    """
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Adaptive (local) histogram equalization — uses small tile-level
    histograms instead of a global one."""
    cla = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    if img.ndim == 2:
        return cla.apply(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cla.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def histogram(img: np.ndarray) -> np.ndarray:
    """Compute the histogram. Returns shape (256,) for gray or (3, 256)
    for color (BGR order)."""
    if img.ndim == 2:
        h, _ = np.histogram(img, bins=256, range=(0, 256))
        return h
    return np.stack([
        np.histogram(img[:, :, c], bins=256, range=(0, 256))[0]
        for c in range(3)
    ])
