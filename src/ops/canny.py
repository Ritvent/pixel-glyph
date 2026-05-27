"""Exer 13 — Canny edge detection."""
from __future__ import annotations

import cv2
import numpy as np


def canny(img: np.ndarray, lo: int = 100, hi: int = 200,
          aperture: int = 3) -> np.ndarray:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aperture = aperture if aperture in (3, 5, 7) else 3
    return cv2.Canny(gray, int(lo), int(hi), apertureSize=aperture)


def canny_auto(img: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Threshold-free Canny: derive lo/hi from the image median.

    A common rule of thumb when the user doesn't want to tune sliders.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    med = float(np.median(gray))
    lo = int(max(0, (1.0 - sigma) * med))
    hi = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(gray, lo, hi)
