"""Exer 5 — Bit-plane slicing."""
from __future__ import annotations

import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def bit_plane(img: np.ndarray, k: int) -> np.ndarray:
    """Return the k-th bit plane (0 = LSB, 7 = MSB) as a binary 0/255 image."""
    gray = _to_gray(img)
    k = int(k) & 7
    return (((gray >> k) & 1) * 255).astype(np.uint8)


def bit_planes_grid(img: np.ndarray) -> np.ndarray:
    """Return all 8 bit planes (MSB → LSB) as a 2x4 grid image."""
    gray = _to_gray(img)
    h, w = gray.shape
    grid = np.zeros((h * 2, w * 4), dtype=np.uint8)
    for i in range(8):
        r, c = divmod(i, 4)
        plane = bit_plane(gray, 7 - i)  # top-left is MSB
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = plane
    return grid
