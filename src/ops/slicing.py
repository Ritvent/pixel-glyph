"""Exer 12 — Intensity-level slicing."""
from __future__ import annotations

import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def intensity_slice(img: np.ndarray,
                    lo: int,
                    hi: int,
                    mode: str = "binary") -> np.ndarray:
    """Highlight pixels whose intensity falls in [lo, hi].

    mode='binary'   -> pixels in range -> 255, others -> 0
    mode='preserve' -> pixels in range -> 255, others keep their value
    """
    lo, hi = int(lo), int(hi)
    if hi < lo:
        lo, hi = hi, lo

    gray = _to_gray(img)
    mask = (gray >= lo) & (gray <= hi)

    if mode == "binary":
        return np.where(mask, 255, 0).astype(np.uint8)

    out = gray.copy()
    out[mask] = 255
    return out


def color_overlay_slice(img: np.ndarray,
                        lo: int,
                        hi: int,
                        color_bgr: tuple[int, int, int] = (0, 0, 255)
                        ) -> np.ndarray:
    """Overlay a color tint on pixels in [lo, hi]; keep the original image
    elsewhere. In PixelGlyph this is used to highlight intensity bands of
    interest when comparing cover vs working images."""
    gray = _to_gray(img)
    mask = (gray >= int(lo)) & (gray <= int(hi))

    base = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = base.copy()
    out[mask] = color_bgr
    return out
