"""Image load / save helpers (OpenCV-backed)."""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as BGR (or BGRA for PNG with alpha). Raises if unreadable."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    # Normalize: keep BGR (drop alpha) for downstream simplicity.
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def save_image(img: np.ndarray, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), img):
        raise IOError(f"Could not write image: {out}")
