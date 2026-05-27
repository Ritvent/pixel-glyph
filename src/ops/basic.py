"""Exer 1 (display / negative / binary / gray) and Exer 3 (transformations).

Pure functions: each accepts an np.ndarray and returns a new np.ndarray.
No GUI / matplotlib imports here.
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Exer 1 — Negative / Gray / Binary
# ---------------------------------------------------------------------------

def negative(img: np.ndarray) -> np.ndarray:
    """Photographic negative. Works for uint8 and float images."""
    if img.dtype == np.uint8:
        return 255 - img
    return img.max() - img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_binary(img: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Threshold to a binary {0, 255} image. Auto-converts to gray first."""
    gray = to_grayscale(img)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


# ---------------------------------------------------------------------------
# Exer 3 — Geometric & intensity transformations
# ---------------------------------------------------------------------------

def resize(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, M, (w, h), borderValue=border)


def flip(img: np.ndarray, axis: str = "horizontal") -> np.ndarray:
    code = 1 if axis == "horizontal" else 0
    return cv2.flip(img, code)


def translate(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, M, (w, h), borderValue=border)


def log_transform(img: np.ndarray) -> np.ndarray:
    """s = c * log(1 + r), normalized to 0..255."""
    f = img.astype(np.float32)
    c = 255.0 / np.log(1.0 + f.max() + 1e-6)
    out = c * np.log(1.0 + f)
    return np.clip(out, 0, 255).astype(np.uint8)


def gamma_correct(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Power-law transform. gamma < 1 brightens; gamma > 1 darkens."""
    inv = 1.0 / max(gamma, 1e-6)
    table = ((np.arange(256) / 255.0) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(img, table)


if __name__ == "__main__":
    # Quick smoke test: python -m src.ops.basic <image>
    import sys
    from src.utils.io import load_image, save_image

    if len(sys.argv) < 2:
        print("usage: python -m src.ops.basic <image>")
        sys.exit(1)
    src = load_image(sys.argv[1])
    save_image(negative(src), "output/_smoke_negative.png")
    save_image(gamma_correct(src, 0.5), "output/_smoke_gamma05.png")
    print("wrote output/_smoke_negative.png and output/_smoke_gamma05.png")
