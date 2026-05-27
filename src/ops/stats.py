"""Exer 7 — Mean, standard deviation, correlation coefficient.

Operates on a full image or a (x, y, w, h) ROI rectangle.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


Roi = Optional[tuple[int, int, int, int]]


def _crop(img: np.ndarray, roi: Roi) -> np.ndarray:
    if roi is None:
        return img
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


def compute_stats(img: np.ndarray, roi: Roi = None) -> dict:
    """Return a dictionary of basic statistics for the region."""
    region = _crop(img, roi)
    if region.size == 0:
        return {"error": "empty region"}

    out: dict = {
        "shape": tuple(region.shape),
        "pixels": int(region.size if region.ndim == 2 else region.shape[0] * region.shape[1]),
    }

    if region.ndim == 2:
        out["mean"] = float(region.mean())
        out["std"] = float(region.std())
        out["min"] = int(region.min())
        out["max"] = int(region.max())
        return out

    # Color: per-channel stats (BGR order)
    mean_bgr = region.reshape(-1, 3).mean(axis=0)
    std_bgr = region.reshape(-1, 3).std(axis=0)
    out["mean_bgr"] = [float(v) for v in mean_bgr]
    out["std_bgr"] = [float(v) for v in std_bgr]
    # Combined luminance stats too
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    out["mean_luma"] = float(gray.mean())
    out["std_luma"] = float(gray.std())
    out["min_luma"] = int(gray.min())
    out["max_luma"] = int(gray.max())
    return out


def correlation_coefficient(img_a: np.ndarray,
                            img_b: np.ndarray,
                            roi: Roi = None) -> float:
    """Pearson correlation between two images (must share shape after crop).

    Color images are flattened to gray luminance for a single scalar."""
    a = _crop(img_a, roi)
    b = _crop(img_b, roi)
    if a.shape[:2] != b.shape[:2]:
        # Resize b to match a — typical use is "original vs working"; if
        # the user has resized, fall back to comparing scaled versions.
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    af -= af.mean()
    bf -= bf.mean()
    denom = np.sqrt((af * af).sum() * (bf * bf).sum())
    if denom == 0:
        return 0.0
    return float((af * bf).sum() / denom)


def mse(a: np.ndarray, b: np.ndarray, roi: Roi = None) -> float:
    """Mean squared error between two same-shape images."""
    ra, rb = _crop(a, roi), _crop(b, roi)
    if ra.shape != rb.shape:
        raise ValueError(f"shape mismatch: {ra.shape} vs {rb.shape}")
    diff = ra.astype(np.float64) - rb.astype(np.float64)
    return float((diff * diff).mean())


def psnr(a: np.ndarray, b: np.ndarray, roi: Roi = None) -> float:
    """Peak signal-to-noise ratio (dB). 'inf' if identical."""
    m = mse(a, b, roi)
    if m == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(m)))


def diff_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Binary 0/255 mask where the two images differ (any channel)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    d = (a != b)
    if d.ndim == 3:
        d = d.any(axis=2)
    return (d.astype(np.uint8) * 255)


def diff_amplified(a: np.ndarray, b: np.ndarray, gain: float = 50.0) -> np.ndarray:
    """|a - b| * gain, clipped to uint8 — makes invisible LSB diffs visible."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)) * float(gain)
    return np.clip(d, 0, 255).astype(np.uint8)


def compare_stats(cover: np.ndarray, working: np.ndarray) -> dict:
    """Cover-vs-working summary used by the Analyze tab."""
    if cover.shape != working.shape:
        return {
            "error": f"shape mismatch: cover {cover.shape} vs working {working.shape} — "
                     "an attack changed image dimensions (rotate/scale)",
        }
    m = mse(cover, working)
    p = psnr(cover, working)
    r = correlation_coefficient(cover, working)
    d = (cover != working)
    if d.ndim == 3:
        d = d.any(axis=2)
    changed = int(d.sum())
    total = int(d.size)
    max_diff = int(np.abs(cover.astype(np.int16) - working.astype(np.int16)).max())
    return {
        "shape": tuple(cover.shape),
        "mse": m,
        "psnr_db": p,
        "correlation": r,
        "pixels_changed": changed,
        "pixels_total": total,
        "pct_changed": changed / total * 100.0 if total else 0.0,
        "max_pixel_diff": max_diff,
    }


def format_stats(stats: dict) -> str:
    """Human-readable formatting for display in a dialog."""
    lines = []
    for k, v in stats.items():
        if isinstance(v, float):
            lines.append(f"{k:>12s} : {v:.3f}")
        elif isinstance(v, list):
            lines.append(f"{k:>12s} : [{', '.join(f'{x:.3f}' for x in v)}]")
        else:
            lines.append(f"{k:>12s} : {v}")
    return "\n".join(lines)
