"""Statistical stego-detection helpers.

Two classical detectors:

1. **Chi-square test** (Westfeld & Pfitzmann, 1999) — uses Exer 4 (histogram).
   Natural images have unequal counts for adjacent value pairs (2k, 2k+1).
   LSB-replacement stego randomly sets the LSB → it *equalizes* those
   pair counts. The chi-square test asks how suspiciously equal they are.

2. **Neighbor-pair analysis** — uses Exer 2 (pixel relationships).
   For natural images, horizontally adjacent pixels usually share the same
   exact value far more often than they differ by exactly 1. LSB stego
   randomizes the LSB so the (same) and (differ-by-1) counts converge.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import stats as scstats


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# LSB plane helper
# ---------------------------------------------------------------------------

def lsb_plane(img: np.ndarray) -> np.ndarray:
    """Return the LSB plane of the grayscale view of `img` as a 0/255 uint8."""
    gray = _to_gray(img)
    return ((gray & 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Chi-square test (Westfeld-Pfitzmann)
# ---------------------------------------------------------------------------

def chi_square_stego(img: np.ndarray) -> dict:
    """Run the chi-square LSB-stego test on the grayscale view.

    Returns chi2 statistic, degrees of freedom, p-value, and a verdict.

    Interpretation (Westfeld convention):
        p ≈ 1.0   pair counts are *suspiciously equal* → stego likely
        p ≈ 0.0   pair counts look natural → image is probably clean
    """
    gray = _to_gray(img)
    if gray.size < 256:
        return {"error": "image too small for chi-square test"}

    hist, _ = np.histogram(gray, bins=256, range=(0, 256))

    chi2 = 0.0
    df = 0
    for k in range(128):
        n0 = float(hist[2 * k])
        n1 = float(hist[2 * k + 1])
        expected = (n0 + n1) / 2.0
        if expected > 0:
            chi2 += ((n0 - expected) ** 2) / expected
            df += 1
    df = max(df - 1, 1)

    p_value = float(1.0 - scstats.chi2.cdf(chi2, df))

    if p_value > 0.5:
        verdict = "SUSPICIOUS — LSB stego likely"
    elif p_value > 0.05:
        verdict = "BORDERLINE — possible LSB stego"
    else:
        verdict = "CLEAN — no LSB stego detected"

    return {
        "chi2_statistic": chi2,
        "degrees_freedom": df,
        "p_value": p_value,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Neighbor-pair analysis (Exer 2 — pixel relationships)
# ---------------------------------------------------------------------------

def neighbor_pair_stats(img: np.ndarray) -> dict:
    """Horizontal neighbor-pair statistics.

    For each adjacent (a, b) pair walking the raster:
        same       : a == b
        lsb_swap   : a XOR b == 1   (differs only in the LSB)
        other      : everything else
    """
    gray = _to_gray(img).astype(np.int16)
    if gray.shape[1] < 2:
        return {"error": "image too narrow for neighbor-pair test"}

    a = gray[:, :-1].ravel()
    b = gray[:, 1:].ravel()
    same = int((a == b).sum())
    lsb_swap = int(((a ^ b) == 1).sum())
    total = int(a.size)
    other = total - same - lsb_swap
    denom = same + lsb_swap
    ratio = lsb_swap / denom if denom > 0 else 0.0

    if ratio > 0.40:
        verdict = "SUSPICIOUS — LSB-randomization signature"
    elif ratio > 0.20:
        verdict = "BORDERLINE — partial-payload stego possible"
    else:
        verdict = "NATURAL — neighbor structure intact"

    return {
        "pairs_total": total,
        "same": same,
        "lsb_swap": lsb_swap,
        "other": other,
        "ratio": ratio,
        "verdict": verdict,
    }
