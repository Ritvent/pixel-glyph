"""Exer 10 — Image compression: DCT, DPCM, Huffman.

The DCT path implements an 8x8 block-DCT codec that keeps only the top
`keep_ratio` coefficients per block and inverse-transforms. The DPCM path
encodes first-order row-neighbor residuals and reports the entropy gain.
The Huffman path builds a Huffman code from the symbol frequencies and
reports average bits/symbol vs the fixed 8-bit baseline — i.e. the
theoretical compression savings (no real bitstream is emitted; honest
simplification for a 1-day study project).
"""
from __future__ import annotations

import heapq

import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def jpeg_attack(img: np.ndarray, quality: int = 75) -> np.ndarray:
    """Re-encode `img` as JPEG at the given quality (1-100) and decode back.

    This is the strongest classical attack against LSB stego — JPEG's
    8x8 block DCT quantization scrambles the spatial domain regardless of
    where the bits were embedded. Even quality=100 (no chroma subsampling
    in OpenCV's encoder) usually destroys an LSB payload because the
    integer DCT round-trip is not bit-exact.
    """
    quality = int(max(1, min(100, quality)))
    ok, buf = cv2.imencode(".jpg", img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    flag = cv2.IMREAD_GRAYSCALE if img.ndim == 2 else cv2.IMREAD_COLOR
    out = cv2.imdecode(buf, flag)
    if out is None:
        raise RuntimeError("JPEG decode failed")
    return out


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB. 'inf' if images are identical."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


# ---------------------------------------------------------------------------
# DCT compression
# ---------------------------------------------------------------------------

def _dct_blockwise(gray: np.ndarray, keep_ratio: float) -> tuple[np.ndarray, int, int]:
    """Apply 8x8 block DCT, keep top-K coefficients per block, return
    (reconstructed, coefs_kept, coefs_total)."""
    h, w = gray.shape
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    padded = np.pad(gray.astype(np.float32) - 128.0,
                    ((0, ph), (0, pw)), mode="edge")
    H, W = padded.shape
    out = np.zeros_like(padded)

    keep = max(1, int(round(64 * float(keep_ratio))))
    kept = 0
    total = 0
    for y in range(0, H, 8):
        for x in range(0, W, 8):
            block = padded[y:y + 8, x:x + 8]
            d = cv2.dct(block)
            flat = np.abs(d).ravel()
            thr = np.partition(flat, 64 - keep)[64 - keep]
            mask = np.abs(d) >= thr
            out[y:y + 8, x:x + 8] = cv2.idct(d * mask)
            kept += int(mask.sum())
            total += 64
    rec = np.clip(out[:h, :w] + 128.0, 0, 255).astype(np.uint8)
    return rec, kept, total


def dct_compress(img: np.ndarray, keep_ratio: float = 0.1) -> np.ndarray:
    """Lossy block-DCT compression — returns the reconstructed image only.
    Color images are processed per-channel."""
    if img.ndim == 2:
        rec, _, _ = _dct_blockwise(img, keep_ratio)
        return rec
    out = np.zeros_like(img)
    for c in range(3):
        rec, _, _ = _dct_blockwise(img[:, :, c], keep_ratio)
        out[:, :, c] = rec
    return out


def dct_stats(img: np.ndarray, keep_ratio: float = 0.1) -> dict:
    gray = _to_gray(img)
    rec, kept, total = _dct_blockwise(gray, keep_ratio)
    return {
        "keep_ratio": float(keep_ratio),
        "coefs_kept": kept,
        "coefs_total": total,
        "coef_compression": total / max(kept, 1),
        "psnr_db": psnr(gray, rec),
    }


# ---------------------------------------------------------------------------
# DPCM (first-order predictive coding)
# ---------------------------------------------------------------------------

def dpcm_residual(img: np.ndarray) -> np.ndarray:
    """Signed residual = pixel - left_neighbor (predictor)."""
    gray = _to_gray(img).astype(np.int16)
    pred = np.zeros_like(gray)
    pred[:, 1:] = gray[:, :-1]
    pred[:, 0] = gray[:, 0]
    return gray - pred


def dpcm_residual_image(img: np.ndarray) -> np.ndarray:
    """Visualize residual as gray (offset +128, clipped)."""
    res = dpcm_residual(img)
    return np.clip(res + 128, 0, 255).astype(np.uint8)


def _entropy(arr: np.ndarray) -> float:
    _, counts = np.unique(arr, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return -float((p * np.log2(p + 1e-20)).sum())


def dpcm_stats(img: np.ndarray) -> dict:
    gray = _to_gray(img)
    res = dpcm_residual(img)
    eo = _entropy(gray.ravel())
    er = _entropy(res.ravel())
    return {
        "original_entropy_bits": eo,
        "residual_entropy_bits": er,
        "entropy_savings_bits": eo - er,
        "savings_pct": (1.0 - er / eo) * 100.0 if eo > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Huffman (theoretical bit savings — no real bitstream)
# ---------------------------------------------------------------------------

def _huffman_codes(freq: dict[int, int]) -> dict[int, str]:
    if not freq:
        return {}
    if len(freq) == 1:
        return {next(iter(freq)): "0"}

    heap: list[tuple[int, int, object]] = []
    counter = 0
    for sym, f in freq.items():
        heapq.heappush(heap, (f, counter, sym))
        counter += 1
    while len(heap) > 1:
        f1, _, a = heapq.heappop(heap)
        f2, _, b = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, counter, (a, b)))
        counter += 1
    _, _, tree = heap[0]

    codes: dict[int, str] = {}

    def walk(node: object, code: str) -> None:
        if isinstance(node, tuple):
            walk(node[0], code + "0")
            walk(node[1], code + "1")
        else:
            codes[int(node)] = code or "0"

    walk(tree, "")
    return codes


def huffman_stats(img: np.ndarray) -> dict:
    gray = _to_gray(img)
    vals, counts = np.unique(gray, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    codes = _huffman_codes(freq)
    total = int(counts.sum())
    avg_bits = sum(len(codes[s]) * f for s, f in freq.items()) / total
    raw_bits = 8.0
    return {
        "unique_symbols": len(vals),
        "avg_bits_per_symbol": avg_bits,
        "fixed_bits_per_symbol": raw_bits,
        "compression_ratio": raw_bits / avg_bits if avg_bits > 0 else float("inf"),
        "savings_pct": (1.0 - avg_bits / raw_bits) * 100.0,
    }
