"""Exer 11 — Image restoration.

In PixelGlyph these double as stego attacks (they erase LSB content) and
as "rescue" attempts on a corrupted stego payload.

Provides:
- Non-local-means denoise
- Median denoise (good for salt-and-pepper)
- Wiener deconvolution with canned motion / Gaussian PSFs (deblur)
- Inpaint helper around an intensity-derived bright-spot mask
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Denoise
# ---------------------------------------------------------------------------

def denoise_nlm(img: np.ndarray, strength: float = 10.0) -> np.ndarray:
    """Non-local means denoise. `strength` ~ 5..20 typical."""
    h = float(max(strength, 0.1))
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, h, 7, 21)
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def denoise_median(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(img, k)


# ---------------------------------------------------------------------------
# PSFs
# ---------------------------------------------------------------------------

def make_motion_psf(length: int = 15, angle_deg: float = 0.0) -> np.ndarray:
    length = max(3, int(length) | 1)  # force odd >= 3
    psf = np.zeros((length, length), dtype=np.float32)
    psf[length // 2, :] = 1.0
    psf /= psf.sum()
    if angle_deg != 0.0:
        M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5),
                                    float(angle_deg), 1.0)
        psf = cv2.warpAffine(psf, M, (length, length))
        s = psf.sum()
        if s > 0:
            psf /= s
    return psf


def make_gaussian_psf(size: int = 15, sigma: float = 3.0) -> np.ndarray:
    size = max(3, int(size) | 1)
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    psf /= psf.sum()
    return psf


# ---------------------------------------------------------------------------
# Wiener deconvolution
# ---------------------------------------------------------------------------

def _wiener_single(channel: np.ndarray, psf: np.ndarray, nsr: float) -> np.ndarray:
    """Wiener filter on a single 2D channel in [0, 1] float."""
    H, W = channel.shape
    psf_pad = np.zeros((H, W), dtype=np.float32)
    ph, pw = psf.shape
    psf_pad[:ph, :pw] = psf
    psf_pad = np.roll(psf_pad, -(ph // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(pw // 2), axis=1)

    PSF_FT = np.fft.fft2(psf_pad)
    IMG_FT = np.fft.fft2(channel)
    # Wiener: F_hat = conj(H) / (|H|^2 + NSR) * G
    denom = (np.abs(PSF_FT) ** 2) + max(nsr, 1e-6)
    F_hat = np.conj(PSF_FT) / denom * IMG_FT
    out = np.real(np.fft.ifft2(F_hat))
    return out


def wiener_deconv(img: np.ndarray, psf: np.ndarray,
                  nsr: float = 0.01) -> np.ndarray:
    """Wiener deconvolution with a known PSF and noise-to-signal ratio."""
    f = img.astype(np.float32) / 255.0
    if f.ndim == 2:
        out = _wiener_single(f, psf, nsr)
    else:
        out = np.zeros_like(f)
        for c in range(f.shape[2]):
            out[:, :, c] = _wiener_single(f[:, :, c], psf, nsr)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Damage inpaint (uses a brightness-derived mask of "very bright dust spots")
# ---------------------------------------------------------------------------

def inpaint_bright_damage(img: np.ndarray,
                          threshold: int = 240,
                          radius: int = 3) -> np.ndarray:
    """Inpaint extreme-bright pixels (treated as damage / dust spots).

    For more controllable masks, pass a pre-computed mask path; this
    helper gives a one-button result based on a brightness threshold.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        target = img
    else:
        gray = img
        target = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    out = cv2.inpaint(target, mask, int(radius), cv2.INPAINT_TELEA)
    if img.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    return out
