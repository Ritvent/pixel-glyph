"""LSB steganography — the heart of PixelGlyph.

Payload layout:
    bits[0..32]   : 32-bit big-endian length of message in bytes (uint32)
    bits[32..]    : message bytes, each unpacked MSB-first into 8 LSB bits

The 32-bit header lets the decoder know when to stop reading. Without it,
arbitrary stop bytes (NULs in the message) would terminate decoding early.

LSB writes happen on the flattened raster of the cover image, walking
through every channel of every pixel in order. For a typical H x W x 3
photo the capacity is `(H * W * 3 - 32) / 8` bytes, which is ~340 KB for
a 1280 x 720 cover.
"""
from __future__ import annotations

import numpy as np


HEADER_BITS: int = 32  # 4-byte big-endian length header


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

def capacity_bits(img: np.ndarray) -> int:
    """Total LSB-storable bits in `img` (i.e. one bit per channel-pixel)."""
    return int(img.size)


def capacity_bytes(img: np.ndarray) -> int:
    """Capacity in payload bytes after the 32-bit length header."""
    return max(0, (capacity_bits(img) - HEADER_BITS) // 8)


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def encode(cover: np.ndarray, message: bytes) -> np.ndarray:
    """Embed `message` into the LSBs of `cover` and return the stego image.

    Raises:
        TypeError if cover isn't uint8.
        ValueError if the message is too large for the cover.
    """
    if cover.dtype != np.uint8:
        raise TypeError("cover image must be uint8")
    if not isinstance(message, (bytes, bytearray)):
        raise TypeError("message must be bytes")

    payload = len(message).to_bytes(4, "big") + bytes(message)
    bit_count = len(payload) * 8

    flat_cover = cover.reshape(-1)
    if bit_count > flat_cover.size:
        raise ValueError(
            f"Message too large: needs {bit_count} bits, "
            f"cover has only {flat_cover.size}"
        )

    # Unpack payload into a uint8 bit array of values {0, 1}, MSB-first per byte
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

    stego_flat = flat_cover.copy()
    # Clear the LSB of the first `bit_count` channel-pixels, then OR our bits in
    stego_flat[:bit_count] = (stego_flat[:bit_count] & np.uint8(0xFE)) | bits
    return stego_flat.reshape(cover.shape)


def encode_text(cover: np.ndarray, text: str,
                encoding: str = "utf-8") -> np.ndarray:
    """Convenience wrapper: encode a Python string."""
    return encode(cover, text.encode(encoding))


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode(stego: np.ndarray) -> bytes:
    """Extract the LSB-embedded message bytes from `stego`.

    Raises ValueError if the header looks invalid (too large for the image).
    """
    flat = stego.reshape(-1)
    if flat.size < HEADER_BITS:
        raise ValueError("image too small to contain a stego header")

    header = np.packbits((flat[:HEADER_BITS] & 1).astype(np.uint8))
    msg_len = int.from_bytes(header.tobytes(), "big")

    needed = HEADER_BITS + msg_len * 8
    if msg_len < 0 or needed > flat.size:
        raise ValueError(
            f"invalid header: claims {msg_len} bytes "
            f"(capacity is {capacity_bytes(stego)})"
        )

    bits = (flat[HEADER_BITS:needed] & 1).astype(np.uint8)
    return bytes(np.packbits(bits))


def decode_text(stego: np.ndarray, encoding: str = "utf-8") -> str:
    """Convenience wrapper: decode + decode to a Python string. Replaces
    bad bytes with U+FFFD instead of raising — so the user sees garbage
    rather than a stack trace when running Decode on a non-stego image."""
    return decode(stego).decode(encoding, errors="replace")
