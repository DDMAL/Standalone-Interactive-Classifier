"""Image conversion utilities.

Replaces ``intermediary/run_length_image.py``. Bidirectional conversions
between RLE binary strings, ``numpy.ndarray`` (bool for ONEBIT, uint8 for
DENSE), PIL ``Image``, and base64-encoded payloads suitable for transport to
the frontend.

True = black (foreground), False = white (background). RLE strings are
whitespace-separated integer run lengths, row-major, starting with a white
run; if the first pixel is black, a leading 0 is emitted so the
white/black alternation holds.
"""
import base64
from io import BytesIO

import numpy as np
from PIL import Image as PILImage


def rle_to_array(rle: str, width: int, height: int) -> np.ndarray:
    runs = np.array(rle.split(), dtype=np.int64)
    colors = np.zeros(runs.size, dtype=bool)
    colors[1::2] = True
    buf = np.repeat(colors, runs)
    expected = width * height
    if buf.size != expected:
        raise ValueError(
            f"RLE length mismatch: decoded {buf.size} pixels, expected {expected}"
        )
    return buf.reshape((height, width))


def array_to_rle(arr: np.ndarray) -> str:
    flat = np.ascontiguousarray(arr, dtype=bool).reshape(-1)
    if flat.size == 0:
        return ""
    prefix = [0] if flat[0] else []
    boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], boundaries, [flat.size]))
    lengths = np.diff(starts).tolist()
    return " ".join(str(n) for n in prefix + lengths)


def array_to_png_base64(arr: np.ndarray) -> str:
    img = PILImage.fromarray(arr.astype(np.uint8) * 255, mode="L").convert("1")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def grayscale_array_to_png_base64(arr: np.ndarray) -> str:
    """Encode a real-pixel crop as a base64 PNG.

    Unlike :func:`array_to_png_base64`, this does *not* treat the input
    as a boolean foreground mask -- it preserves real pixel intensities
    (texture, shading, and colour), which the SSL feature-extraction
    backend needs and the binary RLE mask cannot provide.

    Accepts either an ``(H, W)`` 8-bit greyscale array or an
    ``(H, W, 3)`` 8-bit RGB array -- the SSL extractor's training data
    is real colour manuscript photographs, so callers should prefer
    passing colour crops (see ``ic_core.ingest``'s ``store_real_crop``)
    rather than converting to greyscale before this call.
    """
    mode = "L" if arr.ndim == 2 else "RGB"
    img = PILImage.fromarray(arr.astype(np.uint8), mode=mode)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def png_base64_to_array(data: str) -> np.ndarray:
    """Inverse of :func:`grayscale_array_to_png_base64`.

    Preserves whatever the encoded PNG actually is -- returns an
    ``(H, W, 3)`` array for a colour crop or ``(H, W)`` for a
    greyscale one, rather than forcing everything to greyscale.
    """
    raw = base64.b64decode(data)
    with PILImage.open(BytesIO(raw)) as img:
        if img.mode == "L":
            return np.asarray(img)
        return np.asarray(img.convert("RGB"))
