"""Image conversion utilities.

Replaces ``intermediary/run_length_image.py``. Bidirectional conversions
between RLE binary strings, ``numpy.ndarray`` (bool for ONEBIT, uint8 for
DENSE), PIL ``Image``, and base64-encoded payloads suitable for transport to
the frontend.
"""
