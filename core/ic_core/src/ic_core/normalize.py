"""Feature-space glyph normalization (experimental, opt-in).

These transforms operate on a glyph's **binary mask** to make the
feature vector more robust to detector-box framing and edge noise.
They are applied *only* inside feature extraction (see
:func:`ic_core.features.compute_features` together with the
:func:`ic_core.features.feature_normalization` context manager) — the
stored :class:`ic_core.glyph.Glyph` is never mutated, so the frontend
thumbnail (the binary-mask PNG) and the page bounding-box overlay
(``ulx``/``uly``/``ncols``/``nrows``) are unaffected.

Three independent steps, each off by default in :class:`NormalizeConfig`:

* **despeckle** — drop connected components smaller than ``min_size``
  pixels. Kills stray ink specks that would otherwise drag the centre
  of mass around or inflate a bounding box. Mean-robust by design: a
  single speck contributes ``1/N`` to the centroid, but a whole
  component is removed outright here.
* **center** — translate the ink so its centre of mass sits at the
  geometric centre of the array. This is what normalises the 16-d
  ``volume16regions`` grid against where the neume happens to float
  inside a loosely-drawn detector box. Padding is asymmetric — we pad
  the *shorter* side to the centroid so the result is the minimal
  centred frame (we deliberately do **not** resize to a fixed canvas,
  which would zero out the ``nrows``/``ncols``/``aspect_ratio``
  features under standardisation).
* **pad** — add a uniform background border. Mostly a companion to
  ``center`` (room to shift) and a stabiliser for the border-sensitive
  features (``nholes``, ``compactness``): a guaranteed background ring
  keeps the outer background a single connected component.

Threshold stabilisation is intentionally **not** here: binarisation
happens at ingest on the greyscale page, and the :class:`Glyph` only
ever carries the already-binarised mask. Re-thresholding therefore
belongs in :mod:`ic_core.ingest`, not in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

#: 8-connectivity structuring element. Diagonally-touching pixels count
#: as one component, which matches how a human reads a stroke; the
#: 4-connectivity used by ``features._nholes`` is about background
#: bays, a different question.
_CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class NormalizeConfig:
    """Which normalization steps to apply, and with what parameters.

    The default instance is a **no-op** — every step is disabled — so
    ``NormalizeConfig()`` reproduces the raw-mask feature behaviour.
    Enable steps explicitly for an experiment.

    Attributes:
        despeckle_min_size: Drop connected components with fewer than
            this many foreground pixels. ``<= 1`` disables despeckling.
            Keep this small (a few px): real neumes — even a punctum —
            are far larger, so a low value removes only true specks.
        center: Translate the ink centre of mass to the array centre.
        pad: Uniform background border width in pixels. ``<= 0``
            disables padding.
    """

    despeckle_min_size: int = 0
    center: bool = False
    pad: int = 0

    @property
    def is_noop(self) -> bool:
        """True when no step is enabled (raw-mask features)."""
        return self.despeckle_min_size <= 1 and not self.center and self.pad <= 0


def despeckle(mask: np.ndarray, *, min_size: int) -> np.ndarray:
    """Remove foreground components smaller than ``min_size`` pixels.

    If every component is below the threshold, the single **largest**
    component is kept rather than returning an empty mask — an
    all-blank glyph carries no signal and would only ever happen for a
    pathological all-noise crop.

    Args:
        mask: Boolean foreground mask.
        min_size: Minimum component size (pixels) to keep.

    Returns:
        A new boolean mask with small components cleared. Returned
        unchanged when ``min_size <= 1`` or the mask is empty.
    """
    if min_size <= 1 or not mask.any():
        return mask

    labeled, n = label(mask, structure=_CONNECTIVITY_8)
    if n == 0:
        return mask

    # counts[0] is the background; counts[i] is component i's pixel count.
    counts = np.bincount(labeled.ravel())
    keep = counts >= min_size
    keep[0] = False  # never keep background as foreground

    if not keep.any():
        # Everything is below threshold — fall back to the largest
        # component so we don't blank the glyph entirely.
        largest = int(np.argmax(counts[1:])) + 1
        keep[largest] = True

    return keep[labeled]


def center_by_mass(mask: np.ndarray) -> np.ndarray:
    """Pad ``mask`` so the ink centre of mass lands at the array centre.

    Pads the shorter side (top/bottom and left/right independently) up
    to the centroid, producing the minimal frame in which the centroid
    is centred. Size is *not* fixed — the ink's own extent is
    preserved so the size/aspect-ratio features stay meaningful.

    Args:
        mask: Boolean foreground mask.

    Returns:
        A new boolean mask, centred. Returned unchanged when empty.
    """
    if not mask.any():
        return mask

    ys, xs = np.nonzero(mask)
    cy = float(ys.mean())
    cx = float(xs.mean())
    h, w = mask.shape

    # Equalise the centroid's distance to opposite edges by padding the
    # shorter side. dist-to-low-edge = c; dist-to-high-edge = (size-1) - c.
    pad_top = max(0, int(round((h - 1) - 2.0 * cy)))
    pad_bottom = max(0, int(round(2.0 * cy - (h - 1))))
    pad_left = max(0, int(round((w - 1) - 2.0 * cx)))
    pad_right = max(0, int(round(2.0 * cx - (w - 1))))

    if not (pad_top or pad_bottom or pad_left or pad_right):
        return mask
    return np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)))


def pad_mask(mask: np.ndarray, *, pad: int) -> np.ndarray:
    """Add a uniform ``pad``-pixel background border on all four sides.

    Returns the input unchanged when ``pad <= 0``.
    """
    if pad <= 0:
        return mask
    return np.pad(mask, ((pad, pad), (pad, pad)))


def normalize_mask(mask: np.ndarray, cfg: NormalizeConfig) -> np.ndarray:
    """Apply the configured normalization steps in order.

    Order is deliberate: **despeckle → center → pad**. Despeckling
    first means the centroid is computed on clean ink; padding last
    wraps the centred result in a uniform border without disturbing
    the centring.

    Args:
        mask: Boolean foreground mask (e.g. from ``Glyph.to_array``).
        cfg: Which steps to run.

    Returns:
        The normalized mask (the input itself when ``cfg`` is a no-op).
    """
    if cfg.is_noop:
        return mask
    out = mask
    if cfg.despeckle_min_size > 1:
        out = despeckle(out, min_size=cfg.despeckle_min_size)
    if cfg.center:
        out = center_by_mass(out)
    if cfg.pad > 0:
        out = pad_mask(out, pad=cfg.pad)
    return out
