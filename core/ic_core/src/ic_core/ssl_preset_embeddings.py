"""Companion SSL embeddings for built-in training presets.

Historical training presets (``core/data/presets/*.xml``) only ever carry
the binary RLE mask -- their original source page images were never
retained, so the ``ssl_fusion`` classifier backend (which needs real
texture/shading, not a silhouette -- see :mod:`ic_core.ssl_classifier`)
cannot extract features from them directly.

Where a preset's source page(s) *are* still available (verified once,
offline, by matching each glyph's bounding box + mask against candidate
page images), a companion ``<preset-stem>.ssl_embeddings.npz`` file can be
generated: one precomputed SSL feature vector per glyph, in the same
document order :func:`ic_core.io_xml.load_glyphs` produces. This module
attaches those vectors to the loaded :class:`~ic_core.glyph.Glyph` objects
so the ssl_fusion backend can train on that preset without ever needing a
live crop or a fresh model pass over preset data.

Entirely additive: a preset with no companion file is unaffected, and
nothing here is imported by the default kNN path.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image as PILImage

from ic_core.glyph import Glyph
from ic_core.image import grayscale_array_to_png_base64

#: Matching a glyph's decoded RLE mask against a candidate page's binarised
#: crop must agree on at least this fraction of pixels to accept the page as
#: that glyph's real source (see ``generate_hufnagel_ssl_embeddings.py``,
#: which found exact (1.0) matches for all 557 Hufnagel preset glyphs).
DEFAULT_MATCH_THRESHOLD = 0.98

#: Matches ic_core.ingest.DEFAULT_THRESHOLD -- how a candidate page is
#: binarised before comparing it to a glyph's mask.
DEFAULT_BINARIZE_THRESHOLD = 127

#: Suffix appended to a preset's stem to find its companion embeddings file,
#: e.g. ``Hufnagel.xml`` -> ``Hufnagel.ssl_embeddings.npz``.
EMBEDDINGS_SUFFIX = ".ssl_embeddings.npz"


def embeddings_path_for(xml_path: Path) -> Path:
    """Return the companion embeddings path for a preset XML file."""
    return xml_path.with_name(xml_path.stem + EMBEDDINGS_SUFFIX)


def has_ssl_embeddings(xml_path: Path) -> bool:
    """Whether ``xml_path`` has a companion embeddings file on disk."""
    return embeddings_path_for(xml_path).is_file()


def load_ssl_embeddings(xml_path: Path) -> np.ndarray | None:
    """Load the companion embeddings array for a preset, if present.

    Returns:
        An ``(n_glyphs, dim)`` float array in document order, or ``None``
        if no companion file exists for ``xml_path``.
    """
    npz_path = embeddings_path_for(xml_path)
    if not npz_path.is_file():
        return None
    with np.load(npz_path) as data:
        return data["embeddings"]


def attach_ssl_embeddings(
    glyphs: Sequence[Glyph], embeddings: np.ndarray
) -> list[Glyph]:
    """Return a copy of ``glyphs`` with ``ssl_embedding`` set from ``embeddings``.

    ``embeddings[i]`` is attached to ``glyphs[i]`` -- callers must pass
    glyphs in the same document order the embeddings were generated in
    (i.e. straight from :func:`ic_core.io_xml.load_glyphs_bytes`, before
    any reordering).

    Raises:
        ValueError: If the lengths don't match, or if ``embeddings`` isn't
            a 2-D array of finite floats -- this is user-uploadable (a
            companion ``.ssl_embeddings.npz`` file), so a malformed or
            corrupted upload must fail here with a clear message rather
            than surface as an opaque numpy/sklearn error deep inside
            :class:`~ic_core.ssl_classifier.SSLFusionClassifier`.
    """
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or not np.issubdtype(embeddings.dtype, np.floating):
        raise ValueError(
            f"embeddings must be a 2-D array of floats, got shape "
            f"{embeddings.shape} and dtype {embeddings.dtype} -- the "
            ".ssl_embeddings.npz file is malformed."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(
            "embeddings contains NaN or infinite values -- the "
            ".ssl_embeddings.npz file is corrupted."
        )
    if len(glyphs) != len(embeddings):
        raise ValueError(
            f"Glyph count ({len(glyphs)}) does not match embeddings count "
            f"({len(embeddings)}) -- the preset XML and its companion "
            ".ssl_embeddings.npz file are out of sync."
        )
    return [
        dataclasses.replace(g, ssl_embedding=tuple(float(x) for x in vec))
        for g, vec in zip(glyphs, embeddings)
    ]


def match_glyphs_to_source_pages(
    glyphs: Sequence[Glyph],
    page_arrays: Sequence[np.ndarray],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    binarize_threshold: int = DEFAULT_BINARIZE_THRESHOLD,
) -> list[Glyph]:
    """Best-effort recovery of each glyph's real-pixel crop from a pool of
    candidate source page images, by matching bounding box + binary mask.

    A GameraXML training set never records which page each glyph came
    from -- only its bounding box and binary mask. This tries every
    ``page_arrays`` entry at each glyph's own ``(uly, ulx, nrows, ncols)``,
    binarising that crop the same way ``ic_core.ingest`` does by default
    and comparing it pixel-for-pixel against the glyph's own mask
    (``glyph.to_array()``). If some page agrees on at least ``threshold``
    of pixels, that page's *real* (non-binarised, colour) crop is attached
    as the glyph's ``image_gray_b64``, unlocking it for the ``ssl_fusion``
    backend. This is exactly how ``Hufnagel.ssl_embeddings.npz`` was
    generated (see ``core/scripts/generate_hufnagel_ssl_embeddings.py``),
    generalised so an uploaded GameraXML training file can work with
    ssl_fusion as long as its original source page(s) are uploaded
    alongside it.

    ``page_arrays`` should be ``(H, W, 3)`` RGB arrays -- the DINO SimCLR
    checkpoint the ssl_fusion backend uses was trained on real colour
    manuscript photographs, so the crop attached here must preserve
    colour too. Mask verification still happens on a greyscale
    (luminance) view of each page internally; only the attached crop
    itself is colour. ``(H, W)`` greyscale arrays are also accepted for
    backward compatibility, but lose the colour information the model
    was trained on.

    Glyphs with no page clearing the threshold are returned unchanged (no
    crop attached) -- callers already treat a glyph with neither
    ``image_gray_b64`` nor ``ssl_embedding`` as unusable for ssl_fusion, so
    a partial match across a page pool degrades gracefully rather than
    failing outright.
    """
    # Grey views used only for mask verification, computed once per page via
    # PIL's actual convert("L") -- matching this pixel-for-pixel against
    # ic_core.ingest's own greyscale conversion matters, since a manual
    # luma-weight formula rounds slightly differently and can drop
    # otherwise-correct matches right at `threshold`.
    grey_pages = [
        page if page.ndim == 2
        else np.asarray(PILImage.fromarray(page.astype(np.uint8), mode="RGB").convert("L"))
        for page in page_arrays
    ]

    out = []
    for g in glyphs:
        mask = g.to_array()
        best_page, best_match = None, 0.0
        for page, grey_page in zip(page_arrays, grey_pages):
            if g.uly + g.nrows > page.shape[0] or g.ulx + g.ncols > page.shape[1]:
                continue
            crop = grey_page[g.uly : g.uly + g.nrows, g.ulx : g.ulx + g.ncols]
            match = ((crop <= binarize_threshold) == mask).mean()
            if match > best_match:
                best_match, best_page = match, page
        if best_page is not None and best_match >= threshold:
            real_crop = best_page[g.uly : g.uly + g.nrows, g.ulx : g.ulx + g.ncols]
            out.append(
                dataclasses.replace(
                    g, image_gray_b64=grayscale_array_to_png_base64(real_crop)
                )
            )
        else:
            out.append(g)
    return out
