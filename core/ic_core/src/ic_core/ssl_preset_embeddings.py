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

from ic_core.glyph import Glyph

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
        ValueError: If the lengths don't match.
    """
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
