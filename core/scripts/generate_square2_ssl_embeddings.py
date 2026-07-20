"""One-off generator for ``core/data/presets/Square2.ssl_embeddings.npz``.

The ``Square2.xml`` preset only stores each glyph's binary RLE mask, not
its source page -- but all 144 of its glyphs were verified (bbox + mask,
144/144 exact matches at binarize_threshold=110) to come from
``core/data/train/Einsiedeln__Stiftsbibliothek__Codex_611_014r.jpg``. This
script re-derives each glyph's *real* greyscale crop from that page,
runs it through the validated SSL extractor (DINO SimCLR epoch_005,
cls_mean pooling), and writes the resulting ``(144, 768)`` feature array
in the exact document order ``ic_core.io_xml.load_glyphs`` produces -- so
``ic_core.ssl_preset_embeddings.attach_ssl_embeddings`` can zip it
straight onto the loaded glyphs.

Not part of the application runtime -- run by hand, once, whenever the
preset or the checkpoint changes:

    IC_SSL_CHECKPOINT=/path/to/epoch_005 uv run --project ic_core \
        python ../scripts/generate_square2_ssl_embeddings.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ic_core" / "src"))

from ic_core.io_xml import load_glyphs  # noqa: E402
from ic_core.ssl_extractor import ViTExtractor  # noqa: E402
from ic_core.ssl_preset_embeddings import match_glyphs_to_source_pages  # noqa: E402

PRESET_XML = Path(__file__).resolve().parents[1] / "data" / "presets" / "Square2.xml"
OUT_PATH = PRESET_XML.with_name("Square2.ssl_embeddings.npz")

# Source page all 144 of Square2.xml's glyphs were verified to come from --
# found by brute-force matching each glyph's (uly, ulx, nrows, ncols) +
# decoded RLE mask against this page at binarize_threshold=110, giving
# 144/144 exact (>0.98) matches.
SOURCE_PAGE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "train"
    / "Einsiedeln__Stiftsbibliothek__Codex_611_014r.jpg"
)
BINARIZE_THRESHOLD = 110


def main() -> None:
    checkpoint = os.environ.get("IC_SSL_CHECKPOINT")
    if not checkpoint:
        raise SystemExit("Set IC_SSL_CHECKPOINT to the epoch_005 checkpoint directory.")

    glyphs = load_glyphs(PRESET_XML)
    page = np.array(Image.open(SOURCE_PAGE).convert("L"))

    matched = match_glyphs_to_source_pages(
        glyphs, [page], binarize_threshold=BINARIZE_THRESHOLD
    )
    unmatched = [g for g in matched if g.image_gray_b64 is None]
    if unmatched:
        raise SystemExit(
            f"{len(unmatched)} of {len(matched)} glyph(s) did not match the "
            "source page above the 0.98 threshold -- refusing to write a "
            "partial/misaligned embeddings file. Investigate before re-running."
        )

    print(f"Extracting SSL features for {len(matched)} glyphs...")
    extractor = ViTExtractor(checkpoint=checkpoint)
    embeddings = extractor.extract_batch(matched, pooling="cls_mean")
    print(f"Done: {embeddings.shape}")

    np.savez_compressed(OUT_PATH, embeddings=embeddings.astype(np.float32))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
