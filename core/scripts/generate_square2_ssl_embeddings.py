"""One-off generator for ``core/data/presets/Square notation training data
05.08.26.ssl_embeddings.npz``.

The preset only stores each glyph's binary RLE mask, not its source page --
but all 249 of its glyphs were verified (bbox + mask, 249/249 exact matches)
to come from nine known page images, supplied by the preset's author
(Yueqiao Zhang) after the preset was replaced on 2026-07-15, orphaning the
embeddings the previous preset (``Square2.xml``, 144 glyphs, a single known
source page) had. This script re-derives each glyph's *real* colour crop
from those pages, runs it through the validated SSL extractor (DINO SimCLR
epoch_005, cls_mean pooling), and writes the resulting ``(249, 768)``
feature array in the exact document order ``ic_core.io_xml.load_glyphs``
produces -- so ``ic_core.ssl_preset_embeddings.attach_ssl_embeddings`` can
zip it straight onto the loaded glyphs.

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

PRESET_XML = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "presets"
    / "Square notation training data 05.08.26.xml"
)
OUT_PATH = PRESET_XML.with_name(PRESET_XML.stem + ".ssl_embeddings.npz")

_SUPPLIED_DIR = (
    Path("/home/pmohseni/scratch/ddmal/incoming_presets_images")
    / "extracted"
    / "images"
    / "square"
)

# Source pages the preset's 249 glyphs were verified to come from -- found
# by brute-force matching each glyph's (uly, ulx, nrows, ncols) + decoded
# RLE mask against every candidate page at >98% pixel agreement (see
# match_glyphs_to_source_pages), at the module default binarize_threshold
# (127; the original Square2.xml preset needed 110, but this replacement
# preset matches cleanly at the default). Exact per-page glyph counts:
#
#   Einsiedeln__Stiftsbibliothek__Codex_611_014r copy.jpg   52 glyphs
#   Einsiedeln__Stiftsbibliothek__Codex_611_030r.jpg        42 glyphs
#   Einsiedeln__Stiftsbibliothek__Codex_611_057r.jpg        41 glyphs
#   Einsiedeln__Stiftsbibliothek__Codex_611_045v.jpg        30 glyphs
#   Halifax_..._M2149.L4_I_008r_resized_small.png           26 glyphs
#   liber_0124.jpg                                          23 glyphs
#   liber_0123.jpg                                          22 glyphs
#   liber_0125_scaled.jpg                                    8 glyphs
#   Aarau_MsMurF2_14v copy.jpg                                5 glyphs
#                                                           ---
#                                                           249 / 249 matched
#
# Only 9 of the 12 candidate pages Yueqiao supplied actually contributed a
# match -- the other three (A-Gu-30_144v, CH-Fco Ms. 2_006r,
# F-Pn-Latin-15181_107r) are kept in the list anyway since including a
# non-matching candidate is harmless and this stays a faithful record of
# everything that was tried.
SOURCE_PAGES = [
    _SUPPLIED_DIR / "Einsiedeln__Stiftsbibliothek__Codex_611_014r copy.jpg",  # 52 glyphs
    _SUPPLIED_DIR / "Einsiedeln__Stiftsbibliothek__Codex_611_030r.jpg",  # 42 glyphs
    _SUPPLIED_DIR / "Einsiedeln__Stiftsbibliothek__Codex_611_057r.jpg",  # 41 glyphs
    _SUPPLIED_DIR / "Einsiedeln__Stiftsbibliothek__Codex_611_045v.jpg",  # 30 glyphs
    _SUPPLIED_DIR
    / "Halifax_-_Saint_Mary_s_University_-_Patrick_Power_Library_-_ms._M2149.L4_I_008r_resized_small.png",  # 26 glyphs
    _SUPPLIED_DIR / "liber_0124.jpg",  # 23 glyphs
    _SUPPLIED_DIR / "liber_0123.jpg",  # 22 glyphs
    _SUPPLIED_DIR / "liber_0125_scaled.jpg",  # 8 glyphs
    _SUPPLIED_DIR / "Aarau_MsMurF2_14v copy.jpg",  # 5 glyphs
    _SUPPLIED_DIR / "A-Gu-30_144v copy.jpg",  # 0 -- kept for the record
    _SUPPLIED_DIR / "CH-Fco Ms. 2_006r copy.jpg",  # 0 -- kept for the record
    _SUPPLIED_DIR / "F-Pn-Latin-15181_107r copy.jpeg",  # 0 -- kept for the record
]


def main() -> None:
    checkpoint = os.environ.get("IC_SSL_CHECKPOINT")
    if not checkpoint:
        raise SystemExit("Set IC_SSL_CHECKPOINT to the epoch_005 checkpoint directory.")

    glyphs = load_glyphs(PRESET_XML)
    pages = [np.array(Image.open(p).convert("RGB")) for p in SOURCE_PAGES]

    matched = match_glyphs_to_source_pages(glyphs, pages)
    unmatched = [g for g in matched if g.image_gray_b64 is None]
    if unmatched:
        raise SystemExit(
            f"{len(unmatched)} of {len(matched)} glyph(s) did not match any "
            "source page above the 0.98 threshold -- refusing to write a "
            "partial/misaligned embeddings file. Investigate before re-running."
        )

    # ic_core.ssl_extractor deliberately maps the Narval-local model path
    # baked into the checkpoint (pretrained_models/dino-vits16) to its
    # public HF Hub id (facebook/dino-vits16) -- correct for the deployed
    # API (which has internet and no pretrained_models/ dir), wrong for
    # running this one-off script offline, here, on Narval, where the
    # reverse is true. Override just for this run.
    import ic_core.ssl_extractor as _ssl_extractor  # noqa: E402
    local_dino = "/lustre07/scratch/pmohseni/ddmal/pretrained_models/dino-vits16"
    _ssl_extractor._NARVAL_TO_HF["pretrained_models/dino-vits16"] = local_dino

    print(f"Extracting SSL features for {len(matched)} glyphs...")
    extractor = ViTExtractor(checkpoint=checkpoint)
    embeddings = extractor.extract_batch(matched, pooling="cls_mean")
    print(f"Done: {embeddings.shape}")

    np.savez_compressed(OUT_PATH, embeddings=embeddings.astype(np.float32))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
