"""One-off generator for ``core/data/presets/Hufnagel training data
06.08.26.ssl_embeddings.npz``.

The preset only stores each glyph's binary RLE mask, not its source page --
but all 305 of its glyphs were verified (bbox + mask, 305/305 exact matches)
to come from ten known page images: the three original Hufnagel example
pages under ``core/data/train/``, plus seven pages supplied by the preset's
author (Yueqiao Zhang) after the preset was replaced on 2026-08-05/06,
orphaning the embeddings the previous preset (``Hufnagel.xml``, 557 glyphs)
had. This script re-derives each glyph's *real* colour crop from those
pages, runs it through the validated SSL extractor (DINO SimCLR epoch_005,
cls_mean pooling), and writes the resulting ``(305, 768)`` feature array in
the exact document order ``ic_core.io_xml.load_glyphs`` produces -- so
``ic_core.ssl_preset_embeddings.attach_ssl_embeddings`` can zip it straight
onto the loaded glyphs.

Not part of the application runtime -- run by hand, once, whenever the
preset or the checkpoint changes:

    IC_SSL_CHECKPOINT=/path/to/epoch_005 uv run --project ic_core \
        python ../scripts/generate_hufnagel_ssl_embeddings.py
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
    / "Hufnagel training data 06.08.26.xml"
)
OUT_PATH = PRESET_XML.with_name(PRESET_XML.stem + ".ssl_embeddings.npz")

_TRAIN_DIR = Path(__file__).resolve().parents[1] / "data" / "train"
_SUPPLIED_DIR = Path("/home/pmohseni/scratch/ddmal/incoming_presets_images")

# Source pages the preset's 305 glyphs were verified to come from -- found
# by brute-force matching each glyph's (uly, ulx, nrows, ncols) + decoded
# RLE mask against every candidate page at >98% pixel agreement (see
# match_glyphs_to_source_pages). Exact per-page glyph counts (sequential
# exclusive-match order; a glyph whose crop happens to also clear 98% on a
# later page in this list is still only counted once, against whichever
# page it's checked against first -- this is purely which page gets
# "credit" in this breakdown, not a gap in coverage):
#
#   hufnagel_example_826dd1b4.png                    129 glyphs
#   hufnagel_example_a77ec16f.png                     62 glyphs
#   hufnagel_example_fbed8126.png                     47 glyphs
#   MS025a-01.jpg                                     18 glyphs
#   Antiphonal_12v_hfngl copy.jpg                     11 glyphs
#   Antiphonale officii Windeshemense I p.009.png      8 glyphs
#   MS234 p.005.png                                    3 glyphs
#   Antiphonal_1v_hfngl copy.jpg                       2 glyphs
#   Antiphonal_44v_hfngl.jpg                          25 glyphs
#   (CantusMA1537 p.22 copy.jpg contributes 0 new glyphs once the three
#   original pages are checked first -- kept in the list anyway since it's
#   a faithful record of everything Yueqiao supplied.)
#                                                     ---
#                                                     305 / 305 matched
#
# The three original pages (already shipping under core/data/train/) cover
# 238/305; the remaining 67 come from images Yueqiao supplied afterwards in
# two follow-up batches, after the preset was replaced on 2026-08-05/06 and
# orphaned the embeddings the previous preset (``Hufnagel.xml``, 557
# glyphs) had. The last 25 -- Antiphonal_44v_hfngl.jpg -- only turned up in
# the second batch, after the first six images still left 25 unmatched.
SOURCE_PAGES = [
    _TRAIN_DIR / "hufnagel_example_826dd1b4.png",  # 129 glyphs
    _TRAIN_DIR / "hufnagel_example_a77ec16f.png",  # 62 glyphs
    _TRAIN_DIR / "hufnagel_example_fbed8126.png",  # 47 glyphs
    _SUPPLIED_DIR / "extracted" / "images" / "huf" / "CantusMA1537 p.22 copy.jpg",  # 0 (see above)
    _SUPPLIED_DIR / "extracted" / "images" / "huf" / "MS025a-01.jpg",  # 18 glyphs
    _SUPPLIED_DIR / "extracted" / "images" / "huf" / "Antiphonal_12v_hfngl copy.jpg",  # 11 glyphs
    _SUPPLIED_DIR
    / "extracted"
    / "images"
    / "huf"
    / "Antiphonale officii Windeshemense I p.009.png",  # 8 glyphs
    _SUPPLIED_DIR / "extracted" / "images" / "huf" / "MS234 p.005.png",  # 3 glyphs
    _SUPPLIED_DIR / "extracted" / "images" / "huf" / "Antiphonal_1v_hfngl copy.jpg",  # 2 glyphs
    _SUPPLIED_DIR / "extracted_new" / "new_images" / "Antiphonal_44v_hfngl.jpg",  # 25 glyphs
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
