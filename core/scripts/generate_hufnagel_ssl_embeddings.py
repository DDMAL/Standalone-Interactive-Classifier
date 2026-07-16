"""One-off generator for ``core/data/presets/Hufnagel.ssl_embeddings.npz``.

The ``Hufnagel.xml`` preset only stores each glyph's binary RLE mask, not
its source page -- but its glyphs were verified (bbox + mask, 557/557
exact matches) to come from three known page images that are still on
disk in a sibling checkout. This script re-derives each glyph's *real*
greyscale crop from those pages, runs it through the validated SSL
extractor (DINO SimCLR epoch_011, cls_mean pooling), and writes the
resulting ``(557, 768)`` feature array in the exact document order
``ic_core.io_xml.load_glyphs`` produces -- so
``ic_core.ssl_preset_embeddings.attach_ssl_embeddings`` can zip it
straight onto the loaded glyphs.

Not part of the application runtime -- run by hand, once, whenever the
preset or the checkpoint changes:

    IC_SSL_CHECKPOINT=/path/to/epoch_011 uv run --project ic_core \
        python ../scripts/generate_hufnagel_ssl_embeddings.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ic_core" / "src"))

from ic_core.glyph import Glyph  # noqa: E402
from ic_core.image import grayscale_array_to_png_base64  # noqa: E402
from ic_core.ssl_extractor import ViTExtractor  # noqa: E402

PRESET_XML = Path(__file__).resolve().parents[1] / "data" / "presets" / "Hufnagel.xml"
OUT_PATH = PRESET_XML.with_name("Hufnagel.ssl_embeddings.npz")

# Source pages the preset's 557 glyphs were verified to come from -- found
# by brute-force matching each glyph's (uly, ulx, nrows, ncols) + decoded
# RLE mask against every candidate page at >98% pixel agreement.
SOURCE_PAGES = [
    Path(
        "/Users/home/Desktop/DDMAL/standalone-interactive-classifier/core/data/train/hufnagel_example_826dd1b4.png"
    ),
    Path(
        "/Users/home/Desktop/DDMAL/standalone-interactive-classifier/core/data/train/hufnagel_example_a77ec16f.png"
    ),
    Path(
        "/Users/home/Desktop/DDMAL/standalone-interactive-classifier/core/data/train/hufnagel_example_fbed8126.png"
    ),
]

BINARIZE_THRESHOLD = 127  # matches ic_core.ingest.DEFAULT_THRESHOLD


def _decode_rle_mask(rle: list, nrows: int, ncols: int) -> np.ndarray:
    flat = []
    val = 0
    for run in rle:
        flat.extend([val] * run)
        val = 1 - val
    return np.array(flat[: nrows * ncols]).reshape(nrows, ncols)


def _parse_preset_glyphs(xml_path: Path):
    text = xml_path.read_text()
    pattern = re.compile(
        r'<glyph uly="(\d+)" ulx="(\d+)" nrows="(\d+)" ncols="(\d+)">.*?<data>(.*?)</data>',
        re.S,
    )
    for uly, ulx, nrows, ncols, data in pattern.findall(text):
        yield int(uly), int(ulx), int(nrows), int(ncols), [int(x) for x in data.split()]


def main() -> None:
    checkpoint = os.environ.get("IC_SSL_CHECKPOINT")
    if not checkpoint:
        raise SystemExit("Set IC_SSL_CHECKPOINT to the epoch_011 checkpoint directory.")

    pages = [np.array(Image.open(p).convert("L")) for p in SOURCE_PAGES]

    glyphs = []
    unmatched = 0
    for uly, ulx, nrows, ncols, rle in _parse_preset_glyphs(PRESET_XML):
        mask = _decode_rle_mask(rle, nrows, ncols)
        best_page, best_match = None, 0.0
        for page in pages:
            if uly + nrows > page.shape[0] or ulx + ncols > page.shape[1]:
                continue
            crop = page[uly : uly + nrows, ulx : ulx + ncols]
            match = ((crop <= BINARIZE_THRESHOLD).astype(int) == mask).mean()
            if match > best_match:
                best_match, best_page = match, page
        if best_match < 0.98:
            unmatched += 1
            continue
        real_crop = best_page[uly : uly + nrows, ulx : ulx + ncols]
        glyphs.append(
            Glyph.new(
                class_name="",
                image_rle="",
                ncols=ncols,
                nrows=nrows,
                ulx=ulx,
                uly=uly,
                id_state_manual=False,
                confidence=0.0,
                image_gray_b64=grayscale_array_to_png_base64(real_crop),
            )
        )

    if unmatched:
        raise SystemExit(
            f"{unmatched} glyph(s) did not match any source page above the "
            "0.98 threshold -- refusing to write a partial/misaligned "
            "embeddings file. Investigate before re-running."
        )

    print(f"Extracting SSL features for {len(glyphs)} glyphs...")
    extractor = ViTExtractor(checkpoint=checkpoint)
    embeddings = extractor.extract_batch(glyphs, pooling="cls_mean")
    print(f"Done: {embeddings.shape}")

    np.savez_compressed(OUT_PATH, embeddings=embeddings.astype(np.float32))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
