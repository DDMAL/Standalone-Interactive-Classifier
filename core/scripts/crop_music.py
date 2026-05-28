"""Batch music-region crop pipeline for SSL pre-training data preparation.

Takes a directory of page images and their corresponding Mothra JSON
annotation files, filters for music regions (classId=2), crops them
from the page, and saves them as individual image files ready for
ViT SSL training.

Output structure:
    <output_dir>/
        <page_stem>_<annotation_id>.jpg
        ...
        manifest.json   ← list of all crops with source metadata

Usage::

    python crop_music.py \\
        --images-dir /path/to/pages \\
        --annotations-dir /path/to/mothra/json \\
        --output-dir /path/to/crops

The script expects annotation JSON files to share the same stem as
their corresponding page image (e.g. ``page_001.jpg`` → ``page_001.json``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

# Mothra classId for music regions.
MUSIC_CLASS_ID = 2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def collect_page_annotation_pairs(
    images_dir: Path,
    annotations_dir: Path,
) -> list[tuple[Path, Path]]:
    """Return (image_path, json_path) pairs where both files exist."""
    pairs = []
    for ext in IMAGE_EXTENSIONS:
        for image_path in sorted(images_dir.rglob(f"*{ext}")):
            json_path = annotations_dir / f"{image_path.stem}.json"
            if json_path.exists():
                pairs.append((image_path, json_path))
    return pairs


def crop_music_regions(
    image_path: Path,
    json_path: Path,
    output_dir: Path,
    padding: int = 0,
) -> list[dict]:
    """Crop all music regions from one page and save as individual images.

    Args:
        image_path: Full page image.
        json_path: Mothra JSON annotation file for this page.
        output_dir: Directory to write crop images into.
        padding: Extra pixels to expand each bbox on all sides.

    Returns:
        List of metadata dicts for each saved crop.
    """
    doc = json.loads(json_path.read_bytes())
    annotations = [a for a in doc.get("annotations", []) if a.get("classId") == MUSIC_CLASS_ID]

    if not annotations:
        return []

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    saved = []

    for a in annotations:
        ulx, uly, w, h = a["bbox"]
        x0 = max(0, ulx - padding)
        y0 = max(0, uly - padding)
        x1 = min(img_w, ulx + w + padding)
        y1 = min(img_h, uly + h + padding)

        if x1 <= x0 or y1 <= y0:
            continue

        crop = image.crop((x0, y0, x1, y1))
        ann_id = a["id"].replace("-", "")
        out_path = output_dir / f"{image_path.stem}_{ann_id}.jpg"
        crop.save(out_path, format="JPEG", quality=95)

        saved.append({
            "file": out_path.name,
            "source_image": image_path.name,
            "annotation_id": a["id"],
            "bbox": [x0, y0, x1 - x0, y1 - y0],
            "confidence": a.get("confidence"),
        })

    return saved


def run(
    images_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    padding: int = 0,
) -> None:
    pairs = collect_page_annotation_pairs(images_dir, annotations_dir)

    if not pairs:
        print(f"No matching image/annotation pairs found.")
        return

    print(f"Found {len(pairs)} page(s). Cropping music regions...")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    total_crops = 0

    for image_path, json_path in pairs:
        crops = crop_music_regions(image_path, json_path, output_dir, padding=padding)
        manifest.extend(crops)
        total_crops += len(crops)
        print(f"  {image_path.name}: {len(crops)} music crop{'s' if len(crops) != 1 else ''}")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {total_crops} crops saved to {output_dir}/")
    print(f"Manifest: {output_dir / 'manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-crop music regions from manuscript pages for SSL pre-training."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing full page images.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="Directory containing Mothra JSON annotation files (same stem as images).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write cropped music images and manifest into.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Extra pixels to add around each bbox on all sides (default: 0).",
    )
    args = parser.parse_args()

    run(
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        padding=args.padding,
    )


if __name__ == "__main__":
    main()
