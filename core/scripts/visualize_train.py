"""Render a Hufnagel CSV/PNG training pair with boxes coloured by neume class.

Reuses :func:`convert_hufnagel_csv.parse_via_csv` so the same header
sniffing and ``type`` / ``neume`` attribute handling as the trainer
applies. Reuses the drawing helpers from :mod:`visualize` so the
output is visually consistent with the prediction overlays.

Run::

    cd core/ic_core && uv run python ../scripts/visualize_train.py --id fbed8126
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from convert_hufnagel_csv import (
    PAIR_PNG_TEMPLATE,
    _PAIR_CSV_RE,
    parse_via_csv,
)
from paths import TRAIN_DIR, VIS_DIR
from visualize import (
    _draw_label_pill,
    _draw_translucent_box,
    color_for_class,
    draw_coordinate_scheme,
)


def visualize_pair(csv_path: Path, page_path: Path, output: Path) -> None:
    rows = parse_via_csv(csv_path)
    img = Image.open(page_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    counts: Counter[str] = Counter()
    for x, y, w, h, raw_class in rows:
        color = color_for_class(raw_class)
        _draw_translucent_box(draw, x, y, w, h, color)
        _draw_label_pill(draw, font, x, y, raw_class, color)
        counts[raw_class] += 1

    draw_coordinate_scheme(draw, img.width, img.height)
    output.parent.mkdir(parents=True, exist_ok=True)
    composited = Image.alpha_composite(img, overlay).convert("RGB")
    composited.save(output)

    print(f"Image:       {page_path}")
    print(f"Annotations: {csv_path}")
    print(f"Wrote:       {output}  ({img.width}x{img.height}, {len(rows)} boxes)")
    print("Class counts:")
    for name, n in counts.most_common():
        print(f"  {n:4d}  {name}")


def _resolve_pair(pair_id: str | None, csv_arg: Path | None, page_arg: Path | None
                  ) -> tuple[Path, Path]:
    if csv_arg or page_arg:
        if not (csv_arg and page_arg):
            raise SystemExit("--csv and --page must be given together")
        return csv_arg, page_arg
    if pair_id is None:
        raise SystemExit("either --id or --csv/--page is required")
    csv_path = TRAIN_DIR / f"hufnagel_annotations_{pair_id}.csv"
    page_path = TRAIN_DIR / PAIR_PNG_TEMPLATE.format(id=pair_id)
    if not csv_path.exists():
        raise SystemExit(f"no CSV at {csv_path}")
    if not page_path.exists():
        raise SystemExit(f"no PNG at {page_path}")
    # Sanity-check that the id has the canonical shape the trainer expects.
    if not _PAIR_CSV_RE.match(csv_path.name):
        raise SystemExit(f"{csv_path.name} is not a canonical pair name")
    return csv_path, page_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--id",
        help="Canonical pair id (e.g. fbed8126). Resolves CSV and PNG under TRAIN_DIR.",
    )
    parser.add_argument("--csv", type=Path, help="Override CSV path (pair with --page).")
    parser.add_argument("--page", type=Path, help="Override PNG path (pair with --csv).")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Default: VIS_DIR/<page-stem>_annotated.png",
    )
    args = parser.parse_args()

    csv_path, page_path = _resolve_pair(args.id, args.csv, args.page)
    output = args.out or VIS_DIR / f"{page_path.stem}_annotated.png"
    visualize_pair(csv_path, page_path, output)


if __name__ == "__main__":
    main()
