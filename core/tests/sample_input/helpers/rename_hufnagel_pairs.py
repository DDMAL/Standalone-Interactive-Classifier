"""Rename messy Hufnagel CSV/PNG pairs to a canonical ``{id}``-keyed scheme.

For each CSV in the target directory whose name doesn't already match
``hufnagel_annotations_*.csv``, we peek at the first data row's
``filename`` column to find the partner page image (swapping the
extension to ``.png``), mint a short random id, and rename both files
in place to::

    hufnagel_annotations_{id}.csv
    Hufnagel_example_{id}.png

Once renamed, :mod:`convert_hufnagel_csv` can auto-discover the pairs
by matching ids regardless of the original filenames.

Quirks handled:

* VIA exports occasionally omit the header row (e.g.
  ``Hufnag_annotations_St.Gall.176v.csv``). We auto-detect that and
  treat the file as headerless with the standard VIA columns.
* Some CSVs put the PNG name in their ``filename`` column with a ``.jpg``
  extension even when the actual file on disk is ``.png``. We swap.

Run::

    cd core/ic_core && uv run python ../tests/sample_input/helpers/rename_hufnagel_pairs.py
"""
from __future__ import annotations

import argparse
import csv
import secrets
from pathlib import Path

HERE = Path(__file__).parent
SAMPLE_DIR = HERE.parent

CSV_PATTERN = "hufnagel_annotations_{id}.csv"
PNG_PATTERN = "hufnagel_example_{id}.png"

VIA_COLUMNS = (
    "filename",
    "file_size",
    "file_attributes",
    "region_count",
    "region_id",
    "region_shape_attributes",
    "region_attributes",
)


def _first_filename(csv_path: Path) -> str:
    """Return the value of the ``filename`` column in the first data row.

    Auto-detects whether the CSV has a header by checking for the
    ``region_shape_attributes`` token in the first line. Headerless
    CSVs are read with :data:`VIA_COLUMNS` as the column names.
    """
    with open(csv_path, newline="") as f:
        first = f.readline()
        f.seek(0)
        if "region_shape_attributes" in first:
            reader = csv.DictReader(f)
        else:
            reader = csv.DictReader(f, fieldnames=VIA_COLUMNS)
        for row in reader:
            name = (row.get("filename") or "").strip()
            if name:
                return name
    raise ValueError(f"{csv_path} has no filename column / no data rows")


def _find_partner_png(directory: Path, csv_filename_field: str) -> Path | None:
    """Find a .png in ``directory`` whose stem matches the CSV's filename field.

    The CSV records its source image with whatever extension the
    annotator used (often ``.jpg``); the file on disk for our pairs is
    ``.png``. We look up by stem and only return PNGs.
    """
    stem = Path(csv_filename_field).stem
    candidate = directory / f"{stem}.png"
    if candidate.exists():
        return candidate
    return None


def _short_id() -> str:
    """Random 8-char hex id, plenty for a handful of training pages."""
    return secrets.token_hex(4)


def _already_renamed(path: Path) -> bool:
    name = path.name
    return name.startswith("hufnagel_annotations_") and name.endswith(".csv")


def discover_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """Return ``(csv, png)`` pairs in ``directory`` that need renaming.

    CSVs already matching the canonical scheme are skipped. CSVs with
    no resolvable partner PNG are reported via a printed warning and
    omitted from the returned list.
    """
    pairs: list[tuple[Path, Path]] = []
    for csv_path in sorted(directory.glob("*.csv")):
        if _already_renamed(csv_path):
            continue
        try:
            field = _first_filename(csv_path)
        except ValueError as e:
            print(f"  ! skip {csv_path.name}: {e}")
            continue
        png = _find_partner_png(directory, field)
        if png is None:
            print(
                f"  ! skip {csv_path.name}: no matching PNG for "
                f"{Path(field).stem!r} in {directory}"
            )
            continue
        pairs.append((csv_path, png))
    return pairs


def rename_pairs(
    directory: Path = SAMPLE_DIR, dry_run: bool = False
) -> list[tuple[Path, Path, str]]:
    """Rename every unrenamed pair in ``directory``.

    Returns ``(new_csv, new_png, id)`` triples for the pairs that were
    (or would be, in ``dry_run``) renamed. Existing canonical pairs
    are not touched.
    """
    pairs = discover_pairs(directory)
    results: list[tuple[Path, Path, str]] = []
    for csv_path, png_path in pairs:
        gid = _short_id()
        new_csv = directory / CSV_PATTERN.format(id=gid)
        new_png = directory / PNG_PATTERN.format(id=gid)
        # token_hex(4) collisions are astronomically unlikely for a few
        # pages, but if the destination exists we'd silently clobber —
        # re-roll to be safe.
        while new_csv.exists() or new_png.exists():
            gid = _short_id()
            new_csv = directory / CSV_PATTERN.format(id=gid)
            new_png = directory / PNG_PATTERN.format(id=gid)
        action = "would rename" if dry_run else "renaming"
        print(f"  {action}: {csv_path.name} -> {new_csv.name}")
        print(f"  {action}: {png_path.name} -> {new_png.name}")
        if not dry_run:
            csv_path.rename(new_csv)
            png_path.rename(new_png)
        results.append((new_csv, new_png, gid))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=SAMPLE_DIR,
        help="Directory to scan for pairs (default: sample_input/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be renamed without touching the filesystem.",
    )
    args = parser.parse_args()

    results = rename_pairs(args.dir, dry_run=args.dry_run)
    if not results:
        print("No pairs to rename.")
        return
    verb = "Would rename" if args.dry_run else "Renamed"
    print(f"{verb} {len(results)} pair(s).")


if __name__ == "__main__":
    main()
