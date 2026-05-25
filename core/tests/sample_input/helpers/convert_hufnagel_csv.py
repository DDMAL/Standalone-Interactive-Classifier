"""Convert Hufnagel VIA-format CSV + page-image pairs into GameraXML training data.

A Hufnagel annotation CSV (e.g. ``fixtures/Hufnagel-example.csv`` or
``sample_input/hufnagel_annotations_*.csv``) is a VIA export: one
bbox per row plus a free-form class label. We crop each bbox out of
the companion page image, binarise, and emit a GameraXML database
matching the schema of
``fixtures/Interactive_Classifier_GameraXML_TrainingData.xml`` — so
the resulting glyphs can be loaded by :func:`ic_core.io_xml.load_glyphs`
and used as KNN training data.

Two invocation modes:

* **Batch (default).** Auto-scan ``sample_input/`` for canonical pairs
  named ``hufnagel_annotations_{id}.csv`` / ``Hufnagel_example_{id}.png``
  (see :mod:`rename_hufnagel_pairs`) and merge every glyph from every
  pair into a single training XML at
  ``fixtures/Hufnagel_training_data.xml``.

* **Single pair.** Pass ``--csv`` and ``--page`` to convert one pair
  exactly like the original script did, writing a per-pair XML plus a
  MOTHRA-shaped JSON sidecar for :mod:`visualize`.

Quirks handled:

* The original Hufnagel-example CSV uses ``"type"`` as the class
  attribute key; the new CSVs in ``sample_input/`` use ``"neume"``.
  We accept either.
* Some new CSVs (notably ``Hufnag_annotations_St.Gall.176v.csv``)
  ship without a header row. We auto-detect and use the standard VIA
  column order.

Class labels in the original Hufnagel CSV (``f-clef``, ``puncta``,
``pes``, …) don't all line up with the GameraXML vocabulary, so
:data:`HUFNAGEL_TO_GAMERA` translates the recognised ones. The new
CSVs already store mapped names (``clef.f``, ``neume.punctum``, …),
which fall through the mapping unchanged.

Run::

    cd core/ic_core && uv run python ../tests/sample_input/helpers/convert_hufnagel_csv.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import uuid
from collections import Counter
from dataclasses import replace
from pathlib import Path

from ic_core.glyph import Glyph
from ic_core.ingest import ingest_page
from ic_core.io_xml import write_glyphs

HERE = Path(__file__).parent
SAMPLE_DIR = HERE.parent
FIXTURES = SAMPLE_DIR.parent / "fixtures"

DEFAULT_CSV = FIXTURES / "Hufnagel-example.csv"
DEFAULT_PAGE = SAMPLE_DIR / "Hufnagel-example.png"
DEFAULT_OUTPUT_XML = FIXTURES / "Hufnagel-example_training_data.xml"
#: MOTHRA-shaped JSON sidecar for visualize.py / ingest_page_json
#: consumers. Lives next to the page image so it's picked up by
#: relative paths like SAMPLE_DIR / "Hufnagel-example_annotations.json".
DEFAULT_OUTPUT_JSON = SAMPLE_DIR / "Hufnagel-example_annotations.json"

#: Merged-output destination for batch mode. Separate from the legacy
#: single-page Hufnagel-example XML so the original file isn't
#: clobbered by accident.
DEFAULT_MERGED_XML = FIXTURES / "Hufnagel_training_data.xml"

#: Filename glob used by batch mode to find canonical CSVs renamed by
#: :mod:`rename_hufnagel_pairs`.
PAIR_CSV_GLOB = "hufnagel_annotations_*.csv"
#: Companion PNG template; ``{id}`` is the same hex id used in the CSV.
PAIR_PNG_TEMPLATE = "Hufnagel_example_{id}.png"
_PAIR_CSV_RE = re.compile(r"^hufnagel_annotations_(.+)\.csv$")

#: Standard VIA column order, used when a CSV is shipped without its
#: header row.
VIA_COLUMNS = (
    "filename",
    "file_size",
    "file_attributes",
    "region_count",
    "region_id",
    "region_shape_attributes",
    "region_attributes",
)

#: Map raw Hufnagel CSV class strings to GameraXML vocabulary. Entries
#: where the value equals the key are pass-throughs — they emit the
#: raw label and need a real mapping decision. The newer CSVs in
#: sample_input/ already store mapped names (``clef.c``,
#: ``neume.podatus2``, …); those simply fall through unchanged because
#: they are not keys here.
HUFNAGEL_TO_GAMERA: dict[str, str] = {
    "f-clef": "clef.f",
    "custos": "custos",
    "puncta": "neume.punctum",
    "torculus": "neume.torculus2",
    "virga": "neume.virga",
    "pes": "neume.pescephalicus2",
    "clivis": "neume.clivis2",
    "scandicus": "neume.scandicus22a",
    "divisio": "divisio.maxima",
}


def _open_via_reader(f) -> csv.DictReader:
    """Return a DictReader over ``f``, auto-detecting header presence.

    Some Hufnagel CSVs (e.g. the St.Gall.176v export) lack a header
    row. Detect that by sniffing the first line and fall back to
    :data:`VIA_COLUMNS`.
    """
    first = f.readline()
    f.seek(0)
    if "region_shape_attributes" in first:
        return csv.DictReader(f)
    return csv.DictReader(f, fieldnames=VIA_COLUMNS)


def parse_via_csv(csv_path: Path) -> list[tuple[int, int, int, int, str]]:
    """Return ``(x, y, w, h, raw_class)`` tuples in CSV row order.

    Non-rect shapes are skipped. Accepts both ``"type"`` (original
    Hufnagel-example CSV) and ``"neume"`` (newer sample_input CSVs)
    as the class-label key inside ``region_attributes``.
    """
    rows: list[tuple[int, int, int, int, str]] = []
    with open(csv_path, newline="") as f:
        reader = _open_via_reader(f)
        for row in reader:
            shape = json.loads(row["region_shape_attributes"])
            attrs = json.loads(row["region_attributes"])
            if shape.get("name") != "rect":
                continue
            raw_class = (attrs.get("type") or attrs.get("neume") or "").strip()
            if not raw_class:
                continue
            rows.append(
                (
                    int(shape["x"]),
                    int(shape["y"]),
                    int(shape["width"]),
                    int(shape["height"]),
                    raw_class,
                )
            )
    return rows


def _stable_ids(csv_path: Path, count: int) -> list[str]:
    """Derive stable per-row UUIDs from the CSV filename + row index.

    Using uuid5 (deterministic) means re-running the converter doesn't
    churn the JSON's ids — the sidecar diff stays clean unless the
    underlying CSV actually changes. The CSV's filename is the
    namespace seed, so distinct pairs in batch mode get distinct id
    streams.
    """
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, csv_path.name)
    return [str(uuid.uuid5(namespace, str(i))) for i in range(count)]


def _glyphs_for_pair(csv_path: Path, page_path: Path) -> list[Glyph]:
    """Crop one CSV/page pair and return training-ready Glyphs.

    Each glyph is marked ``id_state_manual=True`` / ``confidence=1.0``
    so it serialises as ``state="MANUAL"`` — matching the legacy
    training database.
    """
    annotations = parse_via_csv(csv_path)
    ids = _stable_ids(csv_path, len(annotations))

    # Hand the bboxes to ingest_page so cropping + binarisation + RLE
    # encoding use the exact same code path as live ingest. We
    # synthesise the MOTHRA JSON shape it expects.
    doc = {
        "annotations": [
            {"id": gid, "bbox": [x, y, w, h]}
            for gid, (x, y, w, h, _) in zip(ids, annotations)
        ],
    }
    glyphs = ingest_page(
        page_path.read_bytes(),
        json.dumps(doc).encode("utf-8"),
        format="json",
    )

    training: list[Glyph] = []
    for g, (_, _, _, _, raw_class) in zip(glyphs, annotations):
        mapped = HUFNAGEL_TO_GAMERA.get(raw_class, raw_class)
        training.append(
            replace(
                g,
                class_name=mapped,
                confidence=1.0,
                id_state_manual=True,
                is_training=True,
            )
        )
    return training


def convert(
    csv_path: Path = DEFAULT_CSV,
    page_path: Path = DEFAULT_PAGE,
    output_xml: Path = DEFAULT_OUTPUT_XML,
    output_json: Path | None = DEFAULT_OUTPUT_JSON,
) -> list[Glyph]:
    """Convert a single CSV/page pair and write XML (+ optional JSON sidecar).

    Returns the glyphs in CSV order. The XML and the JSON share the
    per-row uuid5 ids generated by :func:`_stable_ids`, so glyphs
    ingested from the JSON line up one-to-one with the training XML
    by id.
    """
    training = _glyphs_for_pair(csv_path, page_path)
    annotations = parse_via_csv(csv_path)
    ids = _stable_ids(csv_path, len(annotations))

    write_glyphs(training, output_xml)

    if output_json is not None:
        sidecar = {
            "imageName": page_path.name,
            "annotations": [
                {
                    "id": gid,
                    "classId": 2,
                    "bbox": [x, y, w, h],
                    "type": raw_class,
                }
                for gid, (x, y, w, h, raw_class) in zip(ids, annotations)
            ],
        }
        output_json.write_text(json.dumps(sidecar, indent=2))

    return training


def discover_pairs(directory: Path = SAMPLE_DIR) -> list[tuple[Path, Path]]:
    """Return ``(csv, png)`` pairs in canonical naming form.

    Looks for files matching :data:`PAIR_CSV_GLOB` and, for each,
    expects a sibling PNG named per :data:`PAIR_PNG_TEMPLATE`. A CSV
    without its partner image is skipped with a warning.
    """
    pairs: list[tuple[Path, Path]] = []
    for csv_path in sorted(directory.glob(PAIR_CSV_GLOB)):
        m = _PAIR_CSV_RE.match(csv_path.name)
        if not m:
            continue
        gid = m.group(1)
        png_path = directory / PAIR_PNG_TEMPLATE.format(id=gid)
        if not png_path.exists():
            print(f"  ! skip {csv_path.name}: no companion {png_path.name}")
            continue
        pairs.append((csv_path, png_path))
    return pairs


def convert_batch(
    directory: Path = SAMPLE_DIR,
    output_xml: Path = DEFAULT_MERGED_XML,
) -> list[Glyph]:
    """Convert every canonical pair in ``directory`` into one merged XML.

    The returned glyph list is the concatenation of each pair's
    glyphs, in pair-discovery order (which is alphabetical by id).
    """
    pairs = discover_pairs(directory)
    if not pairs:
        raise SystemExit(
            f"No canonical pairs found in {directory}. Run "
            f"rename_hufnagel_pairs.py first, or use --csv / --page "
            f"to convert a single pair."
        )
    merged: list[Glyph] = []
    for csv_path, png_path in pairs:
        glyphs = _glyphs_for_pair(csv_path, png_path)
        print(f"  + {csv_path.name} ({len(glyphs)} glyphs) <- {png_path.name}")
        merged.extend(glyphs)
    write_glyphs(merged, output_xml)
    return merged


def _looks_canonical(name: str) -> bool:
    """True if ``name`` already looks like a GameraXML class name.

    Canonical names either appear as values in :data:`HUFNAGEL_TO_GAMERA`
    or follow the dotted ``family.subtype`` shape (``clef.c``,
    ``neume.podatus2``, …). Used by the histogram to suppress the
    "needs mapping" flag for labels that are already canonical.
    """
    return name in set(HUFNAGEL_TO_GAMERA.values()) or "." in name


def _print_histogram(glyphs: list[Glyph], raw_classes: list[str]) -> None:
    """Print per-class counts and flag labels that still need a real mapping.

    A label is "needs mapping" iff (a) it wasn't translated by
    :data:`HUFNAGEL_TO_GAMERA` and (b) it doesn't already look like a
    canonical GameraXML name. The newer sample_input CSVs already
    store canonical names, so those rows shouldn't be flagged.
    """
    unmapped = {
        HUFNAGEL_TO_GAMERA.get(r, r)
        for r in raw_classes
        if HUFNAGEL_TO_GAMERA.get(r, r) == r and not _looks_canonical(r)
    }
    counts = Counter(g.class_name for g in glyphs)
    print("Class histogram:")
    for name, n in counts.most_common():
        flag = "  (needs mapping)" if name in unmapped else ""
        print(f"  {n:4d}  {name}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Single-pair mode: VIA CSV to convert. Pair with --page.",
    )
    parser.add_argument(
        "--page",
        type=Path,
        default=None,
        help="Single-pair mode: companion page image.",
    )
    parser.add_argument(
        "--out-xml",
        type=Path,
        default=None,
        help=(
            "Output XML path. Default is "
            f"{DEFAULT_MERGED_XML.name} in batch mode and "
            f"{DEFAULT_OUTPUT_XML.name} in single-pair mode."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Single-pair mode only: MOTHRA-shaped JSON sidecar path.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=SAMPLE_DIR,
        help="Batch mode: directory to scan for canonical pairs.",
    )
    args = parser.parse_args()

    single = args.csv is not None or args.page is not None
    if single and not (args.csv and args.page):
        parser.error("--csv and --page must be given together")

    if single:
        out_xml = args.out_xml or DEFAULT_OUTPUT_XML
        glyphs = convert(args.csv, args.page, out_xml, args.out_json)
        print(f"Wrote {len(glyphs)} glyphs to {out_xml}")
        if args.out_json is not None:
            print(f"Wrote {len(glyphs)} annotations to {args.out_json}")
        raw_classes = [raw for *_, raw in parse_via_csv(args.csv)]
        _print_histogram(glyphs, raw_classes)
        return

    out_xml = args.out_xml or DEFAULT_MERGED_XML
    print(f"Batch mode: scanning {args.dir}")
    glyphs = convert_batch(args.dir, out_xml)
    print(f"Wrote {len(glyphs)} glyphs to {out_xml}")
    raw_classes: list[str] = []
    for csv_path, _ in discover_pairs(args.dir):
        raw_classes.extend(raw for *_, raw in parse_via_csv(csv_path))
    _print_histogram(glyphs, raw_classes)


if __name__ == "__main__":
    main()
