"""GameraXML read/write — the authoritative on-disk format.

Replaces ``intermediary/gamera_xml.py`` from the legacy Rodan job.
Hand-written ``lxml`` parser and writer that round-trips the
GameraXML schema. We stay in XML (no separate native JSON format)
because the output feeds MEI-encoded downstream pipelines;
schema-identical output is a hard requirement.

The schema is undocumented except by example — the fixtures under
``tests/fixtures/`` (copied from the original
``backend/django/code/test/files/``) are the ground truth.

Reader notes
------------

* ``load_glyphs`` is **export-only** in spirit: the main ingest path
  uses page-image + annotation-byte inputs via ``ic_core.ingest``,
  not GameraXML. The reader still exists for the round-trip
  regression test (Phase 1 Verification §4 in the migration plan) —
  write a session, read it back, confirm the semantics survive.

Writer notes
------------

* :func:`write_glyphs` emits glyphs in the order given (no
  re-sorting). Callers are expected to apply any display ordering
  before calling.
* The legacy ``<features>`` block is intentionally **omitted**.
  Per the migration plan §"Risks and gotchas" (1), embedded feature
  vectors are versioned and treated as a clean break — downstream
  MEI consumers must not depend on them. Skipping the block keeps
  the output compact and avoids implying spurious compatibility.
* ``filter_parts`` should be applied **before** calling
  :func:`write_glyphs` so transient ``_group`` / ``_delete`` glyphs
  do not leak into the export.
"""
from pathlib import Path
from typing import Iterable

from lxml import etree

from ic_core.classifier import UNCLASSIFIED
from ic_core.glyph import Glyph

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Top-level Gamera DTD version. The legacy fixtures all use "2.0".
GAMERA_DB_VERSION: str = "2.0"


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def load_glyphs(path: Path) -> list[Glyph]:
    """Parse a GameraXML file into a list of :class:`Glyph` objects.

    Args:
        path: Filesystem path to a ``gamera-database`` XML document.

    Returns:
        A list of glyphs in document order. Each glyph receives a
        fresh UUID — the legacy XML format does not encode them.
    """
    tree = etree.parse(str(path))
    glyphs: list[Glyph] = []
    for g in tree.iterfind(".//glyph"):
        ids = g.find("ids")
        id_el = ids.find("id")
        rle = (g.findtext("data") or "").strip()
        glyphs.append(
            Glyph.new(
                class_name=id_el.get("name"),
                image_rle=rle,
                ncols=int(g.get("ncols")),
                nrows=int(g.get("nrows")),
                ulx=int(g.get("ulx")),
                uly=int(g.get("uly")),
                id_state_manual=ids.get("state") == "MANUAL",
                confidence=float(id_el.get("confidence")),
                is_training=False,
            )
        )
    return glyphs


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_glyphs(glyphs: Iterable[Glyph], path: Path) -> None:
    """Serialise glyphs to a GameraXML file on disk.

    The output is byte-identical (modulo whitespace) to what the
    legacy Gamera ``WriteXMLFile(with_features=False)`` would
    produce: a ``gamera-database`` root with version="2.0", a
    single ``<glyphs>`` child, and one ``<glyph>`` per input.

    Args:
        glyphs: Iterable of :class:`Glyph` to write. Order is
            preserved. Caller is responsible for applying
            ``filter_parts`` to strip transient ``_group`` /
            ``_delete`` entries beforehand.
        path: Destination file. Will be overwritten if it exists.
    """
    xml_bytes = dumps_glyphs(glyphs)
    Path(path).write_bytes(xml_bytes)


def dumps_glyphs(glyphs: Iterable[Glyph]) -> bytes:
    """Serialise glyphs to GameraXML bytes (for in-memory use, e.g. HTTP responses)."""
    root = etree.Element("gamera-database", version=GAMERA_DB_VERSION)
    glyphs_el = etree.SubElement(root, "glyphs")
    for g in glyphs:
        _append_glyph(glyphs_el, g)

    return etree.tostring(
        root,
        xml_declaration=True,
        encoding="utf-8",
        pretty_print=True,
    )


def _append_glyph(parent: etree._Element, glyph: Glyph) -> None:
    """Render a single :class:`Glyph` as a ``<glyph>`` element."""
    g_el = etree.SubElement(
        parent,
        "glyph",
        # The legacy schema orders attributes uly, ulx, nrows, ncols.
        # lxml will serialise in insertion order, so we set them in
        # the same order for byte-level fixture compatibility.
        uly=str(glyph.uly),
        ulx=str(glyph.ulx),
        nrows=str(glyph.nrows),
        ncols=str(glyph.ncols),
    )

    ids_el = etree.SubElement(g_el, "ids", state=_id_state(glyph))
    etree.SubElement(
        ids_el,
        "id",
        name=glyph.class_name,
        # Six decimal places matches the legacy fixture formatting
        # (e.g. ``confidence="1.000000"``).
        confidence=f"{glyph.confidence:.6f}",
    )

    data_el = etree.SubElement(g_el, "data")
    data_el.text = glyph.image_rle


def _id_state(glyph: Glyph) -> str:
    """Map a glyph's internal state to the GameraXML ``state`` attribute.

    Three legacy values exist: ``MANUAL`` (user-confirmed),
    ``AUTOMATIC`` (classifier-assigned), and ``UNCLASSIFIED`` (no
    label yet). We pick based on ``id_state_manual`` and the
    presence of a real class name — never emit ``AUTOMATIC`` for an
    UNCLASSIFIED glyph.
    """
    if glyph.id_state_manual:
        return "MANUAL"
    if glyph.class_name == UNCLASSIFIED:
        return "UNCLASSIFIED"
    return "AUTOMATIC"
