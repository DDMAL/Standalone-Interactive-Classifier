"""GameraXML read/write — the authoritative on-disk format.

Replaces ``intermediary/gamera_xml.py``. Hand-written ``lxml`` parser and
writer that round-trips the GameraXML schema. We stay in XML (no separate
native JSON format) because the output feeds MEI-encoded downstream
pipelines; schema-identical output is a hard requirement. Schema is
undocumented except by example — the fixtures under ``tests/fixtures/``
(copied from the original ``backend/django/code/test/files/``) are the
ground truth.

Before training and before export, ``filter_parts`` strips glyphs whose
class names carry the transient prefixes ``_split``, ``_group``, and
``_delete``.
"""
from pathlib import Path

from lxml import etree


def load_glyphs(path: Path) -> list[dict]:
    tree = etree.parse(str(path))
    glyphs: list[dict] = []
    for g in tree.iterfind(".//glyph"):
        ids = g.find("ids")
        id_el = ids.find("id")
        glyphs.append({
            "class_name": id_el.get("name"),
            "confidence": float(id_el.get("confidence")),
            "id_state_manual": ids.get("state") == "MANUAL",
            "ulx": int(g.get("ulx")),
            "uly": int(g.get("uly")),
            "nrows": int(g.get("nrows")),
            "ncols": int(g.get("ncols")),
        })
    return glyphs
