"""Glyph dataclass.

Replaces ``intermediary/gamera_glyph.py``. Defines the in-memory record for a
single classified (or unclassified) glyph: bounding box, binary image data,
class name, confidence, ``id_state_manual`` flag, and stable UUID. UUIDs are
generated on construction and preserved across round-trips; new glyphs from
manual group/split operations receive fresh ones.
"""
