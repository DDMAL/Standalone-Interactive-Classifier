"""GameraXML read/write (compatibility).

Replaces ``intermediary/gamera_xml.py``. Hand-written ``lxml`` parser and
writer that round-trips the GameraXML schema used by the existing Rodan
pipeline. Schema is undocumented except by example — the fixtures under
``tests/fixtures/`` (copied from the original ``backend/django/code/test/
files/``) are the ground truth.

Before training and before export, ``filter_parts`` strips glyphs whose
class names carry the transient prefixes ``_split``, ``_group``, and
``_delete``.
"""
