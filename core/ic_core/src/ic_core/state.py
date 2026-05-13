"""Session state.

Defines ``ClassifierStateEnum`` (``IMPORT_XML`` → ``CLASSIFYING`` →
``GROUP_AND_CLASSIFY`` → ``SAVE`` → ``EXPORT_XML``) and the session dataclass
that holds the authoritative glyph set, class list, and current state.

Unlike the original Rodan wrapper, which accumulated a per-round mutation
log (``@changed_glyphs``, ``@grouped_glyphs``, ``@deleted_glyphs``,
``@renamed_classes``) in a settings dict, the session here is mutated
directly by API endpoints.

State transitions must preserve the ordering from ``wrapper.py``:
``add_grouped_glyphs`` → ``update_changed_glyphs`` → ``remove_deleted_glyphs``
→ ``remove_deleted_classes`` → ``update_renamed_classes`` → ``filter_parts``.
Reordering causes subtle data loss.
"""
