"""Unit tests for :mod:`ic_core.state`.

Each test maps to a state-machine transition or one of the
direct-mutation operations the API layer will expose. Fixtures are
synthetic so the tests stay fast and don't depend on real ingest.
"""
from __future__ import annotations

import numpy as np
import pytest

from ic_core.classifier import UNCLASSIFIED
from ic_core.glyph import Glyph
from ic_core.image import array_to_rle
from ic_core.state import (
    ClassifierState,
    Session,
    StateTransitionError,
)


def _make_glyph(
    arr: np.ndarray,
    *,
    class_name: str = UNCLASSIFIED,
    id_state_manual: bool = False,
    confidence: float = 0.0,
    ulx: int = 0,
    uly: int = 0,
) -> Glyph:
    arr = np.asarray(arr, dtype=bool)
    nrows, ncols = arr.shape
    return Glyph.new(
        class_name=class_name,
        image_rle=array_to_rle(arr),
        ncols=ncols,
        nrows=nrows,
        ulx=ulx,
        uly=uly,
        id_state_manual=id_state_manual,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Session construction
# ---------------------------------------------------------------------------


def test_new_session_starts_in_import_state():
    s = Session()
    assert s.state is ClassifierState.IMPORT
    assert s.glyphs == []
    assert s.training_glyphs == []
    assert s.imported_class_names == set()


def test_new_session_has_fresh_uuid():
    a, b = Session(), Session()
    assert len(a.id) == 32
    assert a.id != b.id


# ---------------------------------------------------------------------------
# IMPORT → CLASSIFYING
# ---------------------------------------------------------------------------


def test_ingest_transitions_to_classifying():
    s = Session()
    g = _make_glyph(np.ones((4, 4), dtype=bool))
    s.ingest([g])
    assert s.state is ClassifierState.CLASSIFYING
    assert len(s.glyphs) == 1


def test_ingest_seeds_imported_class_names():
    s = Session()
    s.ingest([], class_names=["neume.A", "neume.B"])
    assert "neume.A" in s.imported_class_names
    assert "neume.B" in s.imported_class_names


def test_ingest_rejects_second_call():
    # Once we're in CLASSIFYING we cannot re-ingest — the API
    # contract is "create a new session for a new dataset".
    s = Session()
    s.ingest([_make_glyph(np.ones((2, 2)))])
    with pytest.raises(StateTransitionError):
        s.ingest([])


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


def test_classify_replaces_unmanual_glyphs_keeps_manual():
    manual = _make_glyph(
        np.ones((10, 10), dtype=bool),
        class_name="square",
        id_state_manual=True,
        confidence=1.0,
    )
    query = _make_glyph(np.ones((10, 10), dtype=bool))

    s = Session()
    s.ingest([manual, query])
    s.classify()

    # Manual is preserved (by id, possibly reordered by sort).
    out_manual = next(g for g in s.glyphs if g.id == manual.id)
    assert out_manual == manual

    # Query got a real label and positive confidence.
    out_query = next(g for g in s.glyphs if g.id == query.id)
    assert out_query.class_name == "square"
    assert out_query.confidence > 0.0


def test_classify_outputs_sorted_ascending_by_confidence():
    train = [
        _make_glyph(
            np.ones((10, 10), dtype=bool),
            class_name="A",
            id_state_manual=True,
            confidence=1.0,
        )
    ]
    queries = [_make_glyph(np.ones((s, s), dtype=bool)) for s in (10, 8, 6)]

    sess = Session()
    sess.ingest([*train, *queries])
    sess.classify()

    confs = [g.confidence for g in sess.glyphs]
    assert confs == sorted(confs)


def test_classify_fails_outside_classifying_state():
    s = Session()
    with pytest.raises(StateTransitionError):
        s.classify()


# ---------------------------------------------------------------------------
# update_glyph
# ---------------------------------------------------------------------------


def test_update_glyph_sets_manual_pins_confidence_to_one():
    g = _make_glyph(np.ones((4, 4)), class_name=UNCLASSIFIED)
    s = Session()
    s.ingest([g])

    new = s.update_glyph(g.id, class_name="A", id_state_manual=True)

    assert new.class_name == "A"
    assert new.id_state_manual is True
    assert new.confidence == 1.0
    # UUID preserved.
    assert new.id == g.id


def test_update_glyph_class_only_keeps_manual_flag():
    g = _make_glyph(
        np.ones((4, 4)),
        class_name="A",
        id_state_manual=True,
        confidence=1.0,
    )
    s = Session()
    s.ingest([g])

    new = s.update_glyph(g.id, class_name="B")

    assert new.class_name == "B"
    assert new.id_state_manual is True  # unchanged
    assert new.confidence == 1.0


def test_update_glyph_manual_to_automatic_resets_confidence():
    # A manually-labelled glyph carries the pinned confidence=1.0. Flipping
    # it back to automatic must drop that score so the glyph re-enters the
    # ascending-confidence review queue at the top, rather than sinking to
    # the bottom on a stale 1.0 it never actually earned from the classifier.
    g = _make_glyph(
        np.ones((4, 4)),
        class_name="A",
        id_state_manual=True,
        confidence=1.0,
    )
    s = Session()
    s.ingest([g])

    new = s.update_glyph(g.id, id_state_manual=False)

    assert new.id_state_manual is False
    assert new.confidence == 0.0
    assert new.class_name == "A"


def test_update_glyph_automatic_to_automatic_preserves_confidence():
    # Relabeling an already-automatic glyph should leave its kNN score alone.
    g = _make_glyph(
        np.ones((4, 4)),
        class_name="A",
        id_state_manual=False,
        confidence=0.42,
    )
    s = Session()
    s.ingest([g])

    new = s.update_glyph(g.id, class_name="B", id_state_manual=False)

    assert new.id_state_manual is False
    assert new.confidence == 0.42
    assert new.class_name == "B"


def test_update_glyph_unknown_id_raises_keyerror():
    s = Session()
    s.ingest([_make_glyph(np.ones((2, 2)))])
    with pytest.raises(KeyError):
        s.update_glyph("nope", class_name="A")


# ---------------------------------------------------------------------------
# manual_group
# ---------------------------------------------------------------------------


def test_manual_group_replaces_originals_with_new_glyph():
    a = _make_glyph(np.ones((3, 3)), ulx=0, uly=0)
    b = _make_glyph(np.ones((3, 3)), ulx=10, uly=0)
    s = Session()
    s.ingest([a, b])

    grouped = s.manual_group([a.id, b.id], "compound")

    # Originals are gone; grouped is present.
    ids = {g.id for g in s.glyphs}
    assert a.id not in ids
    assert b.id not in ids
    assert grouped.id in ids

    # Grouped glyph is manual and confident.
    assert grouped.id_state_manual is True
    assert grouped.confidence == 1.0
    assert grouped.class_name == "compound"


def test_manual_group_validates_all_ids_before_mutating():
    a = _make_glyph(np.ones((3, 3)))
    s = Session()
    s.ingest([a])

    # Second id doesn't exist — should raise and leave session unchanged.
    with pytest.raises(KeyError):
        s.manual_group([a.id, "missing"], "X")

    assert [g.id for g in s.glyphs] == [a.id]


def test_manual_group_empty_list_raises():
    s = Session()
    s.ingest([_make_glyph(np.ones((2, 2)))])
    with pytest.raises(ValueError):
        s.manual_group([], "X")


# ---------------------------------------------------------------------------
# delete_glyph
# ---------------------------------------------------------------------------


def test_delete_glyph_removes_immediately():
    a = _make_glyph(np.ones((2, 2)))
    b = _make_glyph(np.ones((3, 3)))
    s = Session()
    s.ingest([a, b])

    s.delete_glyph(a.id)
    assert [g.id for g in s.glyphs] == [b.id]


def test_delete_glyph_unknown_id_raises():
    s = Session()
    s.ingest([_make_glyph(np.ones((2, 2)))])
    with pytest.raises(KeyError):
        s.delete_glyph("nope")


# ---------------------------------------------------------------------------
# rename_class / delete_class
# ---------------------------------------------------------------------------


def test_rename_class_rewrites_glyphs_and_dotted_subclasses():
    a = _make_glyph(np.ones((2, 2)), class_name="neume", id_state_manual=True)
    b = _make_glyph(np.ones((2, 2)), class_name="neume.A", id_state_manual=True)
    c = _make_glyph(np.ones((2, 2)), class_name="other", id_state_manual=True)

    s = Session()
    s.ingest([a, b, c], class_names=["neume", "neume.X", "other"])
    s.rename_class("neume", "punctum")

    name_by_id = {g.id: g.class_name for g in s.glyphs}
    assert name_by_id[a.id] == "punctum"
    assert name_by_id[b.id] == "punctum.A"
    assert name_by_id[c.id] == "other"

    assert "punctum" in s.imported_class_names
    assert "punctum.X" in s.imported_class_names
    assert "other" in s.imported_class_names


def test_rename_class_rejects_unclassified_target():
    s = Session()
    s.ingest([_make_glyph(np.ones((2, 2)), class_name="A", id_state_manual=True)])
    with pytest.raises(ValueError):
        s.rename_class("A", UNCLASSIFIED)


def test_delete_class_drops_name_and_dotted_subclasses_from_imported():
    s = Session()
    s.ingest([], class_names=["neume", "neume.A", "neume.B", "punctum"])
    s.delete_class("neume")
    assert s.imported_class_names == {"punctum"}


# ---------------------------------------------------------------------------
# class_names property
# ---------------------------------------------------------------------------


def test_class_names_unions_working_training_and_imported():
    work = _make_glyph(np.ones((2, 2)), class_name="A", id_state_manual=True)
    train = _make_glyph(np.ones((2, 2)), class_name="B")
    s = Session()
    s.ingest([work], training_glyphs=[train], class_names=["C"])

    assert s.class_names == {"A", "B", "C"}


def test_class_names_excludes_unclassified_and_transient_prefixes():
    work = _make_glyph(np.ones((2, 2)), class_name=UNCLASSIFIED)
    ephemeral = _make_glyph(np.ones((2, 2)), class_name="_group.foo")
    s = Session()
    s.ingest([work, ephemeral])
    assert s.class_names == set()


# ---------------------------------------------------------------------------
# complete (CLASSIFYING → EXPORT)
# ---------------------------------------------------------------------------


def test_complete_transitions_to_export_and_strips_transients():
    keep = _make_glyph(np.ones((2, 2)), class_name="A", id_state_manual=True)
    drop = _make_glyph(np.ones((2, 2)), class_name="_delete")
    s = Session()
    s.ingest([keep, drop])
    s.complete()

    assert s.state is ClassifierState.EXPORT
    assert [g.id for g in s.glyphs] == [keep.id]


def test_complete_freezes_session_against_further_mutation():
    s = Session()
    s.ingest([_make_glyph(np.ones((2, 2)), class_name="A", id_state_manual=True)])
    s.complete()

    with pytest.raises(StateTransitionError):
        s.classify()
    with pytest.raises(StateTransitionError):
        s.delete_glyph("anything")
