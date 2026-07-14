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
    s.classify(k=1)

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
    sess.classify(k=1)

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
# manual_split
# ---------------------------------------------------------------------------


def test_manual_split_replaces_parent_with_children():
    # One parent split into two children — parent goes, children come
    # in at the parent's index so UI ordering doesn't reshuffle.
    a = _make_glyph(np.ones((4, 4), dtype=bool), ulx=0, uly=0)
    parent = _make_glyph(np.ones((4, 4), dtype=bool), ulx=10, uly=0)
    c = _make_glyph(np.ones((4, 4), dtype=bool), ulx=20, uly=0)
    s = Session()
    s.ingest([a, parent, c])

    children = s.manual_split(parent.id, [(10, 0, 2, 4), (12, 0, 2, 4)])

    assert len(children) == 2
    ids = [g.id for g in s.glyphs]
    # Parent gone, children inserted at parent's old index (between a and c).
    assert parent.id not in ids
    assert ids[0] == a.id
    assert ids[1] == children[0].id
    assert ids[2] == children[1].id
    assert ids[3] == c.id


def test_manual_split_children_are_unclassified():
    # Algorithm semantic #8: children re-enter classification — they
    # must NOT inherit the parent's class or training flag.
    parent = _make_glyph(
        np.ones((4, 4), dtype=bool),
        class_name="neume.compound",
        id_state_manual=True,
        confidence=1.0,
    )
    s = Session()
    s.ingest([parent])
    children = s.manual_split(parent.id, [(0, 0, 4, 4)])

    for child in children:
        assert child.class_name == UNCLASSIFIED
        assert child.confidence == 0.0
        assert child.id_state_manual is False
        assert child.id != parent.id  # fresh UUID


def test_manual_split_unknown_glyph_raises_keyerror():
    s = Session()
    s.ingest([_make_glyph(np.ones((2, 2), dtype=bool))])
    with pytest.raises(KeyError):
        s.manual_split("nope", [(0, 0, 1, 1)])


def test_manual_split_empty_regions_raises():
    parent = _make_glyph(np.ones((4, 4), dtype=bool))
    s = Session()
    s.ingest([parent])
    with pytest.raises(ValueError, match="at least one region"):
        s.manual_split(parent.id, [])


def test_manual_split_all_regions_miss_parent_raises():
    # Business rule: silently deleting the parent because every region
    # misses is almost certainly a UI bug — reject with ValueError.
    # The parent must remain in the working set.
    parent = _make_glyph(np.ones((4, 4), dtype=bool), ulx=0, uly=0)
    s = Session()
    s.ingest([parent])
    with pytest.raises(ValueError, match="every region misses"):
        s.manual_split(parent.id, [(100, 100, 5, 5)])
    assert [g.id for g in s.glyphs] == [parent.id]


def test_manual_split_outside_classifying_raises():
    s = Session()
    # No ingest → still in IMPORT.
    with pytest.raises(StateTransitionError):
        s.manual_split("anything", [(0, 0, 1, 1)])


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
# delete_training_glyph
# ---------------------------------------------------------------------------


def test_delete_training_glyph_removes_from_pool_and_provenance():
    keep = _make_glyph(np.ones((2, 2)), class_name="A", id_state_manual=True)
    drop = _make_glyph(np.ones((3, 3)), class_name="B", id_state_manual=True)
    s = Session()
    s.ingest([_make_glyph(np.ones((2, 2)))], training_glyphs=[keep, drop])
    # Provenance sets are populated by the API layer; simulate that here.
    s.preset_training_ids = {keep.id}
    s.uploaded_training_ids = {drop.id}

    s.delete_training_glyph(drop.id)

    assert [g.id for g in s.training_glyphs] == [keep.id]
    # The working set is untouched — only the training pool shrinks.
    assert len(s.glyphs) == 1
    # The dropped id is gone from provenance; the surviving one is kept.
    assert s.uploaded_training_ids == set()
    assert s.preset_training_ids == {keep.id}


def test_delete_training_glyph_unknown_id_raises():
    s = Session()
    s.ingest(
        [_make_glyph(np.ones((2, 2)))],
        training_glyphs=[_make_glyph(np.ones((2, 2)), class_name="A")],
    )
    with pytest.raises(KeyError):
        s.delete_training_glyph("nope")


def test_delete_training_glyph_requires_classifying_state():
    s = Session()  # still in IMPORT — no ingest yet
    with pytest.raises(StateTransitionError):
        s.delete_training_glyph("anything")


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


# ---------------------------------------------------------------------------
# rebinarize — re-derive masks under a new method, carry labels by id
# ---------------------------------------------------------------------------


def _make_glyph_with_id(
    gid: str,
    arr: np.ndarray,
    *,
    class_name: str = UNCLASSIFIED,
    id_state_manual: bool = False,
    confidence: float = 0.0,
) -> Glyph:
    arr = np.asarray(arr, dtype=bool)
    nrows, ncols = arr.shape
    return Glyph.new(
        id=gid,
        class_name=class_name,
        image_rle=array_to_rle(arr),
        ncols=ncols,
        nrows=nrows,
        ulx=0,
        uly=0,
        id_state_manual=id_state_manual,
        confidence=confidence,
    )


def test_rebinarize_swaps_masks_and_carries_labels_by_id():
    # Session as it stands after some labelling: "a" manual, "b" auto.
    old_a = _make_glyph_with_id(
        "a", np.zeros((2, 2)), class_name="X", id_state_manual=True, confidence=1.0
    )
    old_b = _make_glyph_with_id(
        "b", np.zeros((2, 2)), class_name="Y", confidence=0.5
    )
    s = Session()
    s.ingest([old_a, old_b])

    # The fresh ingest under a new method: same ids, *different* masks,
    # all UNCLASSIFIED (as ingest produces).
    new_a = _make_glyph_with_id("a", np.ones((2, 2)))
    new_b = _make_glyph_with_id("b", np.ones((2, 2)))
    new_mask = np.ones((4, 4), dtype=bool)
    s.rebinarize([new_a, new_b], page_mask=new_mask, method="sauvola")

    by_id = {g.id: g for g in s.glyphs}
    # Masks are the new ones.
    assert by_id["a"].image_rle == new_a.image_rle
    assert by_id["b"].image_rle == new_b.image_rle
    # Labels carried forward.
    assert by_id["a"].class_name == "X" and by_id["a"].id_state_manual is True
    assert by_id["b"].class_name == "Y" and by_id["b"].id_state_manual is False
    # Method + page mask updated.
    assert s.binarization_method == "sauvola"
    assert s.page_mask is new_mask


def test_rebinarize_drops_glyphs_absent_from_base_set():
    # A grouped/split glyph carries a fresh id not in the detector's base
    # set, so it falls away when we re-derive from the base annotations.
    base = _make_glyph_with_id("a", np.zeros((2, 2)), class_name="X")
    grouped = _make_glyph_with_id("group-1", np.ones((3, 3)), class_name="G")
    s = Session()
    s.ingest([base, grouped])

    s.rebinarize([_make_glyph_with_id("a", np.ones((2, 2)))], page_mask=None, method="otsu")

    assert [g.id for g in s.glyphs] == ["a"]


def test_rebinarize_requires_classifying_state():
    s = Session()  # still IMPORT
    with pytest.raises(StateTransitionError):
        s.rebinarize([], page_mask=None, method="global")
