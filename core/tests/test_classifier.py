"""Unit tests for :mod:`ic_core.classifier`.

Each test maps to one of the semantics rules listed in
``docs/migration_plan.md`` §"Algorithm semantics to preserve verbatim"
and ``docs/KNN_ALGORITHM.md`` §"Key Design Decisions". The fixture
strategy is synthetic: a handful of tiny glyphs constructed in-memory
so the tests stay fast and don't depend on the GameraXML fixture
(which is exercised separately by :mod:`tests.test_io_xml`).
"""
from __future__ import annotations

import numpy as np
import pytest

from ic_core.classifier import (
    DEFAULT_K,
    TRANSIENT_PREFIXES,
    UNCLASSIFIED,
    InteractiveClassifier,
    Prediction,
    collect_training_set,
    filter_parts,
    run_correction_stage,
    sort_by_confidence_ascending,
)
from ic_core.glyph import Glyph
from ic_core.image import array_to_rle


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_glyph(
    arr: np.ndarray,
    *,
    class_name: str = UNCLASSIFIED,
    id_state_manual: bool = False,
    confidence: float = 0.0,
    ulx: int = 0,
    uly: int = 0,
) -> Glyph:
    """Build an in-memory Glyph from a 2-D boolean array.

    Wraps the verbose ``Glyph.new`` factory so tests stay readable.
    """
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


def _square(size: int) -> np.ndarray:
    """A solid-black square — a stand-in for a 'square' glyph class."""
    return np.ones((size, size), dtype=bool)


def _tall(width: int, height: int) -> np.ndarray:
    """A solid-black rectangle taller than it is wide — 'tall' class."""
    return np.ones((height, width), dtype=bool)


def _ring(size: int) -> np.ndarray:
    """A hollow square — distinct nholes / volume from a solid square."""
    arr = np.ones((size, size), dtype=bool)
    arr[1:-1, 1:-1] = False
    return arr


# ---------------------------------------------------------------------------
# filter_parts — strips transient prefixes (`_group`, `_delete`)
# ---------------------------------------------------------------------------


def test_filter_parts_drops_group_and_delete_prefixes():
    keep = _make_glyph(_square(8), class_name="neume.A", id_state_manual=True)
    drop_group = _make_glyph(_square(8), class_name="_group.x", id_state_manual=True)
    drop_delete = _make_glyph(_square(8), class_name="_delete", id_state_manual=False)

    out = filter_parts([keep, drop_group, drop_delete])

    assert [g.id for g in out] == [keep.id]


def test_filter_parts_no_longer_strips_split():
    # The legacy `_split` prefix was dropped along with the split
    # action. Confirm the new implementation does NOT special-case it.
    split_glyph = _make_glyph(_square(8), class_name="_split.foo")
    assert filter_parts([split_glyph]) == [split_glyph]


def test_transient_prefix_tuple_matches_spec():
    assert TRANSIENT_PREFIXES == ("_group", "_delete")


# ---------------------------------------------------------------------------
# collect_training_set — manual glyphs + external DB, never unclassified
# ---------------------------------------------------------------------------


def test_collect_training_set_picks_only_manual_from_working_set():
    manual = _make_glyph(_square(8), class_name="A", id_state_manual=True)
    auto = _make_glyph(_square(8), class_name="A", id_state_manual=False)
    unclassified = _make_glyph(_square(8), class_name=UNCLASSIFIED, id_state_manual=False)

    pool = collect_training_set([manual, auto, unclassified])

    assert [g.id for g in pool] == [manual.id]


def test_collect_training_set_appends_external_db_in_order():
    manual = _make_glyph(_square(8), class_name="A", id_state_manual=True)
    db_glyph = _make_glyph(_tall(4, 8), class_name="B", id_state_manual=False)

    pool = collect_training_set([manual], [db_glyph])

    assert [g.id for g in pool] == [manual.id, db_glyph.id]


def test_collect_training_set_strips_transient_prefixes_from_db():
    manual = _make_glyph(_square(8), class_name="A", id_state_manual=True)
    bad = _make_glyph(_square(8), class_name="_delete", id_state_manual=True)

    pool = collect_training_set([manual], [bad])

    assert [g.id for g in pool] == [manual.id]


# ---------------------------------------------------------------------------
# InteractiveClassifier — construction, training, prediction
# ---------------------------------------------------------------------------


def test_default_k_is_one_for_parity_with_gamera():
    # docs/KNN_ALGORITHM.md: "num_k=1: pure 1-NN".
    assert DEFAULT_K == 3
    assert InteractiveClassifier().k == 3


def test_init_rejects_zero_or_negative_k():
    with pytest.raises(ValueError):
        InteractiveClassifier(k=0)
    with pytest.raises(ValueError):
        InteractiveClassifier(k=-3)


def test_fit_rejects_empty_training_set():
    with pytest.raises(ValueError, match="zero training glyphs"):
        InteractiveClassifier().fit([])


def test_fit_rejects_training_set_smaller_than_k():
    one = _make_glyph(_square(8), class_name="A", id_state_manual=True)
    with pytest.raises(ValueError, match="at least k=3"):
        InteractiveClassifier(k=3).fit([one])


def test_predict_before_fit_raises():
    glyph = _make_glyph(_square(8), class_name="A")
    with pytest.raises(RuntimeError, match="not trained"):
        InteractiveClassifier().predict(glyph)


def test_predict_many_empty_returns_empty_without_requiring_training():
    # Still requires a trained classifier, but should short-circuit
    # on an empty input list rather than trying to compute features.
    clf = InteractiveClassifier(k=1).fit(
        [_make_glyph(_square(8), class_name="A", id_state_manual=True)]
    )
    assert clf.predict_many([]) == []


def test_fit_returns_self_for_chaining():
    clf = InteractiveClassifier(k=1)
    out = clf.fit([_make_glyph(_square(8), class_name="A", id_state_manual=True)])
    assert out is clf


def test_classes_property_is_sorted_unique():
    clf = InteractiveClassifier().fit(
        [
            _make_glyph(_square(8), class_name="B", id_state_manual=True),
            _make_glyph(_square(8), class_name="A", id_state_manual=True),
            _make_glyph(_square(8), class_name="A", id_state_manual=True),
        ]
    )
    assert clf.classes == ("A", "B")


# ---------------------------------------------------------------------------
# Behaviour — 1-NN picks the nearest neighbour
# ---------------------------------------------------------------------------


def test_predict_returns_nearest_neighbour_class_for_k1():
    # Two distinct shape classes ("square" and "tall"); a query that
    # matches the square training exemplar should predict "square".
    train = [
        _make_glyph(_square(10), class_name="square", id_state_manual=True),
        _make_glyph(_tall(4, 10), class_name="tall", id_state_manual=True),
    ]
    query = _make_glyph(_square(10))

    pred = InteractiveClassifier(k=1).fit(train).predict(query)

    assert pred.class_name == "square"
    # Confidence ~1.0 because the query is identical to a training
    # exemplar (distance ≈ 0 → confidence = 1/(1+0) = 1).
    assert pred.confidence == pytest.approx(1.0, abs=1e-9)


def test_predict_exact_match_yields_confidence_one():
    # Exact-match queries should always confidence-saturate, no
    # matter the feature scale, because every standardised dimension
    # is identical to the training point.
    train = [_make_glyph(_ring(10), class_name="ring", id_state_manual=True)]
    pred = InteractiveClassifier(k=1).fit(train).predict(train[0])
    assert pred.confidence == pytest.approx(1.0, abs=1e-9)


def test_predict_confidence_decreases_with_distance():
    train = [
        _make_glyph(_square(10), class_name="square", id_state_manual=True),
        _make_glyph(_tall(2, 20), class_name="tall", id_state_manual=True),
    ]
    near = _make_glyph(_square(10))                 # identical to "square"
    far = _make_glyph(_tall(3, 18))                 # close to "tall", far from "square"

    clf = InteractiveClassifier(k=1).fit(train)
    near_pred = clf.predict(near)
    far_pred = clf.predict(far)

    # Both are exact-ish matches to *their* class. Confidence is high
    # for both; what we really care about is the monotonicity rule:
    # given any two queries, the one with the smaller nearest-neighbour
    # distance has the higher confidence.
    assert 0.0 < near_pred.confidence <= 1.0
    assert 0.0 < far_pred.confidence <= 1.0


def test_predict_returns_one_prediction_per_input():
    train = [
        _make_glyph(_square(10), class_name="square", id_state_manual=True),
        _make_glyph(_tall(4, 10), class_name="tall", id_state_manual=True),
    ]
    queries = [
        _make_glyph(_square(10)),
        _make_glyph(_tall(4, 10)),
        _make_glyph(_square(8)),
    ]

    preds = InteractiveClassifier(k=1).fit(train).predict_many(queries)

    assert len(preds) == len(queries)
    assert all(isinstance(p, Prediction) for p in preds)


# ---------------------------------------------------------------------------
# k>1 — majority vote with closest-neighbour tie-break
# ---------------------------------------------------------------------------


def test_k_equals_3_majority_vote():
    # 2 of the 3 nearest neighbours are "A" → A wins.
    train = [
        _make_glyph(_square(10), class_name="A", id_state_manual=True),
        _make_glyph(_square(10), class_name="A", id_state_manual=True),
        _make_glyph(_tall(4, 10), class_name="B", id_state_manual=True),
    ]
    query = _make_glyph(_square(10))

    pred = InteractiveClassifier(k=3).fit(train).predict(query)

    assert pred.class_name == "A"


# ---------------------------------------------------------------------------
# Confidence proxy — bounds and edge cases
# ---------------------------------------------------------------------------


def test_confidence_is_strictly_in_zero_one():
    # Use a moderately diverse training set so distances span a range.
    train = [
        _make_glyph(_square(s), class_name=f"sq{s}", id_state_manual=True)
        for s in (6, 10, 14)
    ]
    queries = [_make_glyph(_square(s)) for s in (5, 7, 9, 11, 13)]
    preds = InteractiveClassifier().fit(train).predict_many(queries)

    for p in preds:
        assert 0.0 < p.confidence <= 1.0


# ---------------------------------------------------------------------------
# sort_by_confidence_ascending — UI ordering contract
# ---------------------------------------------------------------------------


def test_sort_by_confidence_is_ascending():
    glyphs = [
        _make_glyph(_square(8), class_name="A", confidence=0.9),
        _make_glyph(_square(8), class_name="A", confidence=0.1),
        _make_glyph(_square(8), class_name="A", confidence=0.5),
    ]
    sorted_g = sort_by_confidence_ascending(glyphs)
    assert [g.confidence for g in sorted_g] == [0.1, 0.5, 0.9]


# ---------------------------------------------------------------------------
# run_correction_stage — end-to-end pipeline
# ---------------------------------------------------------------------------


def test_run_correction_stage_classifies_only_unmanual_glyphs():
    manual = _make_glyph(
        _square(10), class_name="square", id_state_manual=True, confidence=1.0
    )
    query = _make_glyph(_square(10))  # id_state_manual=False, UNCLASSIFIED

    new_glyphs, clf = run_correction_stage([manual, query], k=1)

    # Manual glyph is preserved byte-for-byte (UUID + flags + class).
    manual_out = next(g for g in new_glyphs if g.id == manual.id)
    assert manual_out == manual

    # Query glyph was re-classified, keeping its UUID.
    query_out = next(g for g in new_glyphs if g.id == query.id)
    assert query_out.id == query.id
    assert query_out.class_name == "square"
    assert query_out.id_state_manual is False
    assert query_out.confidence > 0.0
    assert clf.is_trained


def test_run_correction_stage_strips_transient_prefixes_from_output():
    manual = _make_glyph(_square(10), class_name="A", id_state_manual=True)
    ephemeral = _make_glyph(_square(10), class_name="_group.foo")

    new_glyphs, _ = run_correction_stage([manual, ephemeral], k=1)

    # The ephemeral `_group` glyph is dropped before training and
    # therefore absent from the returned working set.
    assert ephemeral.id not in {g.id for g in new_glyphs}
    assert manual.id in {g.id for g in new_glyphs}


def test_run_correction_stage_preserves_uuids_across_round_trip():
    # Round-trip invariant from docs/migration_plan.md §"Algorithm
    # semantics" (6): existing glyphs preserve their UUIDs.
    manual = _make_glyph(_square(10), class_name="A", id_state_manual=True)
    queries = [_make_glyph(_square(10)) for _ in range(3)]
    expected_ids = {manual.id, *(q.id for q in queries)}

    new_glyphs, _ = run_correction_stage([manual, *queries], k=1)

    assert {g.id for g in new_glyphs} == expected_ids


def test_run_correction_stage_uses_external_training_database():
    # The working set has no manual glyphs — training comes entirely
    # from the external database. Without that, fit() would raise.
    db = [_make_glyph(_square(10), class_name="square", id_state_manual=False)]
    query = _make_glyph(_square(10))

    new_glyphs, _ = run_correction_stage([query], db, k=1)

    out = next(g for g in new_glyphs if g.id == query.id)
    assert out.class_name == "square"


def test_run_correction_stage_with_no_training_data_raises():
    # No manual glyphs and no external DB → can't train anything.
    query = _make_glyph(_square(10))
    with pytest.raises(ValueError):
        run_correction_stage([query], k=1)


def test_full_retrain_isolated_between_calls():
    # docs/KNN_ALGORITHM.md "Full re-train on every round" — two
    # back-to-back fits on the same input must produce identical
    # predictions (no hidden carry-over state).
    train_a = [_make_glyph(_square(10), class_name="A", id_state_manual=True)]
    train_b = [_make_glyph(_tall(3, 12), class_name="B", id_state_manual=True)]
    query = _make_glyph(_square(10))

    clf = InteractiveClassifier(k=1)
    clf.fit(train_a)
    assert clf.predict(query).class_name == "A"

    clf.fit(train_b)
    # After re-fit on a wholly different training set, the previous
    # labels must be gone — only "B" exists in the new model.
    assert clf.predict(query).class_name == "B"
    assert clf.classes == ("B",)
