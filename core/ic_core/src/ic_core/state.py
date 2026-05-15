"""Session state and the direct-mutation state machine.

Phase-1 replacement for the legacy Rodan ``ClassifierStateEnum`` plus
the per-round mutation log
(``@changed_glyphs``, ``@grouped_glyphs``, ``@deleted_glyphs``,
``@renamed_classes`` …) that
``../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py``
accumulated in a settings dict.

The new model is **direct-mutation**:

* Each API endpoint operates on a :class:`Session` directly — no
  batched diff, no end-of-round flush.
* This is the approach explicitly recommended by
  ``docs/migration_plan.md`` §"State persistence": *"Don't keep that
  mutation-log pattern. It exists because Rodan re-invokes the task
  with a fresh dict each round."*
* Because mutations apply immediately, the legacy ordering gotcha
  (``add_grouped → update_changed → remove_deleted → ...``) does
  *not* apply at the API surface. Order is the user's call order.

State lifecycle:

.. code-block:: text

    IMPORT  ──ingest──▶  CLASSIFYING  ──complete──▶  EXPORT
                            ▲    │
                            └────┘  (classify / group / edit / delete / …)

``EXPORT`` is terminal: after :meth:`Session.complete` is called, the
session is read-only and the API layer should return the serialised
GameraXML and dispose of the session.

This module deliberately knows nothing about FastAPI, HTTP, JSON, or
disk I/O — those belong to the ``api/`` layer. A :class:`Session` is
just a Python object with methods; the API layer holds a registry of
them and exposes the operations as HTTP endpoints.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from ic_core.classifier import (
    DEFAULT_K,
    UNCLASSIFIED,
    filter_parts,
    run_correction_stage,
    sort_by_confidence_ascending,
)
from ic_core.glyph import Glyph
from ic_core.grouping import manual_group as union_glyphs

# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class ClassifierState(str, Enum):
    """Lifecycle states for a classification session.

    Inherits from ``str`` so the enum members serialise as plain
    JSON strings without the API layer needing a custom encoder.
    Values mirror the legacy ``ClassifierStateEnum`` names but
    collapse a few transient ones (``GROUP_AND_CLASSIFY``, ``SAVE``,
    ``GROUP``) that existed only to dispatch the legacy round-based
    state machine. Under direct mutation they are no longer needed.
    """

    #: Session has been created but no glyphs have been ingested yet.
    IMPORT = "import"

    #: Glyphs are loaded; the user is actively classifying / editing.
    #: Most endpoints operate in (and leave the session in) this state.
    CLASSIFYING = "classifying"

    #: Terminal state. ``Session.complete`` has run, ``glyphs`` is
    #: the final list, and the API layer can serialise to GameraXML.
    EXPORT = "export"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """In-memory authoritative state for one classification session.

    Attributes:
        id: Stable UUID — used by the API layer as the URL token.
        state: Current :class:`ClassifierState`.
        glyphs: The working glyph set. Mixed manual / auto /
            unclassified. Order is "user-visible" — typically sorted
            by ascending confidence after each classify round.
        training_glyphs: Optional external training database loaded
            at ingest time. Treated as read-only in v1: the user
            classifies new glyphs in ``glyphs`` and those become
            training data via ``id_state_manual=True``.
        imported_class_names: Class names declared up front (e.g.
            via the optional class-names text file). These are
            preserved across rounds even if no glyph currently uses
            them, so the UI can offer them as autocomplete
            suggestions.

    The session is intentionally **mutable**. Each operation method
    mutates ``self`` in place and returns ``None``; the API layer is
    expected to serialise the post-mutation session back to the
    caller.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: ClassifierState = ClassifierState.IMPORT
    glyphs: list[Glyph] = field(default_factory=list)
    training_glyphs: list[Glyph] = field(default_factory=list)
    imported_class_names: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> set[str]:
        """Union of every known class name.

        Drawn from (a) the working set, (b) the training database,
        and (c) explicitly imported names. Transient prefixes
        (``_group``, ``_delete``) and the :data:`UNCLASSIFIED`
        sentinel are excluded — those should never be presented as
        selectable labels in the UI.
        """
        names = set(self.imported_class_names)
        for g in self.glyphs:
            names.add(g.class_name)
        for g in self.training_glyphs:
            names.add(g.class_name)
        return {
            n
            for n in names
            if n != UNCLASSIFIED and not _is_transient_name(n)
        }

    def find_glyph(self, glyph_id: str) -> Glyph | None:
        """Return the glyph with the given id from the working set, or ``None``."""
        for g in self.glyphs:
            if g.id == glyph_id:
                return g
        return None

    def _require_state(self, *allowed: ClassifierState) -> None:
        """Raise :class:`StateTransitionError` unless current state is allowed."""
        if self.state not in allowed:
            raise StateTransitionError(
                f"Operation not allowed in state {self.state.value!r}; "
                f"expected one of {[s.value for s in allowed]}"
            )

    # ------------------------------------------------------------------
    # IMPORT → CLASSIFYING
    # ------------------------------------------------------------------

    def ingest(
        self,
        glyphs: Iterable[Glyph],
        *,
        training_glyphs: Iterable[Glyph] | None = None,
        class_names: Iterable[str] | None = None,
    ) -> None:
        """Load the initial glyph set and transition into ``CLASSIFYING``.

        Replaces the legacy ``IMPORT_XML`` stage in ``wrapper.py``
        lines 123–162. Important differences:

        * Inputs are pre-built :class:`Glyph` objects, not GameraXML
          + a CCA stage. The API layer is responsible for converting
          its on-disk format (a directory of cropped PNGs) into
          glyphs before calling this method.
        * The optional training database is split into a separate
          attribute rather than being inlined into ``glyphs``.

        Args:
            glyphs: The working glyph set. May be empty (rare — a
                no-glyph session is allowed for tests).
            training_glyphs: Optional external training database.
            class_names: Optional iterable of class names to seed
                the autocomplete list.

        Raises:
            StateTransitionError: If called from anywhere except
                :attr:`ClassifierState.IMPORT`.
        """
        self._require_state(ClassifierState.IMPORT)
        self.glyphs = list(glyphs)
        self.training_glyphs = list(training_glyphs or [])
        if class_names is not None:
            self.imported_class_names.update(class_names)
        self.state = ClassifierState.CLASSIFYING

    # ------------------------------------------------------------------
    # CLASSIFYING operations
    # ------------------------------------------------------------------

    def classify(self, *, k: int = DEFAULT_K) -> None:
        """Run a full classify round and replace ``glyphs`` with the result.

        Mirrors the legacy ``CLASSIFYING`` stage (wrapper.py lines
        164–188): strip transient prefixes, re-train from manual +
        training-DB glyphs, re-classify every non-manual glyph, sort
        for the UI.

        Args:
            k: Neighbour count. Defaults to 1 (parity with Gamera).

        Raises:
            StateTransitionError: If called outside ``CLASSIFYING``.
            ValueError: If the training pool is empty (propagated
                from :class:`ic_core.classifier.InteractiveClassifier`).
        """
        self._require_state(ClassifierState.CLASSIFYING)
        new_glyphs, _ = run_correction_stage(
            self.glyphs,
            self.training_glyphs,
            k=k,
        )
        # Sort here so the API response is already in display order;
        # the frontend does not have to re-sort. Ascending-confidence
        # ordering is algorithm semantic #3.
        self.glyphs = sort_by_confidence_ascending(new_glyphs)

    def update_glyph(
        self,
        glyph_id: str,
        *,
        class_name: str | None = None,
        id_state_manual: bool | None = None,
    ) -> Glyph:
        """Mutate a single glyph's class label and/or manual flag.

        Used by the API endpoint that handles "user manually
        re-labelled this glyph". Setting ``id_state_manual=True``
        also pins confidence to ``1.0`` to mirror
        :meth:`Glyph.classify_manual` (algorithm semantic #5 — manual
        glyphs feed training, never classification).

        Args:
            glyph_id: The target glyph's UUID.
            class_name: New class label, or ``None`` to leave
                unchanged.
            id_state_manual: New manual flag, or ``None`` to leave
                unchanged.

        Returns:
            The new (replaced) :class:`Glyph`.

        Raises:
            StateTransitionError: If called outside ``CLASSIFYING``.
            KeyError: If no glyph with that id exists in the working
                set.
        """
        self._require_state(ClassifierState.CLASSIFYING)
        idx, old = self._find_index(glyph_id)

        # Decide the new (class_name, manual, confidence) triple.
        # ``classify_manual`` and ``classify_automatic`` on Glyph
        # encapsulate the "manual=True → confidence=1" rule from the
        # algorithm spec; we route through them rather than building
        # a Glyph by hand so the invariant lives in one place.
        if id_state_manual is True:
            new = old.classify_manual(class_name if class_name is not None else old.class_name)
        elif id_state_manual is False:
            new = old.classify_automatic(
                class_name=class_name if class_name is not None else old.class_name,
                confidence=old.confidence,
            )
        else:
            # id_state_manual unchanged — just rename the class.
            if class_name is None:
                return old  # no-op, but allowed
            if old.id_state_manual:
                new = old.classify_manual(class_name)
            else:
                new = old.classify_automatic(class_name, old.confidence)

        self.glyphs[idx] = new
        return new

    def manual_group(self, glyph_ids: Iterable[str], class_name: str) -> Glyph:
        """Union the selected glyphs into one new manual glyph.

        Wraps :func:`ic_core.grouping.manual_group`. The selected
        glyphs are *removed* from the working set and replaced with
        the new grouped glyph (which carries a fresh UUID,
        ``id_state_manual=True``, ``confidence=1.0`` per algorithm
        semantic #7 — it joins training data immediately).

        Args:
            glyph_ids: UUIDs of glyphs to group. Must all exist in
                the working set; otherwise :class:`KeyError` is
                raised before any mutation.
            class_name: Class label for the grouped result.

        Returns:
            The new grouped :class:`Glyph`.

        Raises:
            StateTransitionError: If called outside ``CLASSIFYING``.
            KeyError: If any id is not in the working set.
            ValueError: If ``glyph_ids`` is empty.
        """
        self._require_state(ClassifierState.CLASSIFYING)
        ids = list(glyph_ids)
        if not ids:
            raise ValueError("manual_group requires at least one glyph id")

        # Validate every id up front so a typo in the last id
        # doesn't leave the working set half-mutated.
        targets: list[Glyph] = []
        for gid in ids:
            g = self.find_glyph(gid)
            if g is None:
                raise KeyError(f"No glyph with id {gid!r} in working set")
            targets.append(g)

        grouped = union_glyphs(targets, class_name)

        # Drop the originals and append the grouped result. We keep
        # working-set order otherwise stable so the UI doesn't
        # reshuffle unrelated glyphs.
        target_ids = set(ids)
        self.glyphs = [g for g in self.glyphs if g.id not in target_ids]
        self.glyphs.append(grouped)
        return grouped

    def delete_glyph(self, glyph_id: str) -> None:
        """Remove a glyph from the working set.

        The legacy job used the ``_delete`` class-name prefix to
        defer removal until the next round's ``filter_parts`` pass
        (because mutations were batched). Under direct mutation we
        can just drop the glyph immediately.

        :func:`filter_parts` still recognises the ``_delete`` prefix
        so a frontend that wants to send a batch of "mark these for
        deletion" pseudo-classifications continues to work — it just
        no longer goes through ``delete_glyph``.

        Raises:
            StateTransitionError: If called outside ``CLASSIFYING``.
            KeyError: If no glyph with that id exists.
        """
        self._require_state(ClassifierState.CLASSIFYING)
        idx, _ = self._find_index(glyph_id)
        del self.glyphs[idx]

    def rename_class(self, old_name: str, new_name: str) -> None:
        """Rename a class across glyphs, training set, and imported names.

        Mirrors the legacy ``update_renamed_classes`` (wrapper.py
        lines 447–466) and its dotted-namespace rule: a glyph whose
        class is ``A.subclass`` is renamed to ``new.subclass`` when
        ``A`` is renamed to ``new``. This preserves the chant-neume
        taxonomy convention where ``neume.oblique3`` is a subtype of
        ``neume``.

        Renaming TO :data:`UNCLASSIFIED` is rejected — that sentinel
        is a state, not a label.

        Raises:
            StateTransitionError: If called outside ``CLASSIFYING``.
            ValueError: On a rename to UNCLASSIFIED.
        """
        self._require_state(ClassifierState.CLASSIFYING)
        if new_name == UNCLASSIFIED:
            raise ValueError(f"Cannot rename a class to {UNCLASSIFIED!r}")

        self.glyphs = [_rename_in_glyph(g, old_name, new_name) for g in self.glyphs]
        self.training_glyphs = [
            _rename_in_glyph(g, old_name, new_name) for g in self.training_glyphs
        ]
        # Imported class names — replace ``old`` and ``old.x`` prefixes.
        self.imported_class_names = {
            _rename_in_classname(n, old_name, new_name)
            for n in self.imported_class_names
        }

    def delete_class(self, class_name: str) -> None:
        """Drop a class name from the imported-names list.

        Mirrors ``remove_deleted_classes`` (wrapper.py lines 435–445).
        Does **not** retroactively delete glyphs that still carry
        that class — the user typically deletes individual glyphs
        separately. Removing the class merely takes it off the
        autocomplete list.

        The dotted-namespace rule applies here too: deleting ``A``
        also deletes any ``A.subclass`` entry.
        """
        self._require_state(ClassifierState.CLASSIFYING)
        prefix = class_name + "."
        self.imported_class_names = {
            n
            for n in self.imported_class_names
            if n != class_name and not n.startswith(prefix)
        }

    # ------------------------------------------------------------------
    # CLASSIFYING → EXPORT
    # ------------------------------------------------------------------

    def complete(self) -> None:
        """Finalise the session and transition into ``EXPORT``.

        Performs the last cleanup pass: strip transient prefixes
        (``_group``, ``_delete``) and any lingering UNCLASSIFIED
        glyphs in the training set, so the exported GameraXML
        contains only meaningful data. After this call the session
        is read-only — further mutations raise
        :class:`StateTransitionError`.
        """
        self._require_state(ClassifierState.CLASSIFYING)
        self.glyphs = filter_parts(self.glyphs)
        # Training-set hygiene: drop UNCLASSIFIED and transient
        # entries that snuck in via the original training XML.
        self.training_glyphs = [
            g
            for g in filter_parts(self.training_glyphs)
            if g.class_name != UNCLASSIFIED
        ]
        self.state = ClassifierState.EXPORT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_index(self, glyph_id: str) -> tuple[int, Glyph]:
        for i, g in enumerate(self.glyphs):
            if g.id == glyph_id:
                return i, g
        raise KeyError(f"No glyph with id {glyph_id!r} in working set")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StateTransitionError(RuntimeError):
    """Raised when an operation is invoked in a state that does not allow it.

    The API layer should translate this into HTTP 409 Conflict.
    """


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


_TRANSIENT_NAME_PREFIXES = ("_group", "_delete")


def _is_transient_name(name: str) -> bool:
    return any(name.startswith(p) for p in _TRANSIENT_NAME_PREFIXES)


def _rename_in_glyph(glyph: Glyph, old: str, new: str) -> Glyph:
    """Return ``glyph`` with its ``class_name`` renamed if it matches.

    Honours the dotted-namespace rule: ``A`` → ``new`` also rewrites
    ``A.foo`` → ``new.foo``.
    """
    new_name = _rename_in_classname(glyph.class_name, old, new)
    if new_name == glyph.class_name:
        return glyph
    # Re-route through Glyph's classify_* helpers so the manual /
    # confidence invariants stay consistent.
    if glyph.id_state_manual:
        return glyph.classify_manual(new_name)
    return glyph.classify_automatic(new_name, glyph.confidence)


def _rename_in_classname(name: str, old: str, new: str) -> str:
    """Apply the rename rule to a single string."""
    if name == old:
        return new
    if name.startswith(old + "."):
        return new + name[len(old):]
    return name
