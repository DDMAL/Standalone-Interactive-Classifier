"""kNN training and classification.

Replaces ``prepare_classifier`` and ``run_correction_stage`` from the original
``interactive_classifier.py``, swapping Gamera's ``kNNInteractive`` for
``sklearn.neighbors.KNeighborsClassifier`` (or ``BallTree`` for speed).

Semantics preserved verbatim from ``KNN_ALGORITHM.md``:

* Full re-train every round — discard and rebuild on each submission.
* ``k=1`` default (winner-takes-all, no voting); ``k`` exposed as a parameter.
* Manual glyphs (``id_state_manual=True``) feed training, never classification.
* Results returned sorted ascending by confidence to match the frontend.
"""
