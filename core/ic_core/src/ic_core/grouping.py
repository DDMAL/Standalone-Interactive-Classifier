"""Spatial grouping.

Replaces ``group_and_correct`` and Gamera's grouping functions
(``ShapedGroupingFunction``, ``BoundingBoxGroupingFunction``).

* Shaped grouping: pairwise pixel-distance via
  ``scipy.ndimage.distance_transform_edt``, then a graph build.
* Bounding-box grouping: pure-numpy bounding-box distance check.
* Manual group: ``np.logical_or`` over aligned binary masks; recompute the
  bounding box. Resulting glyph is marked ``id_state_manual=True`` with
  ``confidence=1`` and a fresh UUID — it joins training data immediately.

A ``max_graph_size`` cap mirrors Gamera's ``cknn.group_list_automatic`` to
prevent the grouping graph from blowing up on large pages.
"""
