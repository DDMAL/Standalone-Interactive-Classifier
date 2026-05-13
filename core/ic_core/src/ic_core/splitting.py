"""Connected-component splitting.

Replaces ``gamera.plugins.segmentation``. Default splitter is connected
components via ``scipy.ndimage.label`` + ``skimage.measure.regionprops``;
other splitters are added on demand.

Manual split outputs are emitted as ``UNCLASSIFIED`` with ``confidence=0``
and ``id_state_manual=False`` so they are re-classified on the next round.
Each output piece receives a fresh UUID.
"""
