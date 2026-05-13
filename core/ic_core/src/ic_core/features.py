"""Feature extraction.

Replaces the Gamera-internal feature vector computation. Reimplements the
subset of features used by the classifier (aspect ratio, area, moments,
projection histograms, etc.) on top of ``numpy`` and ``scikit-image``. The
feature vector layout is versioned — files produced with one version are not
feature-compatible with another.
"""
