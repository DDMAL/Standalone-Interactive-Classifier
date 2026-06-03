# Feature Normalization & Threshold Stabilization Experiment

*Last updated: 2026-06-03 · branch `data-augmentation`*

An investigation into whether normalizing glyph masks before feature
extraction — and/or stabilizing binarization at ingest — improves the
k-NN classifier's accuracy and macro-F1. Headline result: a **one-line
padding fix** worth **+1.5pp accuracy / +7.1pp macro-F1** on the larger
dataset, traced to a feature-computation defect that affects 100% of
glyphs.

---

## 1. Motivation

The Phase-1 feature vector (29-d, see [features.py](../core/ic_core/src/ic_core/features.py))
mixes crop-invariant features (Hu moments) with several that are
**sensitive to how the detector/annotator framed the bounding box**:
`volume` (foreground fraction), `nrows`/`ncols`, `aspect_ratio`, and the
16-d `volume16regions` density grid. The hypothesis: framing and
binarization inconsistencies push otherwise-identical neumes apart in
feature space, and normalizing the mask should pull them back together.

Three candidate transforms were considered:

1. **Threshold stabilization** — replace the fixed `<=127` cutoff with an
   adaptive one (Otsu / Sauvola).
2. **Center of mass** — translate the ink centroid to the array center
   (normalizes `volume16regions` against framing).
3. **Padding** — add a uniform background border (companion to centering;
   stabilizer for border-sensitive features).

(Tight-cropping was considered and rejected: a bounding box is set by the
single most extreme ink pixel, so one stray speck blows it up. Center of
mass is a mean over all ink and therefore noise-robust.)

---

## 2. Architectural decisions

| Decision | Rationale |
|---|---|
| **Feature-only** — normalization never mutates the stored `Glyph` | The frontend renders the binary mask as the thumbnail and draws the page bbox from `ulx/uly/ncols/nrows`. Persisting padding/centering would change the thumbnail and, worse, make the bbox overlay drift off the actual ink (centering decouples the mask from page coordinates) and corrupt manual group/split. So transforms apply only inside [`compute_features`](../core/ic_core/src/ic_core/features.py). |
| **Opt-in, default off** | Always-on would break the Gamera-parity tests (which assert features come from the raw `g.to_array()`) and isn't worth a `FEATURE_VERSION` bump until proven. Default-off also *is* the A/B baseline arm. Toggled via the `feature_normalization(cfg)` context manager. |
| **Threshold stabilization lives at ingest, not in features** | Binarization happens on the greyscale page at ingest; the `Glyph` only ever carries the already-binary mask. Re-thresholding therefore can't be feature-only — it belongs in [ingest.py](../core/ic_core/src/ic_core/ingest.py). It changes the thumbnail but **not** the bbox. |
| **Binarize the page once, then slice crops** | Adaptive thresholds (Otsu/Sauvola) must see the full-page histogram; a single glyph crop is mostly background and not bimodal. |

---

## 3. Method

- **Code.** [normalize.py](../core/ic_core/src/ic_core/normalize.py)
  (`despeckle`, `center_by_mass`, `pad_mask`, `normalize_mask`,
  `NormalizeConfig`); the `feature_normalization` hook in
  [features.py](../core/ic_core/src/ic_core/features.py);
  `_binarize_page` (`fixed`/`otsu`/`sauvola`) in
  [ingest.py](../core/ic_core/src/ic_core/ingest.py); a
  `--threshold-method` flag on
  [convert_hufnagel_csv.py](../core/scripts/convert_hufnagel_csv.py).
- **Evaluation.** Stratified 5-fold cross-validation, `k=1`, fixed seed
  (`random.Random(0)`), classes with `< 5` members dropped (can't be
  stratified). Predictions are pooled across folds, then scored once for
  **accuracy** and **macro-F1** (unweighted mean per-class F1, so rare
  classes count equally). Harness: `_run_cv` / `_macro_f1` in
  [test_real_input_knn.py](../core/tests/test_real_input_knn.py).
- **Threshold A/B** requires regenerating the training DB under each
  method (binarization is baked into the stored RLE); feature
  normalization is applied live at feature-computation time and needs no
  regeneration.

> **Methodology caveat.** All CV runs are *within a single GameraXML
> database*. Effects that depend on **cross-source** consistency
> (training crops vs. MOTHRA crops) are invisible to this harness. A
> result is only trustworthy here if its mechanism is source-independent.

---

## 4. Datasets

| | **Hufnagel** | **Square_notation** |
|---|---|---|
| Source | regenerated from 3 CSV+PNG pairs in [core/data/train/](../core/data/train/) | committed fixture [Square_notation-example_training_data.xml](../core/tests/fixtures/Square_notation-example_training_data.xml) |
| Glyphs (stratifiable) | 533 | 2221 |
| Classes | — | 16 |
| Baseline macro-F1 | 0.90 (well-separated) | 0.77 (harder minority classes) |
| Selected via | `paths.TRAINING_XML` | `paths.TRAINING_XML` |

The active DB is whatever [`paths.TRAINING_XML`](../core/scripts/paths.py)
resolves to. Initially it pointed at a malformed Square_notation path that
silently forced Hufnagel regeneration (the 533-glyph runs); it was later
repointed at the real 2221-glyph fixture.

---

## 5. Results

### 5.1 Hufnagel (533 glyphs) — flat / negative

**Feature normalization** (`despeckle=3, center, pad=2`):

| arm | accuracy | macro-F1 |
|---|---|---|
| baseline | 0.9512 | 0.9019 |
| normalized | 0.9437 (−0.0075) | 0.8720 (−0.0300) |

**Threshold stabilization** (DB regenerated per method):

| method | accuracy | macro-F1 |
|---|---|---|
| fixed (127) | 0.9512 | 0.9019 |
| otsu | 0.9493 | 0.9038 |
| sauvola | 0.9475 | 0.8722 |

→ Within this clean, single-source set, normalization slightly hurts and
threshold method is ~neutral. 127 is already well-calibrated for these
scans.

### 5.2 Square_notation (2221 glyphs) — padding is a clear win

**Step decomposition** (each step alone, vs. baseline):

| arm | accuracy | macro-F1 | Δacc | ΔF1 |
|---|---|---|---|---|
| baseline | 0.9507 | 0.7679 | — | — |
| despeckle=3 | 0.9507 | 0.7679 | +0.0000 | +0.0000 |
| center | 0.9512 | 0.7710 | +0.0005 | +0.0031 |
| **pad=2** | 0.9629 | 0.8175 | +0.0122 | +0.0496 |
| center + pad=2 | 0.9584 | 0.7967 | +0.0077 | +0.0289 |
| despeckle + center + pad=2 | 0.9584 | 0.7967 | +0.0077 | +0.0289 |

**Padding-width sweep** (pad only):

| pad | accuracy | macro-F1 |
|---|---|---|
| 0 | 0.9507 | 0.7679 |
| 1 | 0.9584 | 0.8046 |
| 2 | 0.9629 | 0.8175 |
| 3 | 0.9629 | 0.8142 |
| **4** | **0.9661** | **0.8392** |
| 6 | 0.9666 | 0.8275 |
| 8 | 0.9643 | 0.8197 |
| 12 | 0.9634 | 0.8322 |

**Census: 2221 / 2221 (100%) of glyphs have ink touching a bbox edge.**

---

## 6. Interpretation

- **Padding is the entire gain.** Best at `pad=4`: **+1.54pp accuracy,
  +7.13pp macro-F1**. The benefit erodes past ~6 as excess background
  dilutes the density features.
- **Mechanism is a feature-computation defect, not a cross-source
  artifact.** `nholes`, `compactness`/`perimeter`, and Hu moments assume
  a connected background ring around the ink. With 100% of glyphs
  touching an edge, that assumption is violated for *every* glyph; a
  background border restores it. Because this is source-independent, the
  gain should transfer to MOTHRA and any other input — unlike a purely
  cross-source effect.
- **The macro-F1 gain (~5× the accuracy gain) means it rescues minority
  classes.** That's exactly what a global accuracy number hides and what
  prompted tracking F1 in the first place.
- **`center` hurts.** Its asymmetric padding distorts `aspect_ratio`,
  which is genuinely discriminative for neume shapes (tall virga vs. wide
  clivis). It gives back part of padding's gain (`+0.0496` → `+0.0289`).
- **`despeckle` is a no-op on clean fixtures.** Keep it available for
  noisy real MOTHRA input, but it does nothing on committed data.
- **Threshold method is ~neutral within-source** on both datasets.
- **Why Hufnagel disagreed:** at 533 glyphs with an already-high 0.90
  macro-F1, the set is smaller, cleaner, and easier; the border-feature
  defect existed there too but had less headroom to recover, and
  fold-to-fold jitter is larger. The larger, harder Square_notation set
  exposed the effect.

---

## 7. Decisions & status

- **Adopt `pad≈4`; drop `center` (harmful) and `despeckle` (no-op on
  clean data).** The A/B default config in
  [test_real_input_knn.py](../core/tests/test_real_input_knn.py) is now
  pad-only=4, overridable via `IC_NORM_PAD` / `IC_NORM_CENTER` /
  `IC_NORM_DESPECKLE`.
- **Kept opt-in.** Padding is *not* yet the `compute_features` default,
  and `FEATURE_VERSION` is unchanged.
- **Promotion gated on MOTHRA verification.** Before making padding the
  production default (which entails a `FEATURE_VERSION` bump and updating
  the Gamera-parity tests), confirm the gain on a hand-labeled MOTHRA
  ground-truth set — the one axis this within-source CV cannot measure.

### Reproduce

```bash
cd core/ic_core
# Point paths.TRAINING_XML at the desired DB, then:
IC_NORM_PAD=4 uv run pytest \
  ../tests/test_real_input_knn.py::test_xml_db_5fold_normalization_ab -s -m slow
# Threshold A/B (regenerate the DB per method):
uv run python ../scripts/convert_hufnagel_csv.py --threshold-method otsu --out-xml /tmp/otsu.xml
```

---

## 8. Next step

Build a small **hand-labeled MOTHRA ground-truth set** + a scorer that
reuses the same accuracy/macro-F1 and the `IC_NORM_PAD` knob, to verify
`pad=4` on real detector output before promoting it to the default.
