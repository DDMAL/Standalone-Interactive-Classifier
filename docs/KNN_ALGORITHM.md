# KNN Algorithm Implementation

This document describes how the k-Nearest Neighbors classifier is implemented in the Interactive Classifier job.

## Library

Classification is done entirely by **Gamera's `kNNInteractive`** class (`gamera.knn`). The Interactive Classifier does not implement its own distance metric or feature extraction — it delegates all of that to Gamera and only manages the data lifecycle around it.

---

## Training Data Construction (`prepare_classifier`)

Every time the backend re-enters an active state (`CLASSIFYING`, `GROUP_AND_CLASSIFY`, `EXPORT_XML`), a fresh classifier is trained from scratch. There is no incremental update.

```
prepare_classifier(training_database, glyphs, features_file_path)
```

**Step 1 — collect manual glyphs from the working set.**  
All glyphs in `settings['glyphs']` with `id_state_manual=True` are converted from RLE → Gamera `ONEBIT/DENSE` image (via `RunLengthImage`) and marked with `classify_manual(class_name)`. These are the corrections the user has made in the current session.

**Step 2 — merge the pre-existing training database.**  
Glyphs from `settings['training_glyphs']` (loaded from the optional `GameraXML - Training Data` input port) are appended to the database. Each is marked either `classify_manual` or `classify_automatic` depending on its original `id_state_manual` flag. These persist across sessions and provide prior knowledge.

**Step 3 — instantiate the classifier.**
```python
classifier = gamera.knn.kNNInteractive(
    database=database,
    perform_splits=True,
    num_k=1
)
```

- `num_k=1`: pure 1-NN — classification always picks the single nearest training example. There is no voting.
- `perform_splits=True`: allows Gamera's internal classifier to split feature weights during training for finer discrimination between similar classes.

**Step 4 — load optional feature selection.**  
If a `GameraXML - Feature Selection` file was supplied, `classifier.load_settings(features_file_path)` restricts classification to the selected feature subset. Without this, Gamera uses all available features.

---

## Automatic Classification (`run_correction_stage`)

Called during the `CLASSIFYING` state on every user round-trip.

For each glyph that is **not** manually labelled:

1. Convert RLE → Gamera image.
2. `cknn.classify_glyph_automatic(gamera_glyph)` — mutates the glyph's internal classification state in-place (Gamera side).
3. `cknn.guess_glyph_automatic(gamera_glyph)` — returns `(result, confidence)`:
   - `result[0][1]` → the winning class name string
   - `confidence[0]` → a float (0–1); `0` if no confidence could be computed
4. The class name and confidence are written back into the glyph dict. Manual glyphs are left entirely untouched.

The frontend sorts glyphs by **ascending confidence** so that the most uncertain automatic classifications appear first, making manual review efficient.

---

## Grouping and Reclassification (`group_and_correct`)

Triggered when the user submits with the `auto_group` action. Intended for multi-part characters (e.g., letters with diacritics) where each stroke arrives as a separate connected component.

**Step 1 — train a classifier** (same `prepare_classifier` as above).

**Step 2 — choose a grouping function.**  
Two spatial proximity strategies, both parameterised by `distance` (pixel threshold):

| Option | Gamera class | Behaviour |
|---|---|---|
| `"Shaped"` | `ShapedGroupingFunction(distance)` | Uses the actual pixel shape to measure proximity |
| `"BoundingBox"` | `BoundingBoxGroupingFunction(distance)` | Uses bounding-box overlap/distance |

**Step 3 — group and classify.**
```python
add, remove = cknn.group_list_automatic(
    gamera_glyphs,
    grouping_function=func,
    max_parts_per_group=parts,
    max_graph_size=graph,
    criterion=criterion
)
```

Gamera builds a graph of spatially adjacent glyphs, groups candidates up to `max_parts_per_group` parts and `max_graph_size` graph nodes, and reclassifies the merged images. It returns:
- `remove`: original component glyphs that were absorbed into groups (deleted from the working set)
- `add`: newly created grouped glyphs with updated class names and confidence scores

The working glyph list is updated by removing the originals and appending the new grouped glyphs as `GameraGlyph` dicts. Manual glyphs are excluded from grouping and re-inserted afterward.

---

## Manual Operations

### Manual Group (`manual_group`)

The user selects two or more glyphs and assigns a class name. The backend calls:
```python
grouped = image_utilities.union_images(gamera_glyphs)
```
`union_images` computes the pixel-wise union (bitwise OR) of all selected glyph images and produces a single new `ONEBIT` image whose bounding box encompasses all inputs. The resulting glyph gets `id_state_manual=True` and `confidence=1`, so it immediately serves as a training example.

### Manual Split (`manual_split`)

The user selects a glyph and a segmentation algorithm. The backend calls:
```python
splits = segmentation.<split_type>(gamera_glyph)
```
`split_type` is a Gamera segmentation plugin name (e.g., `cc_analysis`). The plugin decomposes the glyph image into sub-images. Each split result starts as `UNCLASSIFIED` with `confidence=0` and will be re-classified in the next `CLASSIFYING` round.

---

## Output (`output_corrected_glyphs`)

At export time, after a final `prepare_classifier` call, each glyph is written to GameraXML:

| Glyph state | Action taken |
|---|---|
| `id_state_manual=True` | `classify_manual(class_name)` — written as MANUAL state in the XML |
| Auto-classified, Training Data port was supplied, class ≠ `"UNCLASSIFIED"` | `classification_state = AUTOMATIC`, `id_name = [(confidence, class_name)]` — preserves the previously computed result |
| Auto-classified, Training Data port was supplied, class = `"UNCLASSIFIED"` | `classify_glyph_automatic(gamera_image)` — re-runs classification |
| Auto-classified, no Training Data port | `classify_glyph_automatic(gamera_image)` — runs classification for any glyph with a non-default class |

After all glyphs are prepared, `cknn.generate_features_on_glyphs(output_images)` computes and embeds feature vectors for every glyph. The full set is then written by `gamera.gamera_xml.WriteXMLFile(glyphs=output_images, with_features=True)`.

---

## Key Design Decisions

- **Full re-train on every round.** The classifier is discarded and rebuilt from scratch after every user submission. This keeps the model consistent with any manual corrections or deletions the user made, at the cost of re-computing features each time.
- **k=1.** Classification is winner-takes-all based on the single nearest training example. This makes the classifier sensitive to the representativeness of the training set.
- **Manual glyphs are training data, not candidates.** Any glyph the user marks as manual is excluded from automatic classification and contributes directly to the training pool. The boundary between "what gets classified" and "what trains the classifier" is purely the `id_state_manual` flag.
- **Confidence drives review order.** The frontend sorts by ascending confidence so users can focus their attention on the glyphs the classifier is least sure about.
- **Special-prefix glyphs are ephemeral.** Glyphs prefixed `_split`, `_group`, or `_delete` are always stripped by `filter_parts()` before training and before export; they exist only to communicate transient user intent between the frontend and the state machine.
