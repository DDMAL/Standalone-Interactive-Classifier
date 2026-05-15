# Interactive Classifier — Per-Stage I/O Reference

## State machine

```
                  ┌─────────────┐
                  │ IMPORT_XML  │  (entry, on first run)
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
       ┌────────► │ CLASSIFYING │ ◄────────┐
       │          └──────┬──────┘          │
  'undo'│        'auto_group'│      'save' │
  resets│                    ▼              │
       │       ┌──────────────────────┐    │
       │       │ GROUP_AND_CLASSIFY   ├────┤
       │       └──────────────────────┘    │
       │                ┌──────┐           │
       │                │ SAVE ├───────────┘
       │                └──────┘
       │           'complete'│
       │                     ▼
       │              ┌────────────┐
       └──────────────┤ EXPORT_XML │  (terminal)
                      └────────────┘
```

State enum at [interactive_classifier.py:21-27](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/interactive_classifier.py#L21-L27): `IMPORT_XML=0`, `CLASSIFYING=1`, `EXPORT_XML=2`, `GROUP_AND_CLASSIFY=3`, `SAVE=5`.

---

## Stage 1 — IMPORT_XML

| | |
|---|---|
| **Trigger** | First invocation of `run_my_task()` ([wrapper.py:118-119](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py#L118-L119)) |
| **Inputs (Rodan ports)** | `GameraXML - Connected Components` (required), `GameraXML - Training Data` (optional), `GameraXML - Feature Selection` (optional), `Plain Text - Class Names` (optional), `PNG Preview Image` |
| **Inputs (user)** | none |
| **Mutations** | Parses XML via `glyphs_from_xml()`; filters training set to `id_state_manual=True AND class_name not in {UNCLASSIFIED, _delete}` ([wrapper.py:127-136](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py#L127-L136)); reads class-name text file line-by-line |
| **Outputs (settings)** | `glyphs` (list[dict]), `training_glyphs` (list[dict]), `imported_class_names` (list[str]), `glyphs_json`, `training_json`, `class_names_json` |
| **Side effect** | If training data was supplied, immediately runs `run_correction_stage()` to pre-classify ([wrapper.py:156-158](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py#L156-L158)) |
| **Next state** | `CLASSIFYING` |

---

## Stage 2 — CLASSIFYING

| | |
|---|---|
| **Trigger** | Default branch in `validate_my_user_input()` — any submission that is not `complete`/`save`/`auto_group`/`group`/`split`/`delete`/`undo` ([wrapper.py:465-476](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py#L465-L476)) |
| **Inputs (user payload)** | `glyphs`, `grouped_glyphs`, `changed_training_glyphs`, `deleted_glyphs`, `deleted_training_glyphs`, `deleted_classes`, `renamed_classes` |
| **Inline-action variants** | (handled in validation, before state runs): `group` returns single new glyph with `id_state_manual=True, confidence=1`; `split` returns list with `class_name=UNCLASSIFIED, id_state_manual=False, confidence=0`; `delete` overwrites `class_name="_delete"` |
| **Mutations (ordered — do NOT reorder)** | `add_grouped_glyphs` → `update_changed_glyphs` → `remove_deleted_glyphs` → `remove_deleted_classes` → `update_renamed_classes` → `filter_parts` ([wrapper.py:168-180](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py#L168-L180)) |
| **Algorithm step** | `run_correction_stage()` — full re-train every round, `k=1`, classifies only non-manual glyphs ([interactive_classifier.py:225-254](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/interactive_classifier.py#L225-L254)) |
| **Outputs** | Updated `settings['glyphs']` (with new `class_name`/`confidence` on auto-classified glyphs); regenerated `glyphs_json`/`training_json`/`class_names_json` |
| **Next state** | `CLASSIFYING` (loops; `WAITING_FOR_INPUT()` returned) |

---

## Stage 3 — GROUP_AND_CLASSIFY

| | |
|---|---|
| **Trigger** | User submission contains `'auto_group'` key ([wrapper.py:422-433](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py#L422-L433)) |
| **Inputs (user payload)** | All CLASSIFYING fields **plus** `user_options: {distance: int, parts: int, graph: int, criterion: str, func: "Shaped"\|"BoundingBox"}` |
| **Mutations** | Same ordered chain as CLASSIFYING |
| **Algorithm step** | `group_and_correct()` — separates manual/non-manual; trains classifier; calls Gamera's `cknn.group_list_automatic(distance, parts, graph)`; replaces grouped originals with new merged glyphs; reinserts manual glyphs ([interactive_classifier.py:147-223](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/interactive_classifier.py#L147-L223)) |
| **Outputs** | `settings['glyphs']` with merged/regrouped entries; regenerated JSON |
| **Next state** | `GROUP_AND_CLASSIFY` (waits for input; user can leave via another action) |

---

## Stage 4 — SAVE

| | |
|---|---|
| **Trigger** | User submission contains `'save'` key ([wrapper.py:447-457](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py#L447-L457)) |
| **Inputs (user payload)** | Same as CLASSIFYING (no extra fields) |
| **Mutations** | Same ordered chain |
| **Algorithm step** | None — no classifier run. Just filter + serialize |
| **Outputs** | Persisted `settings` (no file written, no port output) |
| **Next state** | `SAVE` (waits for input; effectively returns to editing) |

Intent: snapshot intermediate progress without producing the final XML.

---

## Stage 5 — EXPORT_XML

| | |
|---|---|
| **Trigger** | User submission contains `'complete'` key ([wrapper.py:390-401](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py#L390-L401)) |
| **Inputs (user payload)** | Same mutation fields as CLASSIFYING |
| **Mutations** | Same ordered chain; then final `prepare_classifier()` over `training_glyphs + manual glyphs` ([wrapper.py:245](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py#L245)); then `filter_parts()` strips all `_split`/`_group`/`_delete` |
| **Outputs (files)** | `Classified Glyphs` (mandatory, GameraXML with `with_features=True`); `Training Data` (optional, kNN as XML); `Class Names` (optional, sorted text, excludes `UNCLASSIFIED`) — see [interactive_classifier.py:94-117](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/interactive_classifier.py#L94-L117) |
| **Glyph classification rule on export** | Manual → `classify_manual()`; auto with training data → `classification_state=AUTOMATIC, id_name=[(confidence, class_name)]`; otherwise → re-runs `cknn.classify_glyph_automatic()` |
| **Next state** | Terminal — job completes |

---

# Schema Appendix

## A. Glyph dict (canonical)

Produced by `GameraGlyph.to_dict()` ([intermediary/gamera_glyph.py:53-66](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/intermediary/gamera_glyph.py#L53-L66)) and `construct_glyph_dict()` ([interactive_classifier.py:9-26](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/interactive_classifier.py#L9-L26)).

| Field | Type | Description |
|---|---|---|
| `id` | str (32-hex UUID) | Stable across round-trips; new for grouped/split glyphs |
| `class_name` | str | Label, or transient prefix `_split` / `_group` / `_delete` (stripped by `filter_parts`) |
| `image` | str | RLE — space-separated ints (see §C) |
| `image_b64` | bytes | base64-encoded PNG for frontend `<img>`/`<canvas>` |
| `ncols` | int | image width |
| `nrows` | int | image height |
| `ulx` | int | bbox upper-left x on page |
| `uly` | int | bbox upper-left y on page |
| `id_state_manual` | bool | `True` = user-labeled → feeds training; `False` = auto-labeled → classified each round |
| `confidence` | float [0,1] | Manual: always 1.0; UNCLASSIFIED: 0.0; auto: from kNN |
| `is_training` | bool | Marks membership in training set (set in some manual_group/split paths) |

## B. `settings` dict keys

**Persistent (survive across rounds):**

| Key | Type | Meaning |
|---|---|---|
| `@state` | int (enum) | Current `ClassifierStateEnum` |
| `glyphs` | list[dict] | Working glyph set |
| `training_glyphs` | list[dict] | kNN training database |
| `imported_class_names` | list[str] | User-supplied vocabulary |
| `class_names` | list[str] | Sorted union of all class names seen |
| `glyphs_json` | str (JSON) | `{class_name: [glyph, ...]}` sorted asc by confidence |
| `training_json` | str (JSON) | Training + manual glyphs grouped by class |
| `class_names_json` | str (JSON) | Sorted class-name array |

**Transient (`@`-prefixed, populated by `validate_my_user_input`, consumed once by `run_my_task`):**

| Key | Type | Consumer |
|---|---|---|
| `@changed_glyphs` | list[dict] | `update_changed_glyphs` |
| `@grouped_glyphs` | list[dict] | `add_grouped_glyphs` |
| `@changed_training_glyphs` | list[dict] | `update_changed_glyphs` |
| `@deleted_glyphs` | list[dict] | `remove_deleted_glyphs` |
| `@deleted_training_glyphs` | list[dict] | (training removal) |
| `@deleted_classes` | list[str] | `remove_deleted_classes` |
| `@renamed_classes` | dict[str,str] | `update_renamed_classes` |
| `@user_options` | dict | GROUP_AND_CLASSIFY only |

## C. RLE format

[intermediary/run_length_image.py:22-55](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/intermediary/run_length_image.py#L22-L55) — a space-separated string of integers giving alternating run lengths, **starting with white**, scanned row-major. `"2 3 1 4"` = WW BBB W BBBB.

Conversions:
- → PIL/PNG/base64: `get_base64_image()` ([run_length_image.py:83-87](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/intermediary/run_length_image.py#L83-L87))
- → Gamera image: `get_gamera_image()` returns `ONEBIT/DENSE` (NOT `ONEBIT/RLE` — that segfaults; see [run_length_image.py:90](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/intermediary/run_length_image.py#L90))

## D. User-input payload variants

All carry the common mutation envelope (`glyphs`, `grouped_glyphs`, `changed_training_glyphs`, `deleted_glyphs`, `deleted_training_glyphs`, `deleted_classes`, `renamed_classes`). The action is selected by which extra key is present:

| Action key | Extra fields | Validator returns | Next state |
|---|---|---|---|
| `complete` | — | mutation envelope | `EXPORT_XML` |
| `group` | `glyphs: [dict]`, `class_name: str` | `{manual: True, glyph: <new_dict>}` | stays |
| `split` | `glyph: dict`, `split_type: str` | `{manual: True, glyphs: [<new_dicts>]}` | stays |
| `auto_group` | `user_options: dict` | mutation envelope + `@user_options` | `GROUP_AND_CLASSIFY` |
| `delete` | `glyphs: [dict]` | `{manual: True, glyphs: [...]}` with `class_name="_delete"` | stays |
| `save` | — | mutation envelope | `SAVE` |
| `undo` | — | — | `IMPORT_XML` |
| (none) | — | mutation envelope | `CLASSIFYING` |

`user_options` shape: `{distance: int, parts: int, graph: int, criterion: str, func: "Shaped"|"BoundingBox"}`.

## E. Frontend-bound JSON (`get_my_interface`)

Injected into the Django template at [wrapper.py:97-103](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py#L97-L103) — note all four values are **JSON strings**, not dicts:

```python
{
  'glyphs':          <JSON str>  # {class_name: [glyph_dict, ...]} sorted asc by confidence
  'image_path':      <str>       # URL to page preview PNG
  'class_names':     <JSON str>  # ["a", "b", ...]
  'training_glyphs': <JSON str>  # {class_name: [glyph_dict, ...]}
}
```

## F. GameraXML element structure

Parsed by `glyphs_from_xml()` ([intermediary/gamera_xml.py:62-75](../../Rodan-lite/backend/django/code/jobs/interactive_classifier/intermediary/gamera_xml.py#L62-L75)):

```xml
<glyphList>
  <glyph>
    <id name="a" state="MANUAL|AUTOMATIC" confidence="0.87"/>
    <ul x="1024" y="512"/>
    <ncols>32</ncols>
    <nrows>48</nrows>
    <image format="rle">2 3 1 4 ...</image>
    <features>...</features>   <!-- optional, written when with_features=True -->
  </glyph>
  ...
</glyphList>
```

---

## Invariants to preserve in the rewrite

1. **Mutation ordering is load-bearing.** `add_grouped → update_changed → remove_deleted → remove_deleted_classes → update_renamed → filter_parts`. Renaming after deletion silently drops the delete; reordering causes subtle data loss.
2. **Full re-train every round** — discard and rebuild the kNN on each submission ([KNN_ALGORITHM.md](KNN_ALGORITHM.md)).
3. **`id_state_manual` is the training/classification boundary** — manual feeds training; non-manual gets classified.
4. **UUIDs survive round-trips** for existing glyphs; new ones (group/split outputs) get fresh UUIDs.
5. **Manual group:** `id_state_manual=True, confidence=1`. **Manual split:** `class_name=UNCLASSIFIED, id_state_manual=False, confidence=0`.
6. **Confidence sort:** frontend expects ascending — lowest-confidence first.
7. **`_split` / `_group` / `_delete` prefixes** are transient and must be stripped by `filter_parts` before training and before export.
