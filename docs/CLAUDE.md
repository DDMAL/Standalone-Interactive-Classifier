# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Module Does

The Interactive Classifier is a Rodan job that lets users interactively classify document glyphs (symbols/characters) using a k-Nearest Neighbors algorithm from the Gamera library. It consists of a Python/Django backend (a `RodanTask` subclass) and a JavaScript SPA frontend (Backbone.Marionette).

## Frontend Build and Test Commands

All frontend commands run from `ic_frontend/`:

```bash
cd ic_frontend
npm install         # first-time setup
npm run build       # full build via Gulp (JS, CSS, JSDoc)
npm test            # Jest unit tests
npm run lint        # ESLint + JSCS style check (fails on errors)
gulp watch          # watch and rebuild during development
gulp rebuild:js     # quick JS-only rebuild without clean step
```

Compiled output goes to `static/js/compiled/classifier.min.js` and `static/css/classifier.min.css`. The HTML template at `interfaces/interactive_classifier.html` loads those static files.

To run a single Jest test file:
```bash
cd ic_frontend && npx jest public/js/test/models/Glyph.test.js
```

## Architecture Overview

### Backend State Machine

`wrapper.py` implements a `RodanTask` with a multi-stage state machine stored in Django settings between user interactions:

```
IMPORT_XML → CLASSIFYING → GROUP_AND_CLASSIFY → SAVE → EXPORT_XML
```

- `get_my_interface()` serializes glyph data to JSON, injects it into the Django template, and returns the HTML for the Rodan UI.
- `validate_my_user_input()` receives the user's JSON payload and applies mutations (group, split, delete, reclassify) to the in-memory glyph set.
- `run_my_task()` drives state transitions and calls the Gamera kNN classifier between user interactions. It returns `WAITING_FOR_INPUT()` to pause the job until the user submits corrections.

`interactive_classifier.py` contains the pure classification logic: training the `kNNInteractive` model, running auto-classification, grouping, and exporting GameraXML.

### Glyph Lifecycle and Special Prefixes

Glyphs carry an internal UUID to survive client/server round-trips. Three name prefixes mark transient state — `filter_parts()` strips these before export:

| Prefix | Meaning |
|--------|---------|
| `_split` | Marked for splitting via Gamera segmentation |
| `_group` | Marked for merging (image union) |
| `_delete` | Marked for deletion |

### Intermediary Layer

`intermediary/` provides format conversion between Gamera internals and the web layer:
- `GameraXML` — parses `.xml` files via `glyphs_from_xml()` and returns dicts
- `GameraGlyph` — wraps a Gamera image with class name, confidence, and manual state; serializes to `to_dict()` for JSON
- `RunLengthImage` — bidirectional converter: RLE binary string ↔ PIL Image ↔ base64 ↔ Gamera `ONEBIT/DENSE` image

### Frontend Event Architecture

The SPA uses **Backbone.Radio** channels as its pub/sub event bus. All cross-component communication goes through channels defined in `radio/RadioChannels.js`, not direct method calls:

- `edit` channel — glyph selection, zoom, split, group
- `modal` channel — open/close confirmation and error dialogs
- `menu` channel — main menu button actions

Event constant names live in `events/` (e.g. `GlyphEvents.js`, `ClassEvents.js`). Views listen on channels and trigger events; they never reference each other directly.

`auth/Authenticator.js` fires a token-refresh AJAX call every 5 seconds to keep the Rodan session alive while the user works.

### Rodan Integration Points

- `resource_types.yaml` declares the MIME types this job accepts/produces (`application/gamera+xml`, various `image/*+png`, `text/plain`).
- `__init__.py` registers the job with Rodan's module loader (version `"1.0.0"`).
- `gamera_xml_distributor.py` is a separate, simple `RodanTask` that copies a GameraXML file from input to output — used in pipelines where the file needs to fan out.
