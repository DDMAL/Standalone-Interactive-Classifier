# Standalone Interactive Classifier

A modern rewrite of the **Interactive Classifier** — a tool for interactively classifying chant-manuscript neumes using a k-Nearest Neighbors model.

The Interactive Classifier lives here: https://ic.simssa.ca/.

This project replaces the legacy Rodan job (Django + Celery + Gamera + Backbone.Marionette) at [`/Rodan-lite/backend/django/code/jobs/interactive_classifier/`] with a non-Django Python service, a React + Vite frontend, and no Gamera dependency.

## Key differences from the legacy IC

- **Input is a full page image plus a bounding-box annotation file** (for example, MOTHRA JSON or YOLO), not page-level GameraXML. Neume crops are derived from those annotations rather than supplied as pre-cropped glyph images.
- **Manual split is driven by user-drawn rectangles in page coordinates**, not Gamera CCA on a glyph image.
- **Auto-grouping is deferred** to a follow-up — manual grouping covers v1.
- **Output stays as GameraXML** so downstream MEI pipelines keep working.

## Status

| Layer       | Path         | Status                                |
|-------------|--------------|---------------------------------------|
| Algorithm core | [core/ic_core/](core/ic_core/) | Implemented — manual classify / group / split workflow complete; auto-grouping deferred |
| API         | [api/](api/) | Implemented (FastAPI + in-memory store); `POST /auto-group` returns 501 |
| Frontend    | [frontend/](frontend/) | Implemented — React + Vite UI for upload, classify, manual group / split, vocab selection |

## Repository layout

```
ic_new/
├── core/
│   ├── ic_core/        # Phase 1: algorithm core (uv-managed Python package)
│   ├── tests/          # Pytest suite + fixtures
│   ├── data/           # train/, test/, derived/ (derived/ is gitignored)
│   └── scripts/        # CLI helpers: run_pipeline, convert_hufnagel_csv, visualize, …
├── api/                # Phase 2: FastAPI service
├── frontend/           # Phase 3: React + Vite UI
└── docs/
    ├── CLAUDE.md       # Guidance for Claude Code working in this repo
    ├── migration_plan.md
    └── KNN_ALGORITHM.md
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — Python environment / dependency manager (used by the core and API)
- Python ≥ 3.11 (uv will install a matching interpreter if needed)
- Node.js + npm (for the frontend)

To run the full app locally, start the layers in order: **API first** (binds `127.0.0.1:8000`), then the **frontend** dev server (which proxies `/sessions`, `/training-sets`, and `/vocabularies` to the API).

## Quickstart (algorithm core)

The core package uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
cd core/ic_core
uv sync                                    # install dependencies
uv run pytest ../tests                      # run tests (auto-regenerates training XML on first run)
uv run python ../scripts/run_pipeline.py   # end-to-end smoke: train → classify → overlays
uv run ruff check .                        # lint
```

## Quickstart (API)

The API has its own uv project under [api/](api/). Start it before the frontend.

```bash
cd api
uv sync                                    # install dependencies (pulls in sibling ic-core)
uv run ic-api                              # binds to 127.0.0.1:8000
uv run pytest                              # run the API test suite
```

Override the bind address with the `HOST` / `PORT` env vars. CORS is preconfigured for the
Vite dev origin (`localhost:5173` / `127.0.0.1:5173`). See [api/README.md](api/README.md)
for the full endpoint reference.

## Quickstart (frontend)

```bash
cd frontend
npm install
npm run dev                                # Vite dev server on :5173, proxies API calls to 127.0.0.1:8000
npm run check                              # Biome lint/format check
npm run build                              # type-check + production build
```

## Documentation

- [docs/migration_plan.md](docs/migration_plan.md) — full migration strategy, phasing, and risks
- [docs/KNN_ALGORITHM.md](docs/KNN_ALGORITHM.md) — algorithm spec and invariants
- [docs/CLAUDE.md](docs/CLAUDE.md) — architecture notes and conventions for AI-assisted development
- [docs/USER_MANUAL.md](docs/USER_MANUAL.md) — detailed instructions for user and QA
