# ic_new

A modern rewrite of the **Interactive Classifier** — a tool for interactively classifying chant-manuscript neumes using a k-Nearest Neighbors model.

This project replaces the legacy Rodan job (Django + Celery + Gamera + Backbone.Marionette) at [`/Rodan-lite/backend/django/code/jobs/interactive_classifier/`] with a non-Django Python service, a React + Vite frontend, and no Gamera dependency.

## Key differences from the legacy IC

- **Input is a full page image plus a bounding-box annotation file** (for example, MOTHRA JSON or YOLO), not page-level GameraXML. Neume crops are derived from those annotations rather than supplied as pre-cropped glyph images.
- **Manual split (CCA on a single glyph) is deferred.** Reintroduce only if real data shows crops that contain multiple neumes.
- **Output stays as GameraXML** so downstream MEI pipelines keep working.

## Status

| Layer       | Path         | Status                                |
|-------------|--------------|---------------------------------------|
| Algorithm core | `core/ic_core/` | Implemented — manual workflow complete; auto-grouping and splitting intentionally deferred |
| API         | `api/`       | Implemented (FastAPI + in-memory store); `POST /auto-group` returns 501 |
| Frontend    | `frontend/`  | Not started                           |

## Repository layout

```
ic_new/
├── core/
│   ├── ic_core/        # Phase 1: algorithm core (uv-managed Python package)
│   ├── tests/          # Pytest suite + fixtures
│   ├── data/           # train/, test/, derived/ (derived/ is gitignored)
│   └── scripts/        # CLI helpers: run_pipeline, convert_hufnagel_csv, visualize, …
├── api/                # Phase 2: FastAPI service
├── frontend/           # Phase 3: React + Vite UI (not yet started)
└── docs/
    ├── CLAUDE.md       # Guidance for Claude Code working in this repo
    ├── migration_plan.md
    └── KNN_ALGORITHM.md
```

## Quickstart (algorithm core)

The core package uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
cd core/ic_core
uv sync                                    # install dependencies
uv run pytest                              # run tests (auto-regenerates training XML on first run)
uv run python ../scripts/run_pipeline.py   # end-to-end smoke: train → classify → overlays
uv run ruff check .                        # lint
```

The API has its own uv project under [`api/`](api/); see [`api/README.md`](api/README.md) for endpoint reference and run instructions.

## Documentation

- [docs/migration_plan.md](docs/migration_plan.md) — full migration strategy, phasing, and risks
- [docs/KNN_ALGORITHM.md](docs/KNN_ALGORITHM.md) — algorithm spec and invariants
- [docs/CLAUDE.md](docs/CLAUDE.md) — architecture notes and conventions for AI-assisted development
