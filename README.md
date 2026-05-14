# ic_new

A modern rewrite of the **Interactive Classifier** — a tool for interactively classifying chant-manuscript neumes using a k-Nearest Neighbors model.

This project replaces the legacy Rodan job (Django + Celery + Gamera + Backbone.Marionette) at [`../Rodan-lite/backend/django/code/jobs/interactive_classifier/`](../Rodan-lite/backend/django/code/jobs/interactive_classifier/) with a non-Django Python service, a React + Vite frontend, and no Gamera dependency.

## Key differences from the legacy IC

- **Input is pre-cropped neume images** (one image per glyph), not page-level GameraXML. The upstream connected-components-analysis (CCA) stage is removed.
- **Manual split (CCA on a single glyph) is deferred.** Reintroduce only if real data shows crops that contain multiple neumes.
- **Output stays as GameraXML** so downstream MEI pipelines keep working.

## Status

| Layer       | Path         | Status                                |
|-------------|--------------|---------------------------------------|
| Algorithm core | `core/ic_core/` | In progress — scaffolded, partial impl |
| API         | `api/`       | Not started                           |
| Frontend    | `frontend/`  | Not started                           |

## Repository layout

```
ic_new/
├── core/
│   ├── ic_core/        # Phase 1: algorithm core (uv-managed Python package)
│   └── tests/          # Pytest suite + fixtures
├── api/                # Phase 2: FastAPI service (planned)
├── frontend/           # Phase 3: React + Vite UI (planned)
└── docs/
    ├── CLAUDE.md       # Guidance for Claude Code working in this repo
    ├── migration_plan.md
    └── KNN_ALGORITHM.md
```

## Quickstart (algorithm core)

The core package uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
cd core/ic_core
uv sync                 # install dependencies
uv run pytest ../tests  # run tests
uv run ruff check .     # lint
```

## Documentation

- [docs/migration_plan.md](docs/migration_plan.md) — full migration strategy, phasing, and risks
- [docs/KNN_ALGORITHM.md](docs/KNN_ALGORITHM.md) — algorithm spec and invariants
- [docs/CLAUDE.md](docs/CLAUDE.md) — architecture notes and conventions for AI-assisted development
