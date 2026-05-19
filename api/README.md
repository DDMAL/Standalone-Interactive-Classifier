# ic-api

FastAPI service that wraps [`ic_core`](../core/ic_core) and exposes
the Interactive Classifier as HTTP endpoints for the frontend.

This is Phase 2 of the migration plan (`../docs/migration_plan.md`).
The service stores session state in-memory only — sessions are lost
on restart, which is fine for the single-user / local-tool target.
Swap [`store.py`](src/ic_api/store.py) for a SQLite-backed store
when persistence becomes a requirement.

## Run

```bash
uv sync
uv run ic-api          # binds to 127.0.0.1:8000
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/sessions` | Create a session and ingest a page + bbox file |
| `GET` | `/sessions/{id}` | Fetch the current session state |
| `POST` | `/sessions/{id}/classify` | Run a classify round |
| `POST` | `/sessions/{id}/glyphs/{gid}` | Update a single glyph |
| `DELETE` | `/sessions/{id}/glyphs/{gid}` | Delete a glyph |
| `POST` | `/sessions/{id}/group` | Manual group (union N glyphs) |
| `POST` | `/sessions/{id}/auto-group` | **501** — deferred (needs page-coord input) |
| `POST` | `/sessions/{id}/classes/{name}/rename` | Rename a class |
| `DELETE` | `/sessions/{id}/classes/{name}` | Delete a class from autocomplete |
| `POST` | `/sessions/{id}/save` | No-op for the in-memory store; returns current state |
| `POST` | `/sessions/{id}/complete` | Transition to EXPORT, returns GameraXML |
| `DELETE` | `/sessions/{id}` | Discard the session |

See [`src/ic_api/main.py`](src/ic_api/main.py) for the full schemas.
