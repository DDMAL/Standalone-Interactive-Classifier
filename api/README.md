# ic-api

FastAPI service that wraps [`ic_core`](../core/ic_core) and exposes
the Interactive Classifier as HTTP endpoints for the frontend.

This is Phase 2 of the migration plan (`../docs/migration_plan.md`).

Session storage is selected at startup from the environment:

- With **`DATABASE_URL`** (or `IC_DATABASE_URL`) set, sessions are
  persisted to Postgres via
  [`db_store.PersistentSessionStore`](src/ic_api/db_store.py) — an
  in-memory hot cache with write-through on every mutation, hydrated
  from the DB on a cache miss, so sessions survive a restart. When the
  embedding host (mothra) stamps a session with its owning
  `(project_id, image_id)`, the session becomes resumable via
  `GET /sessions/lookup` (see below).
- Without it, the process-local
  [`InMemorySessionStore`](src/ic_api/store.py) is used — sessions are
  lost on restart, which is fine for the single-user / local-tool
  target and needs no external dependency.

## Run

```bash
uv sync
uv run ic-api          # binds to 127.0.0.1:8000
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/sessions` | Create a session and ingest a page + bbox file |
| `GET` | `/sessions/lookup?project_id=&image_id=` | Resumable session id for a mothra page, or 404 |
| `GET` | `/sessions/{id}` | Fetch the current session state |
| `POST` | `/sessions/{id}/classify` | Run a classify round |
| `POST` | `/sessions/{id}/glyphs/{gid}` | Update a single glyph |
| `DELETE` | `/sessions/{id}/glyphs/{gid}` | Delete a glyph |
| `POST` | `/sessions/{id}/group` | Manual group (union N glyphs) |
| `POST` | `/sessions/{id}/auto-group` | **501** — deferred (needs page-coord input) |
| `POST` | `/sessions/{id}/classes/{name}/rename` | Rename a class |
| `DELETE` | `/sessions/{id}/classes/{name}` | Delete a class from autocomplete |
| `POST` | `/sessions/{id}/save` | Persist the session (flush) with the Postgres store; no-op in-memory. Returns current state |
| `POST` | `/sessions/{id}/complete` | Transition to EXPORT, returns GameraXML |
| `DELETE` | `/sessions/{id}` | Discard the session |

See [`src/ic_api/main.py`](src/ic_api/main.py) for the full schemas.
