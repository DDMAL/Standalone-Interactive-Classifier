# syntax=docker/dockerfile:1
#
# Single-origin production image for the Interactive Classifier.
#
# Stage 1 builds the React/Vite frontend into static assets; stage 2 installs
# the FastAPI service and drops those assets into api/src/ic_api/static/, where
# main.py mounts them (see the StaticFiles mount near the bottom of main.py).
# The result serves both the API and the SPA from one origin on port 8000.
#
# Build context is the `ic/` directory:
#   docker build -t ic .
#   docker run --rm -p 8000:8000 ic

# ---------------------------------------------------------------------------
# Stage 1 — build the frontend (outputs to /build/dist)
# ---------------------------------------------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /build

# Install deps first so this layer is cached unless the lockfile changes.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Build: `tsc --noEmit && vite build`. devDependencies (typescript, vite) are
# present because `npm ci` installs them by default.
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — Python API serving the built SPA
# ---------------------------------------------------------------------------
# Bundles a recent uv with Python 3.12. Pin a specific uv version (e.g.
# ghcr.io/astral-sh/uv:0.11-python3.12-bookworm-slim) for reproducible builds.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS api

WORKDIR /app

# `ic-api` depends on `ic-core` via the sibling path `../core/ic_core`
# (see api/pyproject.toml [tool.uv.sources]), so both packages must be present
# with their relative layout preserved.
COPY core/ ./core/
COPY api/ ./api/

WORKDIR /app/api
# Install into the project venv from the committed lockfile, excluding dev deps.
RUN uv sync --frozen --no-dev

# Drop the built frontend where main.py expects it. This dir is gitignored;
# the COPY above does not include it (the build context excludes it too).
COPY --from=frontend /build/dist/ ./src/ic_api/static/

# uvicorn must bind all interfaces inside the container; run() reads these.
ENV HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

# --no-sync: the venv is already synced above, so skip the implicit re-sync.
CMD ["uv", "run", "--no-sync", "ic-api"]
