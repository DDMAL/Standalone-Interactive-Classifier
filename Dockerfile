# Single-image deploy for the Interactive Classifier demo.
#
# One container serves both the API (FastAPI) and the built frontend from the
# same origin, so there is no CORS and no separate API URL to configure (the
# frontend client falls back to same-origin requests in production — see
# frontend/src/api/client.ts). Build context must be the repo root because the
# API package depends on the sibling core/ic_core package via a relative path.
#
#   docker build -t ic-demo .
#   docker run -p 8000:8000 ic-demo   # then open http://localhost:8000

# ---------------------------------------------------------------------------
# Stage 1 — build the React/Vite frontend into static assets
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /build/frontend
# Install deps first (cached unless the lockfile changes). The build runs a
# TypeScript type-check, so dev dependencies are required — a full `npm ci`.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build   # → /build/frontend/dist

# ---------------------------------------------------------------------------
# Stage 2 — Python runtime: API + core library, plus the built frontend
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# uv for dependency management (matches local dev tooling).
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app
# Preserve the repo layout so api's relative dependency on ../core/ic_core
# resolves and the runtime data-dir env vars below point at real directories.
COPY core/ ./core/
COPY api/ ./api/

# Install the API and its editable core dependency into a project venv.
WORKDIR /app/api
RUN uv sync --frozen --no-dev

# Pre-build the Hufnagel training-set XML from the committed annotation/image
# pairs under core/data/train (the prebuilt XML in core/data/derived is
# gitignored, so it is regenerated here). If this ever fails the demo still
# runs — the training-set dropdown is just empty and the user labels glyphs
# interactively.
WORKDIR /app/core/ic_core
RUN uv run python ../scripts/convert_hufnagel_csv.py \
    || echo "WARNING: training-set build failed; dropdown will be empty"

# Drop the built frontend where main.py mounts it (api/src/ic_api/static).
COPY --from=frontend /build/frontend/dist /app/api/src/ic_api/static

WORKDIR /app/api
# Bind to all interfaces and honour the platform-provided $PORT (Render, etc.).
ENV HOST=0.0.0.0 \
    IC_TRAIN_DIR=/app/core/data/train \
    IC_DERIVED_DIR=/app/core/data/derived
EXPOSE 8000
CMD ["uv", "run", "ic-api"]
