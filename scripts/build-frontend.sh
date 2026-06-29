#!/usr/bin/env bash
#
# Build the React/Vite frontend and copy it into the API's static mount so the
# FastAPI app can serve the SPA from a single origin (see the StaticFiles mount
# in api/src/ic_api/main.py). This reproduces what the Dockerfile does, for a
# local production-style run without Docker.
#
# The destination (api/src/ic_api/static/) is gitignored — it is a build
# artifact, not source. Run the API afterwards and the SPA is served at /.
#
# Usage:
#   ic/scripts/build-frontend.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frontend_dir="$repo_root/frontend"
static_dir="$repo_root/api/src/ic_api/static"

echo "==> Building frontend in $frontend_dir"
( cd "$frontend_dir" && npm ci && npm run build )

echo "==> Replacing $static_dir with the fresh build"
rm -rf "$static_dir"
cp -r "$frontend_dir/dist" "$static_dir"

echo "==> Done. Start the API (uv run ic-api) and open http://127.0.0.1:8000/"
