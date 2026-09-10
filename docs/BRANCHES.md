# `main` vs `develop`

Snapshot: `main` @ `553c1e6` (2026-08-24), `develop` @ `e998c10` (2026-08-21).
Merge base: `7a46ac5` (2026-08-21, *"test: pin the grouped-mask invariant
rebinarize relies on"*). `main` is 5 commits ahead, `develop` 38 commits ahead —
these are **two diverged tracks, not one behind the other**.

Re-derive any of this with:

```bash
git rev-list --left-right --count origin/main...origin/develop
git log --oneline origin/main..origin/develop      # develop-only
git log --oneline origin/develop..origin/main      # main-only
git diff --stat origin/main origin/develop
```

## The one-sentence difference

**`main` is the mothra-embedded track**; **`develop` is the standalone
deployment track** that runs the public demo at `ic.simssa.ca`. `main` carries
the host-integration behaviour and mothra's visual identity; `develop` carries
the container/Kubernetes/Render deployment plumbing and the standalone
self-service polish. Neither is a superset of the other.

## What is identical

Everything under `core/` — the whole algorithm layer: glyph model, RLE,
binarisation (including the strip-wise Sauvola), the 29-d feature set, the kNN,
grouping, splitting, the `Session` state machine and `rebinarize`, GameraXML
read/write. Also identical: `api/src/ic_api/store.py`, `db_store.py`,
`schemas.py`, and every file under `docs/`.

So a classification result is the same on both branches. The divergence is
entirely deployment, export semantics, and UI.

Verify: `git diff --stat origin/main origin/develop -- core/ docs/` is empty.

## `develop` only — deployment and standalone polish

| Area | What |
|---|---|
| **CI/CD** | `.github/workflows/build-and-deploy.yml`: on push to `develop` (or manual dispatch), builds the Dockerfile, pushes `ghcr.io/ddmal/interactive-classifier` tagged `<branch>-<sha>` and `<branch>-latest`, then `kubectl apply -f k8s/interactive-classifier/` and `set image` + `rollout status`. Deploy is gated on `github.ref == refs/heads/develop`. |
| **Kubernetes** | `k8s/interactive-classifier/` — namespace, Deployment (1 replica, `develop-latest`, `ghcr-pull-secret`, tcpSocket probes, 200m/512Mi → 500m/1Gi), Service, Ingress (`ic.simssa.ca`, traefik), and a Middleware enforcing a 64 MiB request-body limit. |
| **Render** | `render.yaml` Blueprint — one free-plan Docker web service, `healthCheckPath: /`. |
| **Sample data** | `frontend/public/samples/image_hfn_sample.png` (~4 MB) + its annotations JSON, pre-loaded by `UploadView` so the form is submittable in one click. Also shows a "Loaded: `<filename>`" hint under each file input. |
| **Dockerfile** | A different build: `node:20-slim` + `python:3.11-slim` with `uv` copied in, `UV_LINK_MODE=copy`, `UV_COMPILE_BYTECODE=1`, and `IC_TRAIN_DIR=/app/core/data/train` baked in so `/vocabularies` finds the committed CSVs. It sets no `PORT`, honouring a platform-injected `$PORT`. `main`'s is `uv:python3.12-bookworm-slim`, pins `PORT=8000`, and runs `uv run --no-sync`. |
| **`.dockerignore`** | Shorter and looser; also excludes `core/data/test` and `docs`. `main`'s is stricter and additionally excludes `api/src/ic_api/static/`, `dist`, caches and `*.tsbuildinfo`. |
| **UI wording** | Toolbar has a clickable "Interactive Classifier" home button and a "New session" button (both `clearSession()`); export reads **"Complete & Export ▾"**. |
| **Neutral palette** | `blue-600`/`slate` Tailwind utilities throughout; `tailwind.config.ts` has no custom colour extension. |
| **Test** | `test_list_sessions_scopes_to_project_id` — `?project_id=` must not surface another project's or an unkeyed session. (The feature itself exists on both branches; only this test is develop-only.) |

The ingress annotation carries a real operational finding worth preserving: an
nginx proxy in front of `ic.simssa.ca` rejects bodies over its default
`client_max_body_size` of 1 MB with its own 413 *before* the request reaches
uvicorn (verified at the byte: 1048576 bytes reach the app, 1048577 do not). If
that nginx is a host-level proxy rather than an ingress controller, no
annotation in this repo can reach it — it needs `client_max_body_size 64m;` in
its own server block.

## `main` only — mothra integration

| Area | What |
|---|---|
| **`finalize` on export** | `POST /sessions/{id}/complete?finalize=false` applies the export hygiene to the exported copy only, leaving the session in `CLASSIFYING`, editable and resumable. `develop` has no such parameter — every export finalises. |
| **Frontend uses it** | `api/sessions.ts` sets `finalize=false` whenever `window.parent !== window`. |
| **Export doesn't end the session** | `useComplete` stays on the edit view after the download, so the user can keep correcting and export again. On `develop` it calls `clearSession()` and drops back to the main page — correct there, because the session really is terminal. |
| **`ic:prefill-training`** | `UploadView` announces `ic:ready` and adopts the host's batch-level training selection, normalised to the one-preset invariant (`normalizePresets`), never clobbering a local selection. `develop` uses that effect slot for the sample-file preload instead. |
| **Embedded auto-export** | `useAutoExport` posts `ic:session-created` + `ic:auto-export` and returns instead of downloading, so the host drives completion server-side and the page joins its encode queue. The button becomes "Queue page". On `develop` it always downloads. |
| **Glyph-view toggle reaches everywhere** | `useGlyphImageSrc` (exported from `GlyphImage.tsx`) gives the edit panel and the split canvas the same binarised/original choice the grid honours. `develop` hard-codes `glyphDataUri(glyph)` — the binarised mask — in both places. |
| **Mothra palette** | `tailwind.config.ts` defines `mothra.{cyan,cyan-dark,cyan-faint,cyan-muted,teal}` (`#4AADAA` / `#1E6B70` / `#C8E6E3` / `#B0CDC9` / `#1D3335`) and Button/Toolbar/ClassTree/Lasso/RightDock use them, so the iframe matches its host. |
| **CI** | `.github/workflows/run-tests.yml` — core + api pytest on PRs to `main`. No image build, no deploy. |
| **README** | Points at https://ic.simssa.ca/. |

## Consequences of the split

**mothra can only run `main`.** `landing-page/scripts/ic_api.py`'s
`verify_ic_finalize_support()` reads IC's OpenAPI schema at backend startup and
aborts if `finalize` is absent from `/sessions/{id}/complete` — because such an
IC ignores the unknown query parameter, finalises anyway, and strands the page's
corrections with no error raised anywhere. A `develop`-built image would trip
that check. (`MOTHRA_IC_COMPAT_CHECK=0` bypasses it, but the underlying data
loss is real, so don't.) mothra's submodule pin is `553c1e6` — `main`'s tip.

**`ic.simssa.ca` does not persist sessions.** `develop`'s Deployment sets only
`HOST` and `PORT`; with neither `IC_DATABASE_URL` nor `DATABASE_URL` present,
`build_default_store()` selects `InMemorySessionStore`, so a redeploy, an OOM
kill or a scale-down drops every in-flight session and the next request fails
with "Unknown session id". The code for the alternative is already there and
identical on both branches (`db_store.py`) — this is a manifest gap, not a
missing feature. `GET /healthz` reports it: `{"backend": "in-memory",
"persistent": false, "reachable": null}`.

**`main` has no way to build or deploy an image.** No GHCR build, no manifests.
Its container story is the Dockerfile plus mothra's own CI, which builds IC as
one of its four images (`ghcr.io/ddmal/mothra-ic`) from the submodule pin.

**Merging is not mechanical.** `develop`'s history contains
`Revert "Merge branch 'main' into develop"` and `Revert "resolve conflict"`
(both 2026-07-30), i.e. a previous attempt to sync the two was deliberately
backed out. The genuine conflicts are the palette (mothra-branded vs neutral),
the export lifecycle (`finalize=false` + stay-on-page vs finalise + clear), and
`UploadView`'s single `useEffect` slot (host prefill vs sample preload) —
the last two are the ones that need a real decision, since both behaviours are
correct for their own deployment and the branch condition (`window.parent !==
window`) already exists to carry both. The Dockerfile and `.dockerignore`
differences are straightforward to reconcile in `main`'s favour.

## If you are picking a branch

- Working on the algorithm, the API, or anything mothra sees → **`main`**.
- Working on the public demo, the image, k8s or Render → **`develop`**.
- Touching `core/` → it is identical on both, so land it wherever, but expect to
  cherry-pick it to the other branch; nothing merges the two automatically.
