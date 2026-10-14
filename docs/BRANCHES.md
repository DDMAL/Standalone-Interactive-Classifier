# `main` vs `develop`

**As of 2026-10-14 the two branches are unified: `main` and `develop` point at
the same commit.** Keep it that way — land changes on one and fast-forward the
other, rather than letting them drift into separate tracks again.

Verify:

```bash
git rev-parse origin/main origin/develop     # same SHA
git diff origin/main origin/develop          # empty
```

## History

Until 2026-10-14, `main` was the mothra-embedded track and `develop` the
standalone deployment track behind `ic.simssa.ca`; neither was a superset of
the other. A first sync attempt (2026-07-30) was backed out on `develop` with
`Revert "Merge branch 'main' into develop"` (`4244e8c`) and
`Revert "resolve conflict"` (`d27af7d`). Because git records those reverts as
deliberate `develop` changes, a plain merge would have silently stripped
`main`'s mothra features. The unifying merge therefore started from `main`'s
side of every reverted file and layered `develop`'s genuine additions on top.

## One codebase, two deployments

Where the two deployments genuinely need different behaviour, the frontend
branches at runtime on `window.parent !== window` ("embedded" in a host iframe,
i.e. mothra) rather than on the branch:

| Behaviour | Embedded (mothra) | Standalone (`ic.simssa.ca`) |
|---|---|---|
| Export lifecycle | `completeSession` sends `finalize=false`; session stays `CLASSIFYING`; `useComplete` stays on the edit view | backend default `finalize=true`; session goes terminal `EXPORT`; `useComplete` calls `clearSession()` |
| Export button / default | "Export ▾", whole page ticked | "Complete & Export ▾", manual neumes ticked |
| Toolbar | no home / "New session" buttons (host owns navigation) | "Interactive Classifier" home button and "New session" |
| `UploadView` effects | `ic:ready` / `ic:prefill-training` host prefill | preloads the bundled sample page (`frontend/public/samples/`) |
| Auto-export | posts `ic:session-created` + `ic:auto-export`, button reads "Queue page" | downloads the GameraXML, button reads "Auto-export" |

Shared by both: the mothra palette (`tailwind.config.ts` `mothra.*` colours),
export hygiene (`filter_parts`, no `UNCLASSIFIED` training glyphs), the
glyph-view toggle in the edit panel and split canvas, and the scroll fixes.

**mothra depends on `finalize`.** `landing-page/scripts/ic_api.py`'s
`verify_ic_finalize_support()` reads IC's OpenAPI schema at startup and aborts
if `finalize` is absent from `/sessions/{id}/complete`. Never drop it.

## Build and deploy

| What | Where |
|---|---|
| Tests | `.github/workflows/run-tests.yml` — core + api pytest on PRs to `main`. |
| Public demo | `.github/workflows/build-and-deploy.yml` — on push to `develop` (or manual dispatch), builds the `Dockerfile`, pushes `ghcr.io/ddmal/interactive-classifier:<branch>-<sha>` / `<branch>-latest`, then `kubectl apply -f k8s/interactive-classifier/` and rolls the Deployment. Deploy is gated on `refs/heads/develop`; a dispatch on any other ref only builds. |
| Kubernetes | `k8s/interactive-classifier/` — namespace, Deployment (1 replica), Service, Ingress (`ic.simssa.ca`, traefik) and a Middleware enforcing a 64 MiB request-body limit. |
| Render | `render.yaml` Blueprint — one free-plan Docker web service. |
| mothra | mothra's own CI builds `ghcr.io/ddmal/mothra-ic` from this repo's `Dockerfile` at its submodule pin. Keep the `Dockerfile` working with the repo root as build context and `$PORT` honoured. |

## Operational notes

**nginx body limit.** An nginx proxy in front of `ic.simssa.ca` rejects bodies
over its default `client_max_body_size` of 1 MB with its own 413 *before* the
request reaches uvicorn (verified at the byte: 1048576 bytes reach the app,
1048577 do not). If that nginx is a host-level proxy rather than an ingress
controller, no annotation in this repo can reach it — it needs
`client_max_body_size 64m;` in its own server block.

**`ic.simssa.ca` does not persist sessions.** The Deployment sets no
`IC_DATABASE_URL` / `DATABASE_URL`, so `build_default_store()` selects
`InMemorySessionStore`: every redeploy, OOM kill or scale-down drops all
in-flight sessions, and the next request fails with "Unknown session id".
`db_store.py` already supports a database — this is a manifest gap, not a
missing feature. `GET /healthz` reports it: `{"backend": "in-memory",
"persistent": false, "reachable": null}`.
