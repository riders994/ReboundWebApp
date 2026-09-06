# Rohan Vahalia — personal site

Personal portfolio site (About, Projects, Resume, Blog, Comedy) plus a live interactive
**NBA rebound predictor** demo. Originally a single 2017 Flask app; revamped into a static
portfolio with a small ML backend for the demo.

## Layout
```
site/            Static portfolio (plain HTML/CSS/JS, no build step) — nginx docroot
  index.html         landing + about
  projects/          project grid + rebounding case study & live demo
  resume.html        resume + PDF download
  blog/              posts
  comedy.html        stand-up / performances
  assets/            css, js, images, svg
    img/fallbacks/     default card covers, used when a card has no image of its own
rebound-app/     Flask JSON API for the demo (Python 3) — no static/template serving
  app.py             HTTP only; all features come from the `rebounding` package
  coordinates.py     canvas<->model frame conversion — read before touching geometry
  movement.py        movement model adapter; optional and fail-soft
  telemetry.py       optional prediction logging to Postgres; off by default, fails soft
  config.py          all runtime config, from the environment
  wsgi.py            gunicorn entry point (`wsgi:app`)
  gunicorn.conf.py   NB: preload_app must stay off — see the docstring
  tests/             62 tests, incl. a regression fixture from the pipeline itself
  models/            FinalModel.pkl + MovementModel.pkl — gitignored, copied in
deploy/          nginx.conf, rebound.service
DEPLOY.md        EC2 deployment runbook
```

## Run locally
Static site (refreshes directory-driven manifests, then serves):
```bash
./scripts/serve.sh          # open http://localhost:5500  (PORT=8080 to change)
```
Backend (in another terminal):
```bash
cd rebound-app
python3 -m venv venv
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -e '../../ReboundingPrediction[serve]'  # the feature code
./venv/bin/gunicorn -c gunicorn.conf.py wsgi:app               # http://127.0.0.1:8000

curl -s http://127.0.0.1:8000/healthz          # which model is live
curl -s -X POST http://127.0.0.1:8000/predict \
     -H 'Content-Type: application/json' --data @sample.json
./venv/bin/python -m pytest tests/ -q
```
To exercise the demo against the local backend, set the endpoint on the rebounding page
(`site/projects/rebounding.html`) `window.REBOUND_CONFIG.endpoint` to
`http://localhost:8000/predict`.

### Models
Weights never enter git. They are built in the **ReboundingPrediction** repo
(`python -m rebounding.cli train` / `train-movement`) and copied to
`rebound-app/models/`:

| file | what it does | if missing |
|---|---|---|
| `FinalModel.pkl` | rebound probabilities (LightGBM, grouped softmax over the 10 players) | app refuses to start |
| `MovementModel.pkl` | where players move (torch CVAE); replaces `posnn.h5` + `msd.pkl` | fails soft — players stay put, probabilities unaffected |

Both are `joblib.dump` of a dataclass bundle, not a bare estimator, so **unpickling
imports the `rebounding` package** — it is a hard serving dependency. Each bundle
carries its own provenance (build commit, corpus SHA, library versions, test scores),
which is what `/healthz` reports and what makes a copy-in workflow auditable.

The old model-free fallback is gone: a missing rebounder is now a refusal to start
rather than plausible-looking placeholder numbers served silently.

### Prediction logging
Optional, off unless `REBOUND_DATABASE_URL` is set, and shares the EC2 box's Postgres
with other projects under its own `rebound` schema. Each served prediction becomes a row:
the ten placements in model-frame coordinates, the probabilities returned, latency, and
the bundle's build commit. Two questions it exists to answer — how far real traffic sits
from the corpus the 29.7% below was measured on, and which build produced a given
prediction once the weights on disk have been replaced by a retrain.

Rows are written on a background thread from a bounded queue, so an unreachable database
costs dropped rows and never a failed or slower `/predict`; `/healthz` reports the sink
and its dropped count. Nothing recorded identifies a visitor — no IP, user agent, cookie
or session id. Setup is step 5d of [DEPLOY.md](DEPLOY.md); the table is created on first
write, so there is no migration to run.

### Default card covers
A card with no image of its own used to render an empty box. `site/assets/img/fallbacks/`
holds twelve abstract covers in the site's palette (SVG, light/dark aware) and any card
missing its `<slug>-thumb.png` gets one. `assets/js/thumbs.js` deals them from a shuffled
bag rather than picking per card, so no page repeats a cover while unused ones remain, and
the shuffle is seeded from the manifest so a project keeps the same cover across reloads
and between the projects grid and the resume carousel. It's the `fallbacks` photo feed —
drop images in, run `python3 scripts/gen-manifests.py`. See
[the directory's README](site/assets/img/fallbacks/README.md).

### Accuracy, stated honestly
The bundle reports **29.7% top-1** (66.8% top-3, MRR 0.520) over 5,756 held-out shots,
measured with real listed player positions. The demo's position picker supplies them, so
that is the applicable figure. Without positions every player falls back to the pipeline
default `role = 3.0` and the same model scores **26.9%** — the one field is worth 2.8
points, which is why the picker exists.

Two caveats worth keeping straight:

- 29.7% was measured on real NBA lineups (37% guards, 11% centres). A user sketching five
  centres is outside that distribution, so treat 29.7% as a ceiling, not a promise.
- Never serve `role` defaulted. The pipeline's rule is "remove and retrain, never default":
  a no-role retrain scores 27.6%, *better* than 26.9%, so defaulting is the worst of the
  three options rather than the cheap middle one.

(The 2017 model's ~86% is not comparable — it was scored against rim-time truth while
serving forecasts — and appears nowhere in this repo.)

## Deploy
See [DEPLOY.md](DEPLOY.md).

## TODO content
Bio + headshot, project #2–4, comedy clips/shows, blog posts — all marked with
`TODO` in the HTML. Project thumbnails are no longer blocking: a project with no
`assets/img/<slug>-thumb.png` shows a default cover until a real one is dropped in. The rebounding case study and blog post have been updated for the
2026 retrain; the blog post still carries a `TODO` about putting it in your own voice.
