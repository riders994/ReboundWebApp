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
  assets/            css, js, images, svg  (headshot.jpg, thumbs are TODO)
rebound-app/     Flask JSON API for the demo (Python 3) — no static/template serving
  app.py             HTTP only; all features come from the `rebounding` package
  coordinates.py     canvas<->model frame conversion — read before touching geometry
  movement.py        movement model adapter; optional and fail-soft
  config.py          all runtime config, from the environment
  wsgi.py            gunicorn entry point (`wsgi:app`)
  gunicorn.conf.py   NB: preload_app must stay off — see the docstring
  tests/             40 tests, incl. a regression fixture from the pipeline itself
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
`TODO` in the HTML. The rebounding case study and blog post have been updated for the
2026 retrain; the blog post still carries a `TODO` about putting it in your own voice.
