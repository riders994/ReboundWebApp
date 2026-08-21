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
  assets/            css, js, images, svg  (headshot.jpg, resume.pdf, thumbs are TODO)
rebound-app/     Flask backend for the demo (Python 3)
  app.py             port of the original webapp.py, with a model-free fallback
  requirements.txt
  models/            posnn.h5, msd.pkl (tracked); FinalModel.pkl (retrain & drop in)
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
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python app.py            # serves http://127.0.0.1:8000
curl -s -X POST http://127.0.0.1:8000/predict \
     -H 'Content-Type: application/json' --data @sample.json
```
To exercise the demo against the local backend, set the endpoint on the rebounding page
(`site/projects/rebounding.html`) `window.REBOUND_CONFIG.endpoint` to
`http://localhost:8000/predict`.

Without `FinalModel.pkl` the backend runs in **fallback mode** (placeholder probabilities)
so the page still works. Drop the retrained model at `rebound-app/models/FinalModel.pkl`
for real predictions.

## Deploy
See [DEPLOY.md](DEPLOY.md).

## TODO content
Bio + headshot, resume PDF, project #2–4, comedy clips/shows, blog posts — all marked with
`TODO` in the HTML.
