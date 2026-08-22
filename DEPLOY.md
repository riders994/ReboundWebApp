# Deploying postuptothe.net to EC2

End-to-end runbook: from a bare AWS account to a live HTTPS site at **postuptothe.net**.

The site has two parts:
- **`site/`** — static portfolio, served directly by nginx.
- **`rebound-app/`** — Flask app for the live rebound demo, run under gunicorn and
  reverse-proxied by nginx at `/api/rebound/`.

Assumes Ubuntu 22.04/24.04 on EC2 and a shell user named `ubuntu`.

---

## 0. Provision the EC2 instance

1. **Launch instance** (EC2 → Launch instances):
   - AMI: **Ubuntu Server 24.04 LTS** (or 22.04).
   - Type: **t3.small** or larger. *The rebound backend loads torch — t2/t3.micro's 1 GB
     RAM is not enough; use at least 2 GB.* If you stay on a micro, add swap (below).
   - Key pair: create/download one (e.g. `postup.pem`) so you can SSH in.
   - Storage: 16 GB gp3 is plenty.
2. **Security group** — inbound rules:
   | Type  | Port | Source        | Why            |
   |-------|------|---------------|----------------|
   | SSH   | 22   | *your IP*     | admin access   |
   | HTTP  | 80   | 0.0.0.0/0, ::/0 | web + certbot |
   | HTTPS | 443  | 0.0.0.0/0, ::/0 | web (TLS)     |

   Do **not** open port 8000 — the backend stays bound to localhost.
3. **Elastic IP** — allocate one (EC2 → Elastic IPs) and associate it with the instance so
   the IP survives reboots.
4. **SSH in**:
   ```bash
   chmod 400 postup.pem
   ssh -i postup.pem ubuntu@<ELASTIC_IP>
   ```
5. *(micro instances only)* add 2 GB swap so torch can load:
   ```bash
   sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
   sudo mkswap /swapfile && sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

## 1. Point DNS at the instance

At your domain registrar for **postuptothe.net**, create records pointing at the Elastic IP:

| Type | Name | Value          |
|------|------|----------------|
| A    | @    | `<ELASTIC_IP>` |
| A    | www  | `<ELASTIC_IP>` |

Verify before continuing (propagation can take a few minutes):
```bash
dig +short postuptothe.net        # should print your Elastic IP
```

## 2. Install packages
```bash
sudo apt update
sudo apt install -y nginx python3-venv python3-pip git rsync
```

## 3. Get the code onto the box
```bash
git clone https://github.com/riders994/ReboundWebApp.git ~/ReboundWebApp
cd ~/ReboundWebApp
```

## 4. Static site
```bash
python3 scripts/gen-manifests.py          # refresh gallery manifests before publishing
python3 scripts/projects.py render        # regenerate project detail pages from READMEs
sudo mkdir -p /var/www/site
sudo rsync -a --delete site/ /var/www/site/
sudo chown -R www-data:www-data /var/www/site
```

## 5. Backend (gunicorn + systemd)
```bash
sudo rsync -a rebound-app/ /opt/rebound-app/
cd /opt/rebound-app
python3 -m venv venv
# CPU wheel first, or pip drags in ~2.5 GB of CUDA for an 800k-parameter model.
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install -r requirements.txt
# The feature code. Unpickling the bundles imports this, so it is a hard dependency.
# Use the [serve] extra, not [models]: it is exactly what serve.predict/serve.animate
# need and drops xgboost, which the web host never uses. NB scikit-learn IS required
# even though nothing served here is an sklearn model — BoostedSoftmax wraps LGBMRegressor.
# Pin to the COMMIT the deployed bundle was built at, not a branch — see "The model
# files" below for why a drifted package fails quietly rather than loudly.
./venv/bin/pip install 'rebounding[serve] @ git+https://github.com/riders994/ReboundingPrediction@<commit>'

# Copy both bundles into place (see "The model files" below) BEFORE first start —
# the service refuses to start without the rebounder:
#   /opt/rebound-app/models/FinalModel.pkl
#   /opt/rebound-app/models/MovementModel.pkl

sudo cp ~/ReboundWebApp/deploy/rebound.service /etc/systemd/system/rebound.service
sudo systemctl daemon-reload
sudo systemctl enable --now rebound
systemctl status rebound          # active (running)
curl -s localhost:8000/healthz    # reports WHICH model is live
```
`/healthz` returns the deployed bundle's own provenance — build commit, corpus, fit,
test score — which is the only way to identify weights that never enter git:
```json
{"status":"ok","movement_model":"set-transformer",
 "model":{"features":27,"regime":"served","commit":"d779a8d…","test_top1":0.2969}}
```
> **There is no fallback mode any more.** A missing `FinalModel.pkl` is a refusal to
> start, not placeholder probabilities served silently. A missing `MovementModel.pkl`
> *is* fail-soft: probabilities are unaffected and players simply stay where they were
> placed — `movement_model` reads `none` and `movement_detail` says why.

## 6. nginx
```bash
sudo cp ~/ReboundWebApp/deploy/nginx.conf /etc/nginx/sites-available/postuptothe
sudo ln -sf /etc/nginx/sites-available/postuptothe /etc/nginx/sites-enabled/postuptothe
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```
Visit `http://postuptothe.net` — the site should load over plain HTTP.

> **Why the config caches `/assets/` for 7 days but carves out `*.json`.** Images/CSS/JS
> get a long cache because they're effectively immutable — a changed asset gets a new
> filename. The data/manifest JSON (`assets/data/*.json`, gallery `manifest.json`) is
> different: `scripts/publish.sh` rewrites it **in place** and it can change often, so it's
> given `Cache-Control: no-cache` (revalidate) instead. Without that carve-out, returning
> visitors would keep seeing stale comedy/projects/gallery content for up to a week even
> after a publish. Don't "tidy" the `location ~* \.json$` block back under the `/assets/`
> rule. (nginx still serves Last-Modified/ETag, so an unchanged file 304s — near-zero cost.)

## 7. HTTPS (Let's Encrypt)
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d postuptothe.net -d www.postuptothe.net
```
Certbot edits the nginx config to serve 443 with the cert and adds an HTTP→HTTPS redirect.
Auto-renewal is handled by the certbot timer (`systemctl status certbot.timer`).

Visit `https://postuptothe.net` and exercise the rebound demo end-to-end.

---

## Redeploying after changes
```bash
cd ~/ReboundWebApp && git pull
# static
python3 scripts/gen-manifests.py
python3 scripts/projects.py render
sudo rsync -a --delete site/ /var/www/site/
# backend (preserve the venv and models)
sudo rsync -a rebound-app/ /opt/rebound-app/ --exclude venv --exclude models
sudo systemctl restart rebound
```
This redeploys the *app*, not the model. New weights need the copy-in step below — and if
the retrain moved the feature list, the `rebounding` package must be reinstalled at the
matching commit too. See "Upgrade the package with the weights, not after".

### Content updates from your laptop (`scripts/publish.sh`)
For a **static-content** change (comedy tags/videos, projects, gallery images) you don't
need to SSH in and pull. Run the update tool locally, then push the regenerated `site/`
straight to the box:
```bash
cp scripts/publish.env.example scripts/publish.env   # one time: set PUBLISH_HOST (+ key)
./scripts/publish.sh                                  # regenerate + rsync site/ -> /var/www/site
```
It rsyncs over SSH as root on the box (`--rsync-path="sudo rsync"`) and fixes ownership
back to `www-data`. Grant those two commands passwordless sudo on the box so it doesn't
prompt mid-sync — `sudo visudo`, then:
```
ubuntu ALL=(root) NOPASSWD: /usr/bin/rsync, /usr/bin/chown
```
See `scripts/README.md` for the full options (`-n` dry run, `-v` verbose). The backend is
still deployed with the git-pull block above.

## The model files
Both bundles are gitignored and never in the repo. Build them in the
**ReboundingPrediction** repo (`python -m rebounding.cli train` / `train-movement`) and
copy them to the box:
```bash
scp -i postup.pem FinalModel.pkl MovementModel.pkl ubuntu@<ELASTIC_IP>:/opt/rebound-app/models/
sudo systemctl restart rebound
curl -s localhost:8000/healthz    # confirm the commit you expect is live
```
They are `joblib.dump` of a dataclass bundle, not bare estimators, so the versions in
`rebound-app/requirements.txt` are **pinned, not floored** — unpickling reconstructs
numpy/pandas objects and imports `rebounding`, so a drifting minor version is a runtime
failure rather than a warning. Note `numpy==2.5.2`: the pre-rewrite pin was
`numpy>=1.24,<2.0` and is incompatible.

### Upgrade the package with the weights, not after
A retrain can change the feature list, so **reinstall `rebounding` at the commit the new
bundle was built at** whenever you copy new weights over. Copying bundles alone is only
safe when the feature set did not move, and you cannot tell from the outside.

This matters because the mismatch is **quiet**. `artifact.load` hard-fails on an artifact
schema change, and gives a clear error if the package is missing entirely — but a package
whose *feature definitions* drifted while the schema held only logs a warning:

```
<path> was fitted on a different feature list than this package's 'served' regime;
use artifact.features (27 columns), not the imported list (N)
```

Every feature still has a value, just the wrong one, so predictions stay plausible. After
any upgrade, check for that line before trusting the numbers:
```bash
sudo journalctl -u rebound -e | grep -i "different feature list"   # expect no output
curl -s localhost:8000/healthz    # the commit here should match the package you installed
```

`SourceOfTruth.pkl` from that repo is **not** deployable here — it needs rim-time and
velocity features that do not exist when a visitor is placing dots, and `serve.predict`
refuses it.

## Troubleshooting
- `journalctl -u rebound -e` — backend logs (model load warnings, prediction errors).
- `sudo tail -f /var/log/nginx/error.log` — proxy / static errors.
- 502 on `/api/rebound/predict` → the `rebound` service isn't running or crashed on model load.
- **Worker fails to boot with `ModuleNotFoundError: No module named 'rebounding'`** → the
  feature-code package is not installed in `/opt/rebound-app/venv`. The bundles store the
  model and priors *by class*, so joblib needs the package to reconstruct them; it is a
  hard serving dependency, not a build-time one. Install it as in step 5.
- **Worker fails to boot with `ModuleNotFoundError: No module named 'flask_cors'`** → you set
  `REBOUND_CORS_ORIGINS` without installing the optional dependency. `flask-cors` is commented
  out in `requirements.txt` because same-origin serving does not need it, but setting the env
  var makes the import unconditional, so this is a hard crash rather than a warning. Either
  `pip install flask-cors` or unset the variable. Only needed when the site and API are on
  different origins — in production nginx puts them on the same one.
- **`/healthz` returns 200 but `/predict` hangs until the worker is SIGKILLed** → something
  set `preload_app = True` in `gunicorn.conf.py`. Preloading builds the torch movement
  network in the master and the sync worker forks without exec, inheriting a thread pool
  whose threads did not survive; `/healthz` still answers because it touches no tensors.
  It must stay `False`. Pinned by `rebound-app/tests/test_deploy_config.py`.
- Demo calls `/api/rebound/predict`; nginx strips `/api/rebound/` → gunicorn `/predict`.
  Keep the two in sync if you rename the location.
