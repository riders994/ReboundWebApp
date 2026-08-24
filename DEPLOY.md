# Deploying postuptothe.net to EC2

End-to-end runbook: from a bare AWS account to a live HTTPS site at **postuptothe.net**.

The site has two parts:
- **`site/`** — static portfolio, served directly by nginx.
- **`rebound-app/`** — Flask app for the live rebound demo, run under gunicorn and
  reverse-proxied by nginx at `/api/rebound/`.

Assumes **Ubuntu 24.04** on EC2 and a shell user named `ubuntu`.

> **24.04, not 22.04.** The pinned stack needs **Python ≥ 3.12** — `numpy==2.5.2` and
> `scipy==1.18.0` both declare `Requires-Python >=3.12`, and the model bundles were built
> on 3.12.3. Ubuntu 22.04 ships Python 3.10, so `pip install -r requirements.txt` cannot
> resolve there. If you are stuck on 22.04 you would need a newer interpreter from
> deadsnakes and a venv built against it; picking 24.04 avoids the whole problem.

## Before you start

Five things, two of which have to be prepared off this box:

- [ ] **AWS account** and an EC2 key pair you can SSH with.
- [ ] **The domain's DNS in Route 53** — it already is; nameservers are `ns-*.awsdns-*`.
- [ ] **Both model bundles in hand**, built in the ReboundingPrediction repo:
      `python -m rebounding.cli train` and `train-movement`. The service **refuses to start**
      without `FinalModel.pkl`, so get these before step 5, not after.
- [ ] **The commit SHA those bundles were built at.** Step 5 pins the `rebounding` package to
      it. `python -m rebounding.cli describe --model FinalModel.pkl` will tell you, and
      `/healthz` reports it once deployed.
- [ ] **The retrain work merged into `primary`** — or be ready to check out the branch on the
      box. See step 3.

Rough timings: apt and pip are a few minutes each (torch is the slow one, ~200 MB), DNS
propagation is minutes at TTL 300, and certbot is seconds once DNS resolves.

---

## 0. Provision the EC2 instance

1. **Launch instance** (EC2 → Launch instances):
   - AMI: **Ubuntu Server 24.04 LTS** — it ships Python 3.12, which the pinned stack
     requires. Do not use 22.04 (Python 3.10); see the note above.
   - Type: **t3.small** (2 GB) or larger. Measured footprint of the backend with two
     workers: **~690 MB** total, each worker ~430 MB. Workers do *not* share model memory —
     `preload_app` is off (it deadlocks torch across a fork; see Troubleshooting), so every
     worker loads its own copy. t2/t3.micro's 1 GB will OOM; if you stay on one, add swap
     (below) and set `WEB_CONCURRENCY=1`.
   - **Note the worker count scales with vCPU.** `gunicorn.conf.py` defaults to
     `min(4, cpu_count)`, so a 4-vCPU box spawns 4 workers ≈ 1.4 GB. Pin it in the unit
     file (`Environment="WEB_CONCURRENCY=2"`) if you size up, or the memory math changes
     under you.
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

### Networking / VPC — nothing to build
Every instance lives in a VPC, but you do **not** need to create one. Each region has a
**default VPC** already wired with a public subnet per AZ, an internet gateway, and a route
sending `0.0.0.0/0` to it. Launch without touching the networking panel and you land there,
which is what this runbook assumes.

That is the right shape here: one public instance, nginx terminating TLS, gunicorn bound to
`127.0.0.1`. No private subnet, no load balancer, no database — so **no NAT gateway**, which
is the thing worth checking for, since it bills ~$32/mo whether or not traffic flows. The
security group is the real access control.

Most VPC objects are free. These are not:

| | cost | needed here |
|---|---|---|
| NAT gateway | ~$32/mo + data processing | **no** |
| Interface VPC endpoint | ~$7/mo each | no |
| Public IPv4 address | ~$3.60/mo **each, attached or not** (since Feb 2024) | one, unavoidable |
| Subnets, IGW, route tables, security groups | free | — |

## 1. Point DNS at the instance

**DNS for this domain lives in Route 53**, not at the registrar — the zone's nameservers are
`ns-*.awsdns-*`. Check with `dig +short NS postuptothe.net` if that ever changes. Edits go in
**Route 53 → Hosted zones → postuptothe.net → Create record**.

An **A record** maps a hostname to an IPv4 address. The catch that bites people: DNS treats
every hostname as a *separate name*, so `postuptothe.net` and `www.postuptothe.net` are two
different things and having one says nothing about the other. The bare domain (the "apex" or
"root") is written `@`; the other is written `www`. Both point at the same server — you are
not running two sites, you are telling DNS that two names lead to one place.

| Type | Record name | Value          | Resolves |
|------|-------------|----------------|----------|
| A    | *(leave blank)* | `<ELASTIC_IP>` | `postuptothe.net` |
| A    | `www`       | `<ELASTIC_IP>` | `www.postuptothe.net` |

> **Route 53 does not use `@`.** Most registrar UIs write the apex as `@`, but Route 53's
> "Record name" field is a *prefix* — it appends `.postuptothe.net` for you, shown greyed
> out beside the box. Leave it **empty** for the apex. Typing `@` creates a record for
> the literal name `@.postuptothe.net`, which resolves nothing and looks fine in the list.

Set **TTL 300** while you are changing things, so a mistake corrects in five minutes instead
of an hour. Raise it once the site is stable.

### Doing it, click by click
You need the Elastic IP from step 0.3 first — allocate and associate it before touching DNS,
so you only edit the zone once.

1. AWS console → **Route 53** → **Hosted zones** → **postuptothe.net**.
2. **Update the apex.** There is already an `A` record named `postuptothe.net` pointing at the
   old address. Tick it → **Edit record** → replace **Value** with the new Elastic IP → set
   **TTL** to `300` → **Save**.
3. **Create the `www` record.** **Create record** → **Record name**: type `www` → **Record
   type**: `A` → **Value**: the same Elastic IP → **TTL**: `300` → **Create records**.
4. Wait a minute, then verify with the `dig` commands below. Both must print the new IP.

Ignore the `NS` and `SOA` records already in the zone — they are the zone's own plumbing and
must not be edited.

If you prefer the CLI (`aws` is not installed on this box by default):
```bash
ZONE=$(aws route53 list-hosted-zones-by-name --dns-name postuptothe.net \
       --query 'HostedZones[0].Id' --output text)
aws route53 change-resource-record-sets --hosted-zone-id "$ZONE" --change-batch '{
  "Changes": [
    {"Action":"UPSERT","ResourceRecordSet":{
      "Name":"postuptothe.net","Type":"A","TTL":300,
      "ResourceRecords":[{"Value":"<ELASTIC_IP>"}]}},
    {"Action":"UPSERT","ResourceRecordSet":{
      "Name":"www.postuptothe.net","Type":"A","TTL":300,
      "ResourceRecords":[{"Value":"<ELASTIC_IP>"}]}}
  ]}'
```
`UPSERT` creates the record if absent and overwrites it if present, so the same command
handles both the apex update and the new `www`.

> **Create BOTH records, and before step 7.** Certbot proves you control each name by
> fetching a file over HTTP from it, and step 7 requests one certificate covering both
> (`-d postuptothe.net -d www.postuptothe.net`). If `www` does not resolve, that
> authorization fails and the **entire run aborts** — you get no certificate at all, not
> even for the apex. It reads like a certbot problem when it is a DNS problem.
> `deploy/nginx.conf:8` lists both in `server_name` too. If you genuinely do not want the
> subdomain, drop it from the certbot command *and* from `server_name`.

**Rebuilding after a teardown?** The apex record survives an instance being destroyed, so it
will still be pointing at whatever address you released. Update it — do not assume a missing
site means a missing record. See "Tearing down (or rebuilding from scratch)".

Verify before continuing (propagation can take a few minutes):
```bash
dig +short postuptothe.net        # must print the NEW Elastic IP
dig +short www.postuptothe.net    # must print it too, or certbot will fail
```
`NXDOMAIN` from the second means the name does not exist at all — a missing record, not a
misconfigured one.

> **Optional, and worth it:** make `www` a **CNAME** to `postuptothe.net` instead of a second
> A record. Then the IP lives in exactly one place and the next IP change is a single edit.
> The apex must stay an A record either way — standard DNS does not allow a CNAME there.

| Type  | Record name | Value              |
|-------|-------------|--------------------|
| A     | *(blank)*   | `<ELASTIC_IP>`     |
| CNAME | `www`       | `postuptothe.net`  |

## 2. Install packages
```bash
sudo apt update
sudo apt install -y nginx python3-venv python3-pip git rsync python3-markdown
python3 --version                 # must be 3.12 or newer — see the note at the top
```
`python3-markdown` is needed by `scripts/projects.py render` in step 4 — it renders project
READMEs into detail pages, and without it that step dies with `ModuleNotFoundError`.

> **Install it via apt, not pip.** On 24.04 the system Python is *externally managed*
> (PEP 668), so `pip install markdown` outside a venv fails with
> `error: externally-managed-environment`. The apt package sidesteps that. If you would
> rather use pip, build a venv for the scripts and run them from it.

`scripts/requirements.txt` also lists `boto3`, but it is imported lazily and only for
S3-backed photo galleries. Those are disabled here, so you do not need it.

## 3. Get the code onto the box
```bash
git clone https://github.com/riders994/ReboundWebApp.git ~/ReboundWebApp
cd ~/ReboundWebApp
```

> **Check what you just cloned.** `git clone` gives you the default branch (`primary`). The
> retrained-model backend, the position picker and this runbook all have to be *on* that
> branch or you will deploy the 2017 app — which tries to load `posnn.h5`/`msd.pkl`, files
> that no longer exist, and fails. Either merge the work into `primary` first, or clone the
> branch explicitly:
> ```bash
> git -C ~/ReboundWebApp log --oneline -1     # expect the retrain work, not the old app
> # if not:
> git -C ~/ReboundWebApp checkout rebound-retrain-and-position-picker
> ```

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
# Own it as `ubuntu`, the user the service runs as. Without this the venv cannot be created
# and the scp of the model bundles lands on a root-owned directory and is refused.
sudo chown -R ubuntu:ubuntu /opt/rebound-app
cd /opt/rebound-app
mkdir -p models                            # tracked via models/README.md, but harmless
python3 -m venv venv
# ORDER MATTERS. requirements.txt pins torch==2.13.0+cpu, and that "+cpu" build does not
# exist on PyPI — installing it first from the CPU index satisfies the pin. Reverse these
# two lines and the requirements install fails to resolve torch. Installing torch from
# PyPI instead pulls ~2.5 GB of CUDA onto a host that will never have a GPU.
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

# Smoke-test a real prediction before wiring up nginx, so a failure here is unambiguous.
# Expect HTTP 200 and ten probabilities summing to 1.
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     --data @/opt/rebound-app/sample.json | head -c 200
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

## Tearing down (or rebuilding from scratch)
Releasing an Elastic IP returns it to the regional pool, where AWS can reassign it to
another account — but **your DNS keeps pointing at it**. Until you repoint the A records,
traffic for the domain goes to whoever holds that address next. HTTPS visitors get a
certificate mismatch rather than a silent redirect, since the new holder cannot obtain a
cert for your name without controlling DNS, but a dangling A record is still worth closing.

Order that avoids a gap:

1. Stand up the new instance and allocate + associate its Elastic IP.
2. Update **both** A records to the new IP; confirm with `dig +short`.
3. Only then release the old Elastic IP and terminate the old instance.

Doing it the other way round — as in a teardown-first rebuild — leaves the domain pointing
at an address you no longer control for as long as the rebuild takes.

Terminating an instance releases its *automatic* public IP but **not** an associated Elastic
IP: that must be released explicitly or it keeps billing (~$3.60/mo) while unattached.

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
