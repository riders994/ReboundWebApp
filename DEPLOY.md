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
      without `FinalModel.pkl`, and step 5b copies them up from your laptop — so have them
      locally before you start, not after.
- [ ] **The commit SHA those bundles were built at.** Step 5a pins the `rebounding` package
      to it. `python -m rebounding.cli describe --model FinalModel.pkl` will tell you, and
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
`127.0.0.1`. No private subnet, no load balancer, no RDS — so **no NAT gateway**, which is
the thing worth checking for, since it bills ~$32/mo whether or not traffic flows. The
security group is the real access control.

If you turn on prediction logging (step 5d), Postgres runs **on this same instance**, over
the loopback interface, and none of the above changes: no managed database, no extra subnet,
no NAT gateway, and nothing new opened in the security group. Postgres must not be reachable
from the internet — see 5d.

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

> **Only the site's own records belong on the apex and `www`.** Mail, domain verification and
> anything else that wants a `TXT`/`MX` goes on its own subdomain — see "Mail records
> (Mailgun)". Adding a record to `www` is fine; *replacing* its address record is not.


## Mail records (Mailgun)

Not part of the deploy path — skip this if you are not sending mail. It is documented here
because these records live in the **same hosted zone** as the site's, and the shortcut that
looks obvious breaks the site.

Mailgun needs a **sending domain**. Use a dedicated subdomain — `mg.postuptothe.net` — not the
apex, and emphatically not `www`.

| Type  | Record name | Value |
|-------|-------------|-------|
| TXT   | `mg` | `v=spf1 include:mailgun.org ~all` |
| TXT   | `mx._domainkey.mg` | the DKIM public key, from the Mailgun dashboard |
| MX    | `mg` | `10 mxa.mailgun.org` **and** `10 mxb.mailgun.org` |
| CNAME | `email.mg` | `mailgun.org` |

The DKIM selector and key are generated per domain — copy both out of Mailgun rather than
guessing them. For this zone Mailgun issued the selector `mx`, so the record name is
`mx._domainkey.mg.postuptothe.net`; a re-created sending domain may get a different one.
Both MX values belong in **one** record, one per line in Route 53's value box, not two
separate records — **`mxb` is as required as `mxa`**, and Mailgun leaves the domain
unverified until both are present. The `email.mg` CNAME only powers open/click tracking;
omit it and Mailgun still verifies the domain.

Why a subdomain rather than the apex:

- **Reputation isolation.** The app's mail is scored against `mg.postuptothe.net`. If it ever
  gets spam-flagged, that does not follow mail you send from the bare domain later.
- **No collision with the website.** Mailgun wants MX records. On `mg` they are free; on the
  apex they would fight with anything else that receives mail for the domain.
- **`www` stays a website name**, which is the only thing it should ever be.

> **Do not repurpose `www` for this.** Replacing its address record with an SPF `TXT` takes
> the name off the air *and* does not configure mail: SPF alone authorizes nothing without
> DKIM and MX, and it would be authorizing a sending domain of `www.postuptothe.net`, which
> nothing sends as. The failure is quiet — `dig` returns `NOERROR` with no answer rather than
> `NXDOMAIN`, the apex keeps serving, and the real damage shows up weeks later when certbot
> cannot renew. See "Renewal depends on `www` resolving" under Tearing down.

Verify once the records are in (TTL 300, so a minute or two):
```bash
dig +short TXT mg.postuptothe.net                       # the v=spf1 string
dig +short MX  mg.postuptothe.net                       # BOTH mxa and mxb, priority 10
dig +short TXT mx._domainkey.mg.postuptothe.net         # the DKIM key
dig +short CNAME email.mg.postuptothe.net               # mailgun.org, if tracking is on
```

> **On Mailgun's "sign in with AWS" button.** The integration writes these records for you,
> but it needs broad write access to the entire hosted zone — the same zone holding the
> records that keep this site up — and it can put them on the wrong name. For five records
> you create once, the manual path is the lower-risk one. If you do use it, revoke the
> Route 53 permissions afterwards.

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

Three parts. **5b runs on your laptop, not the box** — that shell switch is the step people
miss, because everything on either side of it is copy-paste over SSH.

### 5a. Code and venv *(on the box)*
```bash
sudo rsync -a rebound-app/ /opt/rebound-app/
# Own it as `ubuntu`, the user the service runs as. Without this the venv cannot be created
# and the scp in 5b lands on a root-owned directory and is refused.
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
```

### 5b. Copy the model bundles in *(from your laptop)*
The bundles are gitignored and never in the repo, so the clone in step 3 did **not** bring
them — `models/` arrives holding only its README. The service refuses to start without
`FinalModel.pkl`, so this happens **before 5c**, not after.

Open a second terminal **on your own machine**, in the directory holding the two bundles you
built during "Before you start":
```bash
# on your laptop — same key you SSH with, same IP from step 0.3
scp -i postup.pem FinalModel.pkl MovementModel.pkl ubuntu@<ELASTIC_IP>:/opt/rebound-app/models/
```
Then back on the box, confirm both landed:
```bash
ls -l /opt/rebound-app/models/    # both .pkl files, owned by ubuntu:ubuntu
```
Failure modes worth naming, because neither says "you skipped a step":
- `Permission denied` — 5a's `chown` was skipped, so the destination is still root-owned.
- `No such file or directory` — 5a never ran, so `/opt/rebound-app/` does not exist yet.

Any transfer that ends with both files in that directory is fine — `rsync -e ssh`, or a
bucket and `aws s3 cp` if the laptop cannot reach the box directly. See "The model files"
for what these bundles are and how to rebuild them.

### 5c. Service, start, smoke test *(on the box)*
```bash
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
A start that fails immediately here, with `journalctl -u rebound -e` showing a missing
model, means 5b did not land — check `ls -l /opt/rebound-app/models/` before anything else.

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

### 5d. Prediction logging *(optional, on the box)*

Off by default: with no `REBOUND_DATABASE_URL` the app never imports a database driver
and behaves exactly as it did before this existed. Turn it on and every **served**
prediction becomes a row — the ten placements the model was given, the probabilities it
returned, its latency, and the bundle's own build commit.

Two things it buys, and both are the reason to bother:

- **The distribution question becomes answerable.** The bundle's 29.7% top-1 was measured
  on real NBA lineups (37% guards, 11% centres). The README calls that a ceiling rather
  than a promise for a visitor who sketches five centres — but nothing measures how far
  real traffic actually sits from the corpus. This table is that measurement, including
  how often `position` arrives null and the pipeline's `role` falls back to its default.
- **Provenance outlives the process.** `/healthz` tells you what a *running* host serves.
  After a retrain that answer is gone, and "which model produced this?" has no answer.
  Every row carries the commit and build time, so it does.

This app is not the reason Postgres is on the box — it shares the server with other
projects — so it gets its own role, database and schema rather than writing into
`public`.

```bash
sudo apt install -y postgresql            # if it is not already there for something else
sudo -u postgres createuser rebound --pwprompt
sudo -u postgres createdb  rebound --owner rebound

# The app creates its schema and table on the first row it writes, so there is no
# migration step and nothing to run by hand — but it can only do that if it is allowed to.
sudo -u postgres psql -d rebound -c 'GRANT CREATE ON DATABASE rebound TO rebound;'
```

Confirm Postgres is listening on **loopback only** before going further. The default on
Ubuntu is `listen_addresses = 'localhost'`, which is what you want; the security group
does not open 5432 either, and neither should you.
```bash
sudo ss -lntp | grep 5432                 # expect 127.0.0.1:5432, not 0.0.0.0:5432
```

The connection string goes in a root-owned file, not in the unit — `systemctl show
rebound` prints `Environment=` lines, password and all, to any user on the box:
```bash
sudo install -d -m 750 /etc/rebound
sudo tee /etc/rebound/telemetry.env >/dev/null <<'ENV'
REBOUND_DATABASE_URL=postgresql://rebound:PUT_THE_PASSWORD_HERE@127.0.0.1:5432/rebound
ENV
sudo chmod 600 /etc/rebound/telemetry.env
sudo systemctl restart rebound
```

`deploy/rebound.service` already reads that path (`EnvironmentFile=-/etc/rebound/…`, where
the `-` means "ignore it if absent"), so a host without the file starts unchanged.

Check it from `/healthz`, which now reports the sink alongside the model:
```bash
curl -s localhost:8000/healthz | python3 -m json.tool   # "telemetry": {"sink": "postgres", …}
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     --data @/opt/rebound-app/sample.json >/dev/null
sudo -u postgres psql -d rebound -c 'SELECT count(*) FROM rebound.predictions;'
```

> **Nothing here identifies a visitor.** No IP, no user agent, no cookie, no session id —
> only the placements somebody deliberately made. Keep it that way: the table is worth
> having precisely because querying it does not require thinking about who is in it.

> **A database problem is never a site problem.** Rows go on a bounded in-memory queue and
> a background thread writes them, so a Postgres that is down, slow, or absent costs
> dropped rows and nothing else — no added latency on `/predict`, no failed responses, no
> failed start. `telemetry.dropped` on `/healthz` is where that shows up, and it is worth
> watching, because by design nothing else will tell you.

Some useful first queries once rows accumulate:
```sql
-- How much traffic is the model seeing without listed positions? (the 26.9% case)
SELECT count(*) FILTER (WHERE p->>'position' IS NULL)::float / count(*) AS null_role_share
FROM rebound.predictions, LATERAL jsonb_array_elements(lineup) AS p;

-- The position mix visitors actually build, against the corpus's 37% G / 11% C.
SELECT p->>'position' AS position, count(*),
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM rebound.predictions, LATERAL jsonb_array_elements(lineup) AS p
GROUP BY 1 ORDER BY 2 DESC;

-- Serving latency by model build, which is how a retrain gets compared to the one before.
SELECT model_commit, count(*), round(avg(latency_ms)::numeric, 2) AS avg_ms
FROM rebound.predictions GROUP BY 1 ORDER BY 2 DESC;
```

### 5e. Backing the database up to the Raspberry Pi *(optional, on the box + the Pi)*

Everything else on this instance is reproducible: the site comes from git, the model
bundles come from `scp`, the config comes from this file. Postgres is the first thing
on the box that exists only here, which turns losing the instance from an afternoon of
rebuilding into losing data. This is the step that takes that back.

Two halves, one on each machine. The box dumps every database to a local spool on a
timer; the Pi reaches in over SSH and pulls the spool down.

> **The Pi pulls — the box never pushes.** The Pi is behind a home NAT, so pushing
> would mean forwarding a port to it, giving your home network a dynamic-DNS name, and
> putting the Pi's SSH on the internet. Pulling needs none of that, because the
> security group already allows 22 from your home IP.
>
> The security argument is the stronger one, though. A push leaves a credential on a
> public cloud box that can write to a machine on your home network — so whoever takes
> the EC2 reaches the Pi, and can delete the backups, which is precisely what someone
> who has taken the box would want to do. Pulling inverts it: the credential lives on
> the trusted side, the box holds nothing that points home, and since `pg-pull.sh`
> never passes `--delete`, nothing that happens on the box can remove a dump the Pi
> already holds. The spool on the box is a staging area; the archive is on the Pi.

**On the box** — install the script and its timer:
```bash
sudo install -m 755 ~/ReboundWebApp/deploy/pg-backup.sh /usr/local/bin/pg-backup.sh
sudo cp ~/ReboundWebApp/deploy/pg-backup.{service,timer} /etc/systemd/system/

# The spool: owned by postgres, group-readable by `backup` so the login user can rsync
# it out, and never world-readable — globals.sql carries role password hashes.
# 2750 is setgid, so every dump the timer writes inherits the group without help.
sudo install -d -o postgres -g backup -m 2750 /var/backups/postgres
sudo usermod -aG backup ubuntu

sudo systemctl daemon-reload
sudo systemctl enable --now pg-backup.timer
sudo systemctl start pg-backup.service      # don't wait until 03:30 to find out
ls -l /var/backups/postgres/latest/
```
You should see `globals.sql`, one `<db>.dump` per database, and `SHA256SUMS`.

> **Why it dumps every database and not just `rebound`.** This cluster is shared with
> your other projects. A backup that only knows about the project that happened to set
> it up is the kind whose gaps are discovered at restore time, so the script reads the
> database list out of the cluster instead of hardcoding one. `globals.sql` matters for
> the same reason: a `pg_dump` contains a database's contents but not the roles that own
> it, and restoring onto a fresh cluster with no `rebound` role fails on every `GRANT`.

**On the Pi** — install the puller:
```bash
# If the Pi already reaches the box for other jobs, skip these two lines and point
# PG_PULL_HOST at the existing ~/.ssh/config Host block, with PG_PULL_USER= empty.
# A key of its own is only worth it to make this job revocable on its own.
ssh-keygen -t ed25519 -f ~/.ssh/postup-backup -C 'pg-pull from the pi'
ssh-copy-id -i ~/.ssh/postup-backup.pub ubuntu@<ELASTIC_IP>

sudo install -d -o pi -g pi /srv/backups
sudo install -o pi -g pi -m 755 pg-pull.sh /srv/backups/pg-pull.sh
cp pg-pull.env.example /srv/backups/pg-pull.env   # then edit: host, key path
sudo cp pg-pull.{service,timer} /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now pg-pull.timer
/srv/backups/pg-pull.sh -n            # dry run: proves the SSH path works
/srv/backups/pg-pull.sh               # for real
```

> **Restrict the key while you are there.** The Pi only ever needs to read one
> directory, so on the box prefix that key's line in `~ubuntu/.ssh/authorized_keys`
> with `command="rrsync -ro /var/backups/postgres",no-pty,no-agent-forwarding,`
> `no-port-forwarding`. That way a compromised Pi gets a read-only view of the dumps
> and not a shell. `rrsync` ships with rsync — `dpkg -L rsync | grep rrsync` to find
> it, and gunzip it out of `/usr/share/doc/rsync/scripts/` if it is not on `PATH`.

#### Restoring

Roles first, then the database — in that order, or the grants have nothing to grant to:
```bash
scp pi@<pi>:/srv/backups/postuptothe/latest/{globals.sql,rebound.dump} .
sudo -u postgres psql -f globals.sql                  # roles + passwords
sudo -u postgres createdb rebound --owner rebound     # only if the database is gone
sudo -u postgres pg_restore -d rebound --clean --if-exists rebound.dump
```

> **Rehearse it once, now, while nothing is wrong.** Restore the newest dump into a
> scratch database (`createdb restore_test`, `pg_restore -d restore_test`, count the
> rows, `dropdb restore_test`) the day you set this up. Until a dump has been read back
> it is a file that is the right size, which is not the same thing as a backup.

#### When it breaks

`pg-pull.sh` exits non-zero on a failed transfer, a checksum mismatch, or a spool that
has stopped being refreshed — so the unit goes to `failed` and `systemctl --failed` on
the Pi is where this reports. Worth an `OnFailure=` notifier if you have one, because
nothing else will ever mention it.

| symptom | cause |
|---|---|
| `Connection timed out` | your home IP changed — update the SSH rule in the security group |
| `Permission denied` reading the spool | `usermod -aG backup ubuntu` needs a fresh login to take effect |
| `newest dump is Nd old` | the transfer is fine; `pg-backup.timer` on the box is not — `systemctl status pg-backup` there |
| `checksum mismatch` on an *old* dump | SD-card rot on the Pi, not a transfer fault. That dump is gone; the others are checked every night for the same reason |

> **An `~/.ssh/config` alias does not fix the timeout**, though it is worth having for
> the key and user. The alias names the *destination*, and the destination is an Elastic
> IP that never changes. What breaks is an inbound rule on the box keyed on the
> *source* — your home address — which nothing on the Pi can influence. Widening the
> rule to `0.0.0.0/0` trades a chore for a permanently internet-exposed SSH port; the
> real fixes are to give the two machines an overlay address (Tailscale or WireGuard,
> after which port 22 can be closed to the world entirely) or to reach the box over
> SSM Session Manager, which needs no inbound rule at all. Neither is worth building
> before the rule has actually gone stale on you once — the freshness check turns this
> into a visible chore rather than lost backups.
>
> Note also that this is not a risk the backup introduces. Every other job the Pi runs
> against the box rides on the same rule, so a changed home IP breaks those too and you
> will hear about it from them first. If you ever do fix it, fix it once for the whole
> SSH path rather than for the backup alone.

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

### 6a. Campaign tracking *(UTM, on the box)*

Tracking which post sent which visitor needs **no analytics vendor and no JavaScript**.
A UTM tag is query string, nginx sees it on every request, and the config above writes
the tagged ones to their own file.

```bash
# The log directory and the retention policy for it.
sudo cp ~/ReboundWebApp/deploy/logrotate-utm.conf /etc/logrotate.d/postuptothe-utm
sudo chmod 0644 /etc/logrotate.d/postuptothe-utm      # logrotate skips a writable config
sudo logrotate -d /etc/logrotate.d/postuptothe-utm    # dry run: prints, changes nothing

# Read the log from your laptop without sudo. /var/log/nginx is root:adm 0640, and
# `adm` is the supported way to grant log reads — narrower than widening the NOPASSWD
# sudoers entry publish.sh uses. Log back in for the group to take effect.
sudo usermod -aG adm ubuntu
```

Then reload nginx (step 6) and check that a tagged request lands:

```bash
curl -sI "http://postuptothe.net/?utm_source=smoketest&utm_medium=cli&utm_campaign=setup" >/dev/null
tail -1 /var/log/nginx/postuptothe.utm.tsv    # one tab-separated line, no IP
```

Reading it, from your laptop — it fetches over SSH using the same host and key as
`scripts/publish.sh`, rotations included:

```bash
python3 scripts/utm.py links                     # the links to paste into a post
python3 scripts/utm.py report                    # hits by source / medium / campaign
python3 scripts/utm.py report --by source,day    # ...per day
python3 scripts/utm.py report --by page,status   # where they landed, and what 404'd
```

**Three things to know before you read a number off it.**

*The crawlers get there first.* LinkedIn and Twitter fetch a shared URL themselves to
build the preview card, using the same tagged link a human would. On a post nobody
clicks, they are the only traffic. `utm.py` excludes them by default and says how many
it dropped; `--include-bots` shows them.

*Retention is the whole reason `logrotate-utm.conf` exists.* The stock
`/etc/logrotate.d/nginx` keeps 14 rotations of `/var/log/nginx/*.log`, which would age
out a launch week a fortnight later. The campaign log is named `.tsv` so it sits outside
that glob (logrotate errors on a log claimed by two configs) and gets `rotate 104`
instead. Same argument as prediction logging in 5d: this traffic cannot be collected
retroactively.

*A tag marks the landing request and nothing after it.* The file answers "which post
sent how many people" and cannot answer "what did they read next" — that needs a
per-visitor id, which the log is deliberately written without. `log_format utm` carries
no `$remote_addr`, matching the property `rebound-app/telemetry.py` maintains for
prediction rows.

#### Adding a campaign link

Short links live in the `$shortlink` map at the top of `deploy/nginx.conf`, so a post
carries `postuptothe.net/go/li-rebound` rather than a URL with four parameters glued on.
Add a line, reinstall the config, reload:

```nginx
/go/li-nextpost  "/blog/posts/next-post.html?utm_source=linkedin&utm_medium=social&utm_campaign=next-post";
```

```bash
sudo cp ~/ReboundWebApp/deploy/nginx.conf /etc/nginx/sites-available/postuptothe
sudo nginx -t && sudo systemctl reload nginx
python3 scripts/utm.py links     # confirms each target actually exists in site/
```

The redirect hop carries no `utm_source`, so it is not logged; the landing request the
browser follows it with is. That is one line per click, not two. Raw
`?utm_source=…` URLs work exactly the same way and need no config change — the short
links are a convenience for what gets pasted, not a requirement.

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
> **The pull on its own changes nothing the service sees.** The clone in `~/ReboundWebApp`
> is a staging copy; the service runs from `/opt/rebound-app` (step 5a) and nginx serves
> `/var/www/site` (step 4). The `rsync` lines are what actually deploy, and the restart is
> what picks the backend up. A pull with no rsync leaves the box running exactly what it
> was running before, with a git tree that says otherwise — which is the confusing version
> of this mistake rather than the loud one.

> **Neither rsync line touches nginx.** `deploy/nginx.conf` is installed by hand (step 6),
> so a change to it — a new `/go/` campaign link, say — needs its own `cp` + `nginx -t` +
> `systemctl reload nginx`. See "Adding a campaign link" in 6a.

Note `--exclude models`: this redeploys the *app*, not the weights, and deliberately will
not clobber the bundles already on the box. New weights need the step 5b copy-in again —
and if the retrain moved the feature list, the `rebounding` package must be reinstalled at
the matching commit too. See "Upgrade the package with the weights, not after".

Two things the block above deliberately does not do, because they are rare and both need
sudo beyond a restart. Check them against `git log` when you pull:

**If `rebound-app/requirements.txt` changed** — the block excludes `venv`, so a new
dependency is on the box but not installed:
```bash
/opt/rebound-app/venv/bin/pip install -r /opt/rebound-app/requirements.txt
sudo systemctl restart rebound
```
Getting this wrong is quiet by design: a missing `psycopg` degrades prediction logging to
`"sink": "none"` on `/healthz` with the reason beside it, and predictions carry on.

**If `deploy/rebound.service` changed** — the unit that systemd runs is the copy under
`/etc/systemd/system`, and nothing above touches it:
```bash
sudo cp ~/ReboundWebApp/deploy/rebound.service /etc/systemd/system/rebound.service
sudo systemctl daemon-reload
sudo systemctl restart rebound
```
Skipping this is quiet too — the service keeps running the previous unit, so a new
`Environment=` or `EnvironmentFile=` line simply does not exist as far as the process is
concerned. `systemctl show rebound -p EnvironmentFiles` says what it is actually using.

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
copy them to the box. On a first deploy this is **step 5b**; the same command replaces the
weights on a running box:
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

### Renewal depends on `www` resolving
The certificate issued in step 7 covers **both** names, and certbot renews it as a single
unit: every SAN is re-authorized over HTTP-01 on each attempt. So `www` is not a one-time
requirement at issuance — it has to keep resolving for the life of the deployment. If it
stops, the renewal for the *whole* certificate aborts and the apex expires with it.

Nothing warns you. The certbot timer runs twice daily and starts renewing at 30 days
remaining, so a `www` record broken in, say, month one fails silently for a month before the
site goes to a browser interstitial. Check the pair whenever you touch the zone:

```bash
dig +short postuptothe.net www.postuptothe.net   # both must resolve to the current Elastic IP
                                                 # (www prints `postuptothe.net.` first — it
                                                 #  is a CNAME; the address below it is what
                                                 #  matters)
sudo certbot renew --dry-run                     # exercises the real authorizations
openssl s_client -connect postuptothe.net:443 -servername postuptothe.net </dev/null 2>/dev/null \
  | openssl x509 -noout -dates -ext subjectAltName
```

`certbot renew --dry-run` is the honest check — it performs the same authorizations against
Let's Encrypt's staging endpoint, so a missing `www` fails there exactly as it would for
real, without burning rate limit. After fixing DNS, force the cert back into a good state
with `sudo certbot renew --force-renewal` rather than waiting for the timer.

The two ways `www` goes missing are a rebuild that repoints the apex and forgets the second
record, and a zone edit that overwrites `www` with something that is not an address — see the
callout in "Mail records (Mailgun)".

## Troubleshooting
- `journalctl -u rebound -e` — backend logs (model load warnings, prediction errors).
- `sudo tail -f /var/log/nginx/error.log` — proxy / static errors.
- 502 on `/api/rebound/predict` → the `rebound` service isn't running or crashed on model load.
- **Service refuses to start, log says the rebounder is missing** → `models/FinalModel.pkl`
  is not on the box. The clone does not carry it; it goes up separately in step 5b. Check
  with `ls -l /opt/rebound-app/models/`, and note an `scp` onto a root-owned
  `/opt/rebound-app` fails with `Permission denied` — rerun 5a's `chown` and copy again.
- **Worker fails to boot with `ModuleNotFoundError: No module named 'rebounding'`** → the
  feature-code package is not installed in `/opt/rebound-app/venv`. The bundles store the
  model and priors *by class*, so joblib needs the package to reconstruct them; it is a
  hard serving dependency, not a build-time one. Install it as in step 5a.
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
- **`/healthz` shows `"telemetry": {"sink": "none", …}`** → the `detail` beside it says
  which of the four reasons it is: no `REBOUND_DATABASE_URL` (the default — the file in 5d
  is missing or was not picked up by a restart), `REBOUND_TELEMETRY=0`, `psycopg not
  installed`, or a schema name that is not a valid identifier. None of these affect
  predictions.
- **`sink` is `postgres` but `dropped` climbs and `recorded` does not** → the writer thread
  cannot reach the database or cannot insert; the reason is in `detail` on the same
  endpoint and in `journalctl -u rebound`. Usually the role lacks `CREATE` on the database
  (the table is created on first write) or the password in `/etc/rebound/telemetry.env` is
  wrong. The site is unaffected while you fix it, which is why this needs looking at
  rather than waiting to be reported.
- **`recorded` looks low** → it is per worker, and `WEB_CONCURRENCY=2` means two of them.
  `/healthz` answers from whichever worker took the request, so the counts alternate. The
  row count in Postgres is the real total.
- Demo calls `/api/rebound/predict`; nginx strips `/api/rebound/` → gunicorn `/predict`.
  Keep the two in sync if you rename the location.
