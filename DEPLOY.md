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
   - Type: **t3.small** or larger. *The rebound backend loads TensorFlow — t2/t3.micro's
     1 GB RAM is not enough; use at least 2 GB.* If you stay on a micro, add swap (below).
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
5. *(micro instances only)* add 2 GB swap so TensorFlow can load:
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
sudo mkdir -p /var/www/site
sudo rsync -a --delete site/ /var/www/site/
sudo chown -R www-data:www-data /var/www/site
```

## 5. Backend (gunicorn + systemd)
```bash
sudo rsync -a rebound-app/ /opt/rebound-app/
cd /opt/rebound-app
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Drop your retrained model in place (see note below):
#   /opt/rebound-app/models/FinalModel.pkl

sudo cp ~/ReboundWebApp/deploy/rebound.service /etc/systemd/system/rebound.service
sudo systemctl daemon-reload
sudo systemctl enable --now rebound
systemctl status rebound          # active (running)
curl -s localhost:8000/health     # {"status":"ok","models_ready":true|false}
```
> If `models_ready` is `false`, the app still runs in **fallback mode** (placeholder
> probabilities) so the page works — upload `FinalModel.pkl` and `sudo systemctl restart
> rebound` to switch to real predictions.

## 6. nginx
```bash
sudo cp ~/ReboundWebApp/deploy/nginx.conf /etc/nginx/sites-available/postuptothe
sudo ln -sf /etc/nginx/sites-available/postuptothe /etc/nginx/sites-enabled/postuptothe
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```
Visit `http://postuptothe.net` — the site should load over plain HTTP.

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
sudo rsync -a --delete site/ /var/www/site/
# backend (preserve the venv and models)
sudo rsync -a rebound-app/ /opt/rebound-app/ --exclude venv --exclude models
sudo systemctl restart rebound
```

## The model file
`FinalModel.pkl` (the RandomForest) is gitignored and not in the repo — copy it to the box
manually, e.g. from your laptop:
```bash
scp -i postup.pem FinalModel.pkl ubuntu@<ELASTIC_IP>:/opt/rebound-app/models/
sudo systemctl restart rebound
```
It must be pickled with the **same scikit-learn version** listed in
`rebound-app/requirements.txt`, or it won't unpickle. Pin that version once you retrain.

## Troubleshooting
- `journalctl -u rebound -e` — backend logs (model load warnings, prediction errors).
- `sudo tail -f /var/log/nginx/error.log` — proxy / static errors.
- 502 on `/api/rebound/predict` → the `rebound` service isn't running or crashed on model load.
- Demo calls `/api/rebound/predict`; nginx strips `/api/rebound/` → gunicorn `/predict`.
  Keep the two in sync if you rename the location.
