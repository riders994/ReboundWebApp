#!/usr/bin/env bash
# Publish the static site to the EC2 box from your local workstation.
#
# Runs the same generators as serve.sh (gallery manifests + project pages), then
# rsyncs site/ straight into the live web root over SSH — no git round-trip. This is
# the "content update" path: use it after running comedy-tags.py / register-tag.py /
# projects.py / dropping new gallery images, to push the result live.
#
#   ./scripts/publish.sh          # generate, then push site/ -> the box
#   ./scripts/publish.sh -n       # dry run: show what rsync WOULD change, transfer nothing
#   ./scripts/publish.sh -v       # verbose (per-file rsync output)
#   ./scripts/publish.sh -c       # also commit the regenerated site/ and push to origin
#
# Connection details come from scripts/publish.env (gitignored — copy the example and
# fill it in) or from the environment. Required: PUBLISH_HOST. See publish.env.example.
#
# The remote /var/www/site is owned by www-data, so rsync runs as root on the box via
# --rsync-path="sudo rsync", then a follow-up `sudo chown` restores www-data ownership.
# Both need passwordless sudo on the remote user (see the note in DEPLOY.md).
#
# This publishes ONLY the static site. The rebound Flask backend + model are deployed
# separately (see DEPLOY.md).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- flags ------------------------------------------------------------------
# NB: rsync flags are kept to what Apple's bundled rsync (openrsync / 2.6.9) also
# accepts, so this runs on a stock macOS workstation. That rules out --info and --chown.
DRY=()          # rsync dry-run
COMMIT=0        # also commit the regenerated site/ and push to origin
RSYNC_VERBOSITY="--stats"
for arg in "$@"; do
  case "$arg" in
    -n|--dry-run) DRY=(--dry-run); ;;
    -v|--verbose) RSYNC_VERBOSITY="--stats --progress -v"; ;;
    -c|--commit) COMMIT=1; ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0; ;;
    *) echo "✗ unknown option: $arg (try -h)" >&2; exit 1; ;;
  esac
done

# --- config -----------------------------------------------------------------
# Load scripts/publish.env if present; explicit env vars still win over the file.
ENV_FILE="$ROOT/scripts/publish.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$ENV_FILE"; set +a
fi

HOST="${PUBLISH_HOST:-}"
USER="${PUBLISH_USER:-ubuntu}"
KEY="${PUBLISH_KEY:-}"                        # optional path to .pem; falls back to ssh-agent/config
REMOTE="${PUBLISH_REMOTE:-/var/www/site}"
OWNER="${PUBLISH_OWNER:-www-data:www-data}"

if [[ -z "$HOST" ]]; then
  echo "✗ PUBLISH_HOST is not set." >&2
  echo "  Copy scripts/publish.env.example to scripts/publish.env and fill it in," >&2
  echo "  or run: PUBLISH_HOST=<elastic-ip> ./scripts/publish.sh" >&2
  exit 1
fi

# --- generate ---------------------------------------------------------------
echo "→ refreshing manifests…"
python3 "$ROOT/scripts/gen-manifests.py"

echo "→ rendering project pages…"
python3 "$ROOT/scripts/projects.py" render || echo "  (project render skipped — see message above)"

# --- optional: commit the regenerated site/ so git history tracks what's live -
# Runs before the push so the commit reflects exactly what gets deployed. Skipped
# entirely on a dry run. A failed push warns but does not block publishing.
if [[ "$COMMIT" -eq 1 && ${#DRY[@]} -eq 0 ]]; then
  if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    if [[ -n "$(git -C "$ROOT" status --porcelain -- site)" ]]; then
      MSG="${PUBLISH_COMMIT_MSG:-Publish site content update ($(date +%Y-%m-%d))}"
      echo "→ committing site/ changes…"
      git -C "$ROOT" add -A -- site
      git -C "$ROOT" commit -q -m "$MSG"
      echo "→ pushing to origin…"
      git -C "$ROOT" push -q || echo "  ⚠ git push failed — commit is local; push it yourself. Publishing anyway."
    else
      echo "→ no site/ changes to commit — skipping commit."
    fi
  else
    echo "  ⚠ --commit given but this isn't a git repo — skipping commit."
  fi
fi

# --- ssh transport ----------------------------------------------------------
SSH_CMD="ssh -o StrictHostKeyChecking=accept-new"
[[ -n "$KEY" ]] && SSH_CMD="$SSH_CMD -i $KEY"

# --- push -------------------------------------------------------------------
[[ ${#DRY[@]} -gt 0 ]] && echo "→ DRY RUN — no files will be transferred"
echo "→ syncing site/ → ${USER}@${HOST}:${REMOTE}/"
rsync -a --delete "${DRY[@]}" $RSYNC_VERBOSITY \
  -e "$SSH_CMD" \
  --rsync-path="sudo rsync" \
  "$ROOT/site/" "${USER}@${HOST}:${REMOTE}/"

if [[ ${#DRY[@]} -gt 0 ]]; then
  echo "✓ dry run complete — re-run without -n to publish."
  exit 0
fi

# Remote rsync ran as root (via sudo), so freshly-written files are root-owned.
# Restore the web-server ownership the deploy expects. (openrsync has no --chown.)
echo "→ fixing ownership to ${OWNER} on the box…"
$SSH_CMD "${USER}@${HOST}" "sudo chown -R ${OWNER} ${REMOTE}"

echo "✓ published to https://${HOST%/} (via ${REMOTE})"
