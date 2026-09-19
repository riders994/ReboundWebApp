#!/usr/bin/env bash
# Refresh the live static site from this clone. Run ON THE BOX, in ~/ReboundWebApp.
#
# The box-side counterpart of publish.sh: after dropping images into a feed, tagging
# clips or editing a project README here, this regenerates the manifests + project pages
# and copies site/ into the web root nginx serves. Editing the clone alone changes
# nothing live - /var/www/site is a separate copy.
#
#   ./scripts/refresh-site.sh        # generate, then sync site/ -> /var/www/site
#   ./scripts/refresh-site.sh -p     # git pull first
#   ./scripts/refresh-site.sh -n     # dry run: show what rsync WOULD change
#
# The sync is `rsync --delete`, so the web root ends up an exact copy of this clone's
# site/ - anything live that is not in the clone is removed. This covers the static site
# only; the backend redeploy is in DEPLOY.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${SITE_ROOT:-/var/www/site}"
OWNER="${SITE_OWNER:-www-data:www-data}"

DRY=()
PULL=0
for arg in "$@"; do
  case "$arg" in
    -n|--dry-run) DRY=(--dry-run --itemize-changes); ;;
    -p|--pull) PULL=1; ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0; ;;
    *) echo "✗ unknown option: $arg (try -h)" >&2; exit 1; ;;
  esac
done

if [[ "$PULL" -eq 1 ]]; then
  echo "→ pulling…"
  git -C "$ROOT" pull --ff-only
fi

echo "→ refreshing manifests…"
python3 "$ROOT/scripts/gen-manifests.py"

echo "→ rendering project pages…"
python3 "$ROOT/scripts/projects.py" render || echo "  (project render skipped — see message above)"

echo "→ syncing site/ → $REMOTE${DRY:+ (dry run)}"
sudo mkdir -p "$REMOTE"
sudo rsync -a --delete "${DRY[@]}" "$ROOT/site/" "$REMOTE/"

if [[ ${#DRY[@]} -eq 0 ]]; then
  sudo chown -R "$OWNER" "$REMOTE"
  echo "✓ live at $REMOTE"
fi
