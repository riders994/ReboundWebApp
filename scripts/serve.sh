#!/usr/bin/env bash
# Startup script for local development.
# Refreshes directory-driven manifests (headshots slideshow, and any other galleries
# registered in gen-manifests.py), then serves the static site. Run before each session
# so newly-dropped media is picked up:
#
#   ./scripts/serve.sh            # serves on :5500
#   PORT=8080 ./scripts/serve.sh  # custom port
#
# The rebound demo backend runs separately (see README.md / rebound-app/).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-5500}"

echo "→ refreshing manifests…"
python3 "$ROOT/scripts/gen-manifests.py"

echo "→ serving $ROOT/site at http://localhost:$PORT (Ctrl-C to stop)"
cd "$ROOT/site"
exec python3 -m http.server "$PORT"
