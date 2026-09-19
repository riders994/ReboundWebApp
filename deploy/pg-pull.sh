#!/usr/bin/env bash
# Pull the EC2 box's Postgres dumps down to the Raspberry Pi, verify them, and prune
# the local archive. Runs ON THE PI, on a systemd timer (deploy/pg-pull.timer).
#
#   ./pg-pull.sh           # pull, verify, prune
#   ./pg-pull.sh -n        # dry run: show what rsync WOULD transfer, write nothing
#   ./pg-pull.sh -v        # per-file rsync output
#   ./pg-pull.sh -h        # this header
#
# Connection details come from pg-pull.env beside this script (gitignored — copy
# pg-pull.env.example and fill it in) or from the environment. See DEPLOY.md step 5e.
#
# ## Why the Pi pulls instead of the box pushing
#
# The Pi sits behind a home NAT with no inbound path, so a push would mean forwarding
# a port to it, giving the home network a dynamic-DNS name, and exposing the Pi's SSH
# to the internet — three new pieces of public attack surface to move a few megabytes
# a night. A pull needs none of them: the Pi opens the connection outbound, and the
# EC2 security group already allows 22 from your home IP.
#
# The security argument is the stronger one. A push keeps a credential on a public
# cloud box that can write to a machine on your home network, so whoever takes the
# EC2 also reaches the Pi *and* can delete the backups — including, in the case that
# makes backups matter, deliberately. Pulling inverts it: the credential lives on the
# trusted side, the EC2 holds nothing that points home, and because this script never
# passes --delete, nothing that happens on the box can remove a dump the Pi already
# has. The box is a spool the Pi copies from; the archive only grows here.
#
# The one thing that breaks: your home IP changes and the security group's SSH rule
# stops matching. That shows up as a connection timeout here, not as silent data loss.
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)"

DRY=()
VERBOSE=()
for arg in "$@"; do
  case "$arg" in
    -n|--dry-run) DRY=(--dry-run); ;;
    -v|--verbose) VERBOSE=(-v --progress); ;;
    -h|--help) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0; ;;
    *) echo "✗ unknown option: $arg (try -h)" >&2; exit 1; ;;
  esac
done

# shellcheck source=/dev/null
[ -f "$SELF/pg-pull.env" ] && . "$SELF/pg-pull.env"

PG_PULL_HOST="${PG_PULL_HOST:-}"
# No colon in the default, so an explicitly empty PG_PULL_USER= is honoured rather
# than falling back: that is how you hand the whole connection to an ~/.ssh/config
# Host block and let User, IdentityFile and the rest come from there.
PG_PULL_USER="${PG_PULL_USER-ubuntu}"
PG_PULL_KEY="${PG_PULL_KEY:-}"
PG_PULL_REMOTE="${PG_PULL_REMOTE:-/var/backups/postgres}"
PG_PULL_DEST="${PG_PULL_DEST:-/srv/backups/postuptothe}"
# The Pi is the archive, so it keeps far more history than the box's 7 days. At a few
# hundred bytes per prediction row this is megabytes a year; the number is generous
# because there is no reason for it not to be.
PG_PULL_KEEP_DAYS="${PG_PULL_KEEP_DAYS:-180}"
# Fail if the newest dump on hand is older than this. Catches the failure this whole
# arrangement is otherwise blind to: the dump timer dying on the box while rsync keeps
# succeeding perfectly against a spool that stopped changing weeks ago.
PG_PULL_STALE_DAYS="${PG_PULL_STALE_DAYS:-2}"

[ -n "$PG_PULL_HOST" ] || {
  echo "✗ PG_PULL_HOST is not set — copy pg-pull.env.example to pg-pull.env" >&2; exit 1; }

log() { echo "[pg-pull] $*"; }

# --- ssh transport ----------------------------------------------------------
# BatchMode because this runs unattended: a host-key or passphrase prompt must fail
# the unit, not sit forever waiting for a keyboard that is not there.
SSH_CMD="ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
[ -n "$PG_PULL_KEY" ] && SSH_CMD="$SSH_CMD -i $PG_PULL_KEY"

# Prefixing user@ unconditionally would override a Host block's User, so an empty
# PG_PULL_USER addresses the host bare and ssh resolves everything itself.
TARGET="$PG_PULL_HOST"
[ -n "$PG_PULL_USER" ] && TARGET="$PG_PULL_USER@$PG_PULL_HOST"

mkdir -p "$PG_PULL_DEST"

log "$TARGET:$PG_PULL_REMOTE/ -> $PG_PULL_DEST/"

# No --delete, deliberately: see the header. No -z either — .dump files are already
# compressed, so it would spend Pi CPU to save nothing.
rsync -a --human-readable --stats \
      "${DRY[@]}" "${VERBOSE[@]}" \
      --exclude='.partial-*' \
      -e "$SSH_CMD" \
      "$TARGET:$PG_PULL_REMOTE/" "$PG_PULL_DEST/"

if [ "${#DRY[@]}" -gt 0 ]; then
  log "dry run — nothing verified, nothing pruned"
  exit 0
fi

# --- verify -----------------------------------------------------------------
# Every dated directory, not just tonight's. Re-reading the whole archive costs a few
# seconds at this size and it is the only thing that would ever notice an old dump
# quietly rotting on the SD card — which is the failure mode this hardware actually
# has, and which a restore would otherwise discover at the worst possible moment.
FAILED=0
CHECKED=0
for dir in "$PG_PULL_DEST"/20??-??-??; do
  [ -d "$dir" ] || continue
  CHECKED=$((CHECKED + 1))
  if [ ! -f "$dir/SHA256SUMS" ]; then
    echo "✗ $(basename "$dir"): no SHA256SUMS — incomplete transfer?" >&2
    FAILED=$((FAILED + 1))
    continue
  fi
  if ! ( cd "$dir" && sha256sum --quiet -c SHA256SUMS ); then
    echo "✗ $(basename "$dir"): checksum mismatch" >&2
    FAILED=$((FAILED + 1))
  fi
done

[ "$CHECKED" -gt 0 ] || { echo "✗ no dumps found in $PG_PULL_DEST" >&2; exit 1; }

# --- prune ------------------------------------------------------------------
# Only ever local, and only directories this script's counterpart created.
find "$PG_PULL_DEST" -mindepth 1 -maxdepth 1 -type d -name '20??-??-??' \
     -mtime "+$PG_PULL_KEEP_DAYS" -exec rm -rf {} +

# --- freshness --------------------------------------------------------------
NEWEST="$(find "$PG_PULL_DEST" -mindepth 1 -maxdepth 1 -type d -name '20??-??-??' \
          -printf '%f\n' | sort | tail -1)"
# Only reachable if PG_PULL_KEEP_DAYS was set low enough to prune everything just
# pulled, which is a misconfiguration rather than a backup problem — say so.
[ -n "$NEWEST" ] || { echo "✗ pruning left nothing — is PG_PULL_KEEP_DAYS ($PG_PULL_KEEP_DAYS) too low?" >&2; exit 1; }
AGE_DAYS=$(( ( $(date -u +%s) - $(date -u -d "$NEWEST" +%s) ) / 86400 ))

log "$CHECKED dump(s) on hand, newest $NEWEST (${AGE_DAYS}d old), $(du -sh "$PG_PULL_DEST" | cut -f1) total"

if [ "$FAILED" -gt 0 ]; then
  echo "✗ $FAILED dump(s) failed verification — see above" >&2
  exit 1
fi
if [ "$AGE_DAYS" -gt "$PG_PULL_STALE_DAYS" ]; then
  echo "✗ newest dump is ${AGE_DAYS}d old (limit ${PG_PULL_STALE_DAYS}d) — is pg-backup.timer still running on the box?" >&2
  exit 1
fi

date -u +%FT%TZ > "$PG_PULL_DEST/LAST_SUCCESS"
log "ok"
