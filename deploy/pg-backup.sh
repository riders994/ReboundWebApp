#!/usr/bin/env bash
# Dump every database on the box's Postgres cluster to a local spool directory.
# Runs ON THE EC2 INSTANCE, as the `postgres` user, on a systemd timer
# (deploy/pg-backup.timer). It does NOT transfer anything: the Raspberry Pi reaches
# in and pulls the spool down itself (deploy/pg-pull.sh), so nothing here needs a
# credential for, or a route to, the home network. See DEPLOY.md step 5e.
#
#   sudo -u postgres /usr/local/bin/pg-backup.sh          # run it by hand
#   sudo systemctl start pg-backup.service                # or via the unit
#
# Layout it produces under $SPOOL (default /var/backups/postgres):
#
#   2026-09-19/globals.sql      roles + passwords (pg_dumpall --globals-only)
#   2026-09-19/rebound.dump     one pg_dump -Fc per database
#   2026-09-19/SHA256SUMS       checksums, verified by the Pi after it pulls
#   latest -> 2026-09-19
#
# Two properties the Pi depends on:
#
# **A dated directory appears only when it is complete.** The dump is built in a
# .partial-<pid> directory and renamed into place at the end, so a pull that fires
# mid-dump sees yesterday's tree and not a half-written file. rsync would happily
# copy a truncated .dump and the checksum would not catch it, because SHA256SUMS
# would have been written against the truncated file too.
#
# **Globals are dumped, not just databases.** A pg_dump carries a database's contents
# but not the roles that own them: restore `rebound.dump` onto a fresh cluster with no
# `rebound` role and it fails partway through on every GRANT. globals.sql is the small
# file that makes the big ones restorable.
set -euo pipefail

# Group-readable so the login user can rsync the spool out; world-readable never,
# because globals.sql contains role password hashes.
umask 027

SPOOL="${PG_BACKUP_SPOOL:-/var/backups/postgres}"
# Days of dumps to keep ON THE BOX. This is a staging area, not the archive — the Pi
# keeps the long history. Enough days that a Pi that has been off over a long weekend
# still finds everything it missed.
KEEP_DAYS="${PG_BACKUP_KEEP_DAYS:-7}"

STAMP="$(date -u +%F)"
DEST="$SPOOL/$STAMP"
WORK="$SPOOL/.partial-$$"

log() { echo "[pg-backup] $*"; }

command -v pg_dump >/dev/null || { echo "✗ pg_dump not found" >&2; exit 1; }

mkdir -p "$SPOOL"
# A previous run killed mid-dump leaves a .partial behind. Clear our own on any exit.
trap 'rm -rf "$WORK"' EXIT
rm -rf "$WORK"
mkdir -p "$WORK"

log "dumping cluster -> $DEST"

# Roles, role passwords and tablespaces. Cluster-wide, so it is not in any pg_dump.
pg_dumpall --globals-only > "$WORK/globals.sql"

# Every database that is not a template and allows connections. Reading the list from
# the cluster rather than hardcoding `rebound` is the point: this box's Postgres is
# shared with other projects, and a backup that only knows about the project that set
# it up is the kind that gets discovered to be incomplete at restore time.
# The grep is not decoration: it drops blank lines, so the count below is a real
# count. Without it an empty or failed psql yields one empty-string "database", and
# the guard waves through a backup containing nothing.
mapfile -t DBS < <(psql -Atqc \
  "SELECT datname FROM pg_database WHERE NOT datistemplate AND datallowconn ORDER BY datname" \
  | grep -v '^[[:space:]]*$')

# Checked before anything is dumped, so a cluster that is down or empty leaves the
# last good dump in place rather than replacing it with an empty one. A backup system
# that overwrites yesterday's working dump when today's fails is worse than none.
if [ "${#DBS[@]}" -eq 0 ]; then
  echo "✗ no databases found (is the cluster running?) — refusing to publish an empty backup" >&2
  exit 1
fi

for db in "${DBS[@]}"; do
  log "  $db"
  # -Fc: compressed custom format. Restores with pg_restore, and supports pulling a
  # single table out of the archive without replaying the whole thing.
  pg_dump -Fc --file="$WORK/$db.dump" "$db"
done

( cd "$WORK" && sha256sum ./* > SHA256SUMS )

# Publish atomically. An existing directory for today (a second run) is replaced.
rm -rf "$DEST"
mv "$WORK" "$DEST"
ln -sfn "$STAMP" "$SPOOL/latest"

# Prune old dumps. -maxdepth 1 so this never descends into a dated directory, and the
# name pattern so it can only ever match something this script created.
find "$SPOOL" -mindepth 1 -maxdepth 1 -type d -name '20*-*-*' -mtime "+$KEEP_DAYS" \
     -exec rm -rf {} +

log "done: $(du -sh "$DEST" | cut -f1) in $DEST (${#DBS[@]} database(s))"
