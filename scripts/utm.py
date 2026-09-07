#!/usr/bin/env python3
"""Read the campaign log nginx writes, and list the tracking links you can share.

There is no analytics vendor behind any of this. A UTM tag is just query string, nginx
already sees every request, and `deploy/nginx.conf` writes the tagged ones to their own
tab-separated file. This script is the reader for that file — it fetches it off the box
over SSH (the same host/key `publish.sh` uses) and groups it.

  utm.py report                        # hits by source / medium / campaign
  utm.py report --by source,day        # ...broken out per day
  utm.py report --by page              # which posts the tagged traffic landed on
  utm.py report --since 2026-09-01     # a launch window
  utm.py report --include-bots         # keep the preview crawlers in
  utm.py raw --source linkedin         # the matching log lines, unaggregated
  utm.py links                         # the /go/... links to paste into a post
  utm.py report --file ./utm.tsv       # a local copy instead of fetching over SSH

**Preview crawlers are excluded by default and this matters more than it sounds.**
LinkedIn and Twitter fetch a shared URL themselves to build the preview card, using the
same tagged link a human would. On a post that nobody clicks, the crawlers are the only
traffic, and counted naively they read as an audience. `--include-bots` shows them; the
default answer is about people.

**A UTM tag only ever marks the landing request.** The parameters are on the link that
was clicked and on nothing after it, so this file answers "which post sent how many
people" and cannot answer "what did they read next". That would need a per-visitor id,
which is exactly what the log is written without — see the log_format comment in
deploy/nginx.conf.
"""
import argparse
import collections
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NGINX_CONF = ROOT / "deploy" / "nginx.conf"
SITE = ROOT / "site"

# Must match the `log_format utm` field order in deploy/nginx.conf. Changing one without
# the other is the single most likely way to break this script, so they name each other.
FIELDS = ["time", "status", "page", "source", "medium", "campaign", "referer", "agent"]

DEFAULT_LOG = "/var/log/nginx/postuptothe.utm.tsv"

# Preview crawlers and command-line clients. Anything matching is dropped unless
# --include-bots. Deliberately broad: a human miscounted as a bot understates a number,
# a crawler counted as a human invents an audience, and only one of those is a mistake
# you would act on.
BOT_RE = re.compile(
    r"bot|crawler|spider|slurp|preview|fetcher|scraper|monitor|"
    r"facebookexternalhit|linkedinbot|twitterbot|discordbot|slackbot|telegrambot|"
    r"whatsapp|skypeuripreview|embedly|quora link preview|redditbot|applebot|"
    r"curl|wget|python-requests|python-urllib|go-http-client|libwww-perl|okhttp|"
    r"headlesschrome|phantomjs|lighthouse|pingdom|uptimerobot",
    re.I,
)

GROUPABLE = ["source", "medium", "campaign", "page", "day", "referer", "status"]

# nginx's escape=default writes \xXX for control bytes (a literal tab inside a value
# becomes \x09, which is what keeps splitting on tab safe) and \" for a quote.
ESCAPE_RE = re.compile(r"\\x([0-9A-Fa-f]{2})")


def unescape(value):
    """Undo nginx's log escaping, and render its empty-value '-' as an empty string."""
    if value == "-":
        return ""
    return ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), value).replace('\\"', '"')


# --------------------------------------------------------------------------- sourcing

def load_env():
    """Host/user/key from scripts/publish.env, so this shares publish.sh's config.

    Same file, same variable names, same precedence (real environment wins) — one place
    to write down where the box is.
    """
    env = {}
    path = ROOT / "scripts" / "publish.env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    for key in ("PUBLISH_HOST", "PUBLISH_USER", "PUBLISH_KEY"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def fetch_remote(args):
    """Cat the live log plus every rotation of it, over SSH, in one connection.

    Rotations are included because the interesting question is usually about a launch
    that has since rotated. `gzip -cdf` decompresses the .gz ones and passes the plain
    ones through, so both arrive on the same stream; order does not matter because the
    rows carry timestamps and get sorted here.
    """
    env = load_env()
    host = args.host or env.get("PUBLISH_HOST")
    if not host:
        sys.exit("✗ no host to read from.\n"
                 "  Set PUBLISH_HOST in scripts/publish.env (same file publish.sh uses),\n"
                 "  pass --host <ip>, or read a local copy with --file <path>.")
    user = args.user or env.get("PUBLISH_USER") or "ubuntu"
    key = args.key or env.get("PUBLISH_KEY")

    log = args.remote_log
    pipeline = (
        f"find {shlex.quote(str(pathlib.PurePosixPath(log).parent))} -maxdepth 1 -type f "
        f"-name {shlex.quote(pathlib.PurePosixPath(log).name + '*')} -print0 "
        f"| xargs -0 -r gzip -cdf --"
    )
    # No sudo by default. /var/log/nginx is root:adm 0640, and `adm` membership is the
    # supported way to read logs on Debian/Ubuntu -- one `usermod -aG adm` beats widening
    # the NOPASSWD sudoers entry publish.sh needs, which is scoped to rsync and chown and
    # would have to grow a shell to cover this. --sudo is the fallback for a box where
    # adm was not granted; it prompts unless that user has passwordless sudo.
    remote = f"sudo sh -c {shlex.quote(pipeline)}" if args.sudo else pipeline

    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new"]
    if key:
        cmd += ["-i", os.path.expanduser(key)]
    cmd += [f"{user}@{host}", remote]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip()
        sys.exit(f"✗ could not read {log} on {host}:\n  {detail}\n"
                 f"  Permission denied → the login user is not in `adm`. On the box:\n"
                 f"      sudo usermod -aG adm {user}    # then reconnect\n"
                 f"    or re-run this with --sudo.\n"
                 f"  No such file → the updated deploy/nginx.conf is not installed yet;\n"
                 f"    see \"6a. Campaign tracking\" in DEPLOY.md.")
    return proc.stdout


def read_rows(args):
    """Parse the log into dicts, whatever it was fetched from."""
    if args.file:
        path = pathlib.Path(args.file)
        if not path.exists():
            sys.exit(f"✗ no such file: {path}")
        text = path.read_text(errors="replace")
    else:
        text = fetch_remote(args)

    rows, malformed = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != len(FIELDS):
            malformed += 1
            continue
        row = {name: unescape(value) for name, value in zip(FIELDS, parts)}
        row["day"] = row["time"][:10]
        # $request_uri keeps the query string; the parameters are already their own
        # fields, so the grouping key is the path a visitor actually landed on.
        row["page"] = row["page"].split("?", 1)[0]
        row["bot"] = bool(BOT_RE.search(row["agent"]))
        rows.append(row)

    if malformed:
        print(f"note: skipped {malformed} line(s) that did not have {len(FIELDS)} "
              f"tab-separated fields — a log written before the current log_format?",
              file=sys.stderr)
    rows.sort(key=lambda r: r["time"])
    return rows


def filter_rows(rows, args):
    kept = []
    for row in rows:
        if not args.include_bots and row["bot"]:
            continue
        if args.since and row["day"] < args.since:
            continue
        if args.until and row["day"] > args.until:
            continue
        if args.source and row["source"].lower() != args.source.lower():
            continue
        if args.campaign and row["campaign"].lower() != args.campaign.lower():
            continue
        kept.append(row)
    return kept


# -------------------------------------------------------------------------- commands

def cmd_report(args):
    rows = read_rows(args)
    if not rows:
        print("(no tagged requests logged yet)\n"
              "Nothing has been clicked, or the log is not being written — check with:\n"
              "  sudo tail /var/log/nginx/postuptothe.utm.tsv")
        return

    bots = sum(1 for r in rows if r["bot"])
    kept = filter_rows(rows, args)
    dims = [d.strip() for d in args.by.split(",") if d.strip()]
    unknown = [d for d in dims if d not in GROUPABLE]
    if unknown:
        sys.exit(f"✗ cannot group by {', '.join(unknown)} — pick from: {', '.join(GROUPABLE)}")

    groups = collections.defaultdict(list)
    for row in kept:
        groups[tuple(row[d] or "(none)" for d in dims)].append(row)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    if args.json:
        print(json.dumps([
            {**dict(zip(dims, key)), "hits": len(rs),
             "first": rs[0]["time"], "last": rs[-1]["time"]}
            for key, rs in ordered
        ], indent=2))
        return

    # The span describes what is being reported, not what is in the file — quoting the
    # log's full range under a --since would misdescribe every number below it.
    shown = kept or rows
    span = f"{shown[0]['day']} → {shown[-1]['day']}"
    excluded = "" if args.include_bots else f", {bots} crawler hit(s) excluded"
    print(f"{len(kept)} tagged landing(s) over {span}{excluded}\n")

    if not kept:
        print("(everything was filtered out — try --include-bots or a wider --since)")
        return

    widths = [max(len(d), max(len(k[i]) for k, _ in ordered)) for i, d in enumerate(dims)]
    header = ("  ".join(d.ljust(w) for d, w in zip(dims, widths))
              + f"  {'hits':>5}  first        last")
    print(header)
    print("-" * len(header))
    for key, rs in ordered:
        cells = "  ".join(part.ljust(w) for part, w in zip(key, widths))
        print(f"{cells}  {len(rs):>5}  {rs[0]['day']}   {rs[-1]['day']}")


def cmd_raw(args):
    rows = filter_rows(read_rows(args), args)
    if not rows:
        print("(nothing matched)")
        return
    for row in rows:
        print(f"{row['time']}  {row['status']}  {row['source']}/{row['medium'] or '-'}"
              f"/{row['campaign'] or '-'}  {row['page']}"
              f"{'  [bot]' if row['bot'] else ''}")
        if args.agents:
            print(f"    ref={row['referer'] or '-'}\n    ua={row['agent'] or '-'}")


SHORTLINK_RE = re.compile(r"^\s*(/go/\S+)\s+\"([^\"]+)\"\s*;", re.M)


def cmd_links(args):
    """Print the shareable links, read from the one place they are defined.

    The `$shortlink` map in deploy/nginx.conf is the source of truth — retyping the
    slugs here would mean a link that this command prints and nginx 404s.
    """
    if not NGINX_CONF.exists():
        sys.exit(f"✗ {NGINX_CONF} not found")
    entries = SHORTLINK_RE.findall(NGINX_CONF.read_text())
    if not entries:
        print("(no /go/ short links defined — add them to the $shortlink map in "
              f"{NGINX_CONF.relative_to(ROOT)})")
        return

    base = args.base.rstrip("/")
    print(f"{len(entries)} campaign link(s) from {NGINX_CONF.relative_to(ROOT)}:\n")
    missing = 0
    for slug, target in entries:
        path = target.split("?", 1)[0].lstrip("/")
        exists = (SITE / path).exists()
        if not exists:
            missing += 1
        print(f"  {base}{slug}")
        print(f"    → {target}{'' if exists else '   ⚠ no such page in site/'}")
    print()
    if missing:
        print(f"⚠ {missing} link(s) point at a page that does not exist in site/ — "
              f"they will 404 for anyone who clicks them.")
    print("Paste the /go/ form into a post. nginx 302s it to the target, and the landing\n"
          "request (not the redirect) is what lands in the campaign log.")


def main():
    p = argparse.ArgumentParser(
        description="Read the nginx campaign log and list the tracking links.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Reads the log off the box over SSH by default, using the host and key in "
               "scripts/publish.env.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_source_flags(sp):
        sp.add_argument("--file", help="read a local log file instead of fetching over SSH")
        sp.add_argument("--host", help="override PUBLISH_HOST from scripts/publish.env")
        sp.add_argument("--user", help="override PUBLISH_USER (default: ubuntu)")
        sp.add_argument("--key", help="override PUBLISH_KEY (path to an SSH .pem)")
        sp.add_argument("--remote-log", default=DEFAULT_LOG,
                        help=f"log path on the box (default: {DEFAULT_LOG}); rotations of "
                             f"it are read too")
        sp.add_argument("--sudo", action="store_true",
                        help="read the log via sudo, for a box where the login user is "
                             "not in the `adm` group")

    def add_filter_flags(sp):
        sp.add_argument("--since", metavar="YYYY-MM-DD", help="drop landings before this day")
        sp.add_argument("--until", metavar="YYYY-MM-DD", help="drop landings after this day")
        sp.add_argument("--source", help="only this utm_source (e.g. linkedin)")
        sp.add_argument("--campaign", help="only this utm_campaign")
        sp.add_argument("--include-bots", action="store_true",
                        help="keep preview crawlers, which are excluded by default")

    r = sub.add_parser("report", help="grouped hit counts")
    add_source_flags(r)
    add_filter_flags(r)
    r.add_argument("--by", default="source,medium,campaign",
                   help=f"comma-separated grouping ({', '.join(GROUPABLE)}); "
                        f"default: source,medium,campaign")
    r.add_argument("--json", action="store_true", help="machine-readable output")
    r.set_defaults(func=cmd_report)

    w = sub.add_parser("raw", help="the matching log lines, unaggregated")
    add_source_flags(w)
    add_filter_flags(w)
    w.add_argument("--agents", action="store_true", help="also show referer and user agent")
    w.set_defaults(func=cmd_raw)

    k = sub.add_parser("links", help="the /go/ links to paste into a post")
    k.add_argument("--base", default="https://postuptothe.net",
                   help="site base URL (default: https://postuptothe.net)")
    k.set_defaults(func=cmd_links)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
