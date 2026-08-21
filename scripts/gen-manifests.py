#!/usr/bin/env python3
"""Regenerate per-feed image manifests.

A "feed" is a set of photos the site displays (e.g. the headshot slideshow; later maybe
vacation, cat, ...). Feeds are declared in scripts/feeds.json. Each enabled feed writes a
manifest.json into its directory under site/, which the front-end reads to build the
slideshow/gallery.

Two sources are supported:
  - "local": images live in the repo under the feed's `dir`; the manifest lists their
    filenames (the page prepends `dir` as the base path). This is the default, no-deps flow.
  - "s3": images live in an S3 bucket; the manifest lists their full public URLs, so no
    image files are committed — only the small manifest.json is. Requires boto3
    (pip install -r scripts/requirements.txt) and AWS credentials with s3:ListBucket.

Run before serving/deploying (scripts/serve.sh and the deploy steps call this):
    python3 scripts/gen-manifests.py
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
CONFIG = pathlib.Path(__file__).resolve().parent / "feeds.json"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}


def is_image(name):
    return pathlib.PurePosixPath(name).suffix.lower() in IMG_EXTS


def write_manifest(directory, items):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(items, indent=2) + "\n")


def gen_local(feed):
    """List image filenames in the repo directory."""
    directory = SITE / feed["dir"]
    if not directory.is_dir():
        print(f"  skip (directory missing: {feed['dir']})")
        return None
    files = sorted(p.name for p in directory.iterdir() if p.is_file() and is_image(p.name))
    write_manifest(directory, files)
    return len(files)


def gen_s3(feed):
    """List image objects in an S3 bucket and record their full public URLs."""
    keys = _list_s3(feed["bucket"], feed.get("prefix", ""))
    base = feed.get("base_url") or f"https://{feed['bucket']}.s3.amazonaws.com/"
    if not base.endswith("/"):
        base += "/"
    urls = sorted(base + k for k in keys if is_image(k))
    write_manifest(SITE / feed["dir"], urls)
    return len(urls)


def _list_s3(bucket, prefix):
    try:
        import boto3
    except ImportError:
        sys.exit("  boto3 not installed — run: pip install -r scripts/requirements.txt")
    s3 = boto3.client("s3")
    keys, token = [], None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        keys.extend(obj["Key"] for obj in resp.get("Contents", []))
        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            break
    return keys


SOURCES = {"local": gen_local, "s3": gen_s3}


def main():
    feeds = json.loads(CONFIG.read_text()).get("feeds", [])
    for feed in feeds:
        name = feed.get("name", "?")
        if not feed.get("enabled", True):
            print(f"{name}: disabled — skipping")
            continue
        source = feed.get("source", "local")
        handler = SOURCES.get(source)
        if handler is None:
            print(f"{name}: unknown source '{source}' — skipping")
            continue
        print(f"{name} [{source}] -> {feed['dir']}")
        count = handler(feed)
        if count is not None:
            print(f"  {count} image(s)")


if __name__ == "__main__":
    main()
