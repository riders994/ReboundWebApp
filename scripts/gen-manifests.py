#!/usr/bin/env python3
"""Regenerate media manifests for directory-driven galleries.

Each entry in GALLERIES is a directory (under site/) that gets a manifest.json listing
its media files, sorted by filename. The front-end reads those manifests to build
slideshows/galleries, so newly added files show up after this runs. The startup script
(scripts/serve.sh) runs this before serving, and the deploy step runs it before rsync.

To make another section directory-driven, add its path (and allowed extensions) to
GALLERIES and read `<dir>/manifest.json` from the page.

    python3 scripts/gen-manifests.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE = ROOT / "site"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}

# directory (relative to site/) -> allowed file extensions
GALLERIES = {
    "assets/img/headshots": IMG_EXTS,
    # add more directory-driven sections here, e.g.:
    # "assets/img/comedy": IMG_EXTS,
}


def build(rel, exts):
    directory = SITE / rel
    if not directory.is_dir():
        print(f"skip {rel} (directory missing)")
        return
    files = sorted(
        p.name for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    (directory / "manifest.json").write_text(json.dumps(files, indent=2) + "\n")
    print(f"{rel}: {len(files)} file(s)")


if __name__ == "__main__":
    for rel, exts in GALLERIES.items():
        build(rel, exts)
