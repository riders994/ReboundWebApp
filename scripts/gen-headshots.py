#!/usr/bin/env python3
"""Regenerate the headshot slideshow manifest.

Scans site/assets/img/headshots/ for image files and writes manifest.json (a JSON
array of filenames, sorted). The landing-page slideshow reads that manifest. Run this
whenever you add or remove headshot images:

    python3 scripts/gen-headshots.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
IMG_DIR = HERE.parent / "site" / "assets" / "img" / "headshots"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}

images = sorted(
    p.name for p in IMG_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in EXTS
)
(IMG_DIR / "manifest.json").write_text(json.dumps(images, indent=2) + "\n")
print(f"wrote {len(images)} image(s) to {IMG_DIR / 'manifest.json'}")
