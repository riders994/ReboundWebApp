# Headshot slideshow

Drop image files here (`.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`, `.avif`) — they become the
rotating photo in the landing-page hero, cycled in sorted filename order.

After adding or removing images, regenerate the manifest:

```bash
python3 scripts/gen-manifests.py
```

`scripts/serve.sh` also runs this automatically before starting the local site, and the
deploy step runs it before publishing. That writes `manifest.json` (the list the slideshow reads).

This is the `headshots` **feed** (source `local`) declared in `scripts/feeds.json`. To source a
feed from an S3 bucket instead of the repo, see `scripts/README.md`. With no images, the hero shows a
placeholder. With one image it just displays; with several it crossfades every few seconds.
