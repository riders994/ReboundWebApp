# Default card covers

Images here are used as the picture on any card that has no image of its own — most
project cards, which look for `assets/img/<slug>-thumb.png` and until now fell back to an
empty tinted box.

Drop image files here (`.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`, `.avif`, `.svg`). The
committed set is twelve abstract covers built from the site's own palette, with a
`prefers-color-scheme` block inside each SVG so they follow light/dark like everything
else. Replace or add to them freely — nothing in the code knows their names.

After adding or removing images, regenerate the manifest:

```bash
python3 scripts/gen-manifests.py
```

`scripts/serve.sh` runs this before serving locally, and `scripts/publish.sh` runs it
before deploying. It writes `manifest.json`, the list the front-end reads.

This is the `fallbacks` **feed** (source `local`) declared in `scripts/feeds.json`, the
same mechanism as the headshot slideshow; it can be pointed at an S3 bucket instead — see
`scripts/README.md`.

## How covers get assigned

[`assets/js/thumbs.js`](../../js/thumbs.js) deals covers to a page from a shuffled bag,
rather than picking one at random per card, so a cover is only reused once every other one
has been used. With more covers than cards — twelve against eleven projects today — no
page repeats a cover at all. Keep at least as many covers here as the largest card count
on a page (the projects index shows the highlight grid plus six completed cards at once).

The shuffle is seeded from the manifest rather than from `Math.random()`, so a project
keeps the same cover across reloads and shows the same one on the projects index and in
the resume carousel. Adding or removing a cover reshuffles the deck.

To give a project a picture of its own instead, add `assets/img/<slug>-thumb.png` — the
card prefers it and the default is never fetched.
