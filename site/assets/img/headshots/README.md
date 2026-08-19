# Headshot slideshow

Drop image files here (`.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`, `.avif`) — they become the
rotating photo in the landing-page hero, cycled in sorted filename order.

After adding or removing images, regenerate the manifest:

```bash
python3 scripts/gen-headshots.py
```

That writes `manifest.json` (the list the slideshow reads). With no images, the hero shows a
placeholder. With one image it just displays; with several it crossfades every few seconds.
