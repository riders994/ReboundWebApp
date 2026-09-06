# Launch checklist

Open items before announcing. Line numbers verified 2026-09-05 at `faec2bf`.
The site is already live (`postuptothe.net`, backend healthy) — this is the
announcement, not the first deploy.

## 1. Deploy what's already written

The **backend on the box is running pre-2026-08-27 code**: live `/healthz`
returns no `telemetry` key, which current `rebound-app/app.py:188` emits
unconditionally. Undeployed: `3b6b181` (Postgres prediction logging) and
`3e67afc` (gunicorn log routing).

- Backend redeploy — [DEPLOY.md](DEPLOY.md) "Redeploying after changes", line 493
- Static site is also behind (`assets/js/thumbs.js` 404s on prod) — `scripts/publish.sh`
- Verify after: `curl -s https://postuptothe.net/api/rebound/healthz` should show `telemetry`

**Decide before announcing, not after:** prediction logging stays off until
`REBOUND_DATABASE_URL` is set — [DEPLOY.md](DEPLOY.md) step 5d, line 368. Launch
traffic is the traffic that answers how far real usage sits from the corpus the
29.7% was measured on. Can't be collected retroactively.

## 2. Dev notes currently visible to visitors

All three render as dashed boxes on public pages.

| File | Line | What |
|---|---|---|
| `site/projects/rebounding.html` | 190 | "Running locally? Start the Flask app…" — on the flagship demo page |
| `site/blog/posts/building-the-rebound-predictor.html` | 31 | "TODO: this is a starter draft — edit it into your own voice." |
| `site/blog/index.html` | 39 | Visible `<li>`: "TODO: add more posts under `blog/posts/`…" |

## 3. Starter copy still in place

| File | Line | What |
|---|---|---|
| `site/index.html` | 34 | Hero one-liner — first thing on the site |
| `site/index.html` | 59 | About/bio paragraph |

## 4. Content gaps

**`position-predictor` has an empty blurb** and is *featured*, so it sits in the
highlight grid and the resume carousel with a title, five tags, and no
description. Edit `site/assets/data/projects.json:45`, then
`python3 scripts/projects.py render`. (There's no CLI subcommand for blurbs —
`add` sets them, nothing edits them.)

**All 11 comedy clips are untagged** while `comedy-tags.json` registers 73 tags
across 8 categories, so the filter bar on `comedy.html` renders nothing but the
Sort control. Largest built-and-unused surface on the site, and the biggest time
sink here — the natural thing to cut if next week gets tight.

```bash
python3 scripts/comedy-tags.py videos          # list clips + current tags
python3 scripts/comedy-tags.py tag <VIDEO_ID> kkj chi
```

## 5. Stale doc

`README.md:12` still says "resume + PDF download". The download was deliberately
dropped in `cf51e7a`. One-line fix.

---

Not blocking, noted for later: blog has one post; `assets/img/headshots/` has one
image (the hero slideshow only crossfades with two or more).
