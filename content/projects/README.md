# Project writeup overrides

Drop `<slug>.md` here to override a project's detail-page writeup with content from THIS
repo instead of fetching the project repo's README. Useful when you want a curated writeup,
or when a project's page needs different content than its README.

`scripts/projects.py render` uses this file if present; otherwise it fetches the repo's
README. Projects marked `"page": "custom"` (e.g. the rebound demo, which has a hand-authored
page with an embedded app) are skipped by `render` entirely — their page is edited directly
under `site/projects/`.
