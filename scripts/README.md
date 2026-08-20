# Scripts

## Photo feeds

A **feed** is a set of photos the site shows (the headshot slideshow today; vacation, cat,
etc. later). Feeds are declared in [`feeds.json`](feeds.json) and turned into a
`manifest.json` per feed by [`gen-manifests.py`](gen-manifests.py), which the front-end reads.

`scripts/serve.sh` and the deploy steps run the generator automatically, or run it by hand:

```bash
python3 scripts/gen-manifests.py
```

### Feed config (`feeds.json`)

Each feed is an object in the `feeds` array:

| field       | applies to | meaning                                                            |
|-------------|------------|--------------------------------------------------------------------|
| `name`      | all        | label for logs                                                     |
| `source`    | all        | `"local"` or `"s3"`                                                 |
| `dir`       | all        | directory under `site/` where `manifest.json` is written           |
| `enabled`   | all        | set `false` to skip                                                 |
| `bucket`    | s3         | S3 bucket name                                                      |
| `prefix`    | s3         | key prefix to list (e.g. `"vacation/"`)                            |
| `base_url`  | s3         | public URL base; defaults to `https://<bucket>.s3.amazonaws.com/` (use your CloudFront domain if you have one) |

### `local` feeds (current headshots flow)

Images live in the repo under `dir`; the manifest lists **filenames** and the page prepends
`dir`. Drop images in the folder, run the generator, commit. No dependencies.

### `s3` feeds (future-proofing for large/rotating sets)

Images live in an S3 bucket; the manifest lists **full public URLs**, so **no image files
are committed** — only the small `manifest.json`. Setup:

1. `pip install -r scripts/requirements.txt` (installs boto3) and configure AWS credentials
   (`aws configure` / env vars / instance role) with `s3:ListBucket` on the bucket.
2. Make the objects readable by browsers: either a public-read bucket/objects, or serve them
   through CloudFront and set `base_url` to that domain. (`<img>` loads need no CORS.)
3. Add/enable the feed in `feeds.json`, then `python3 scripts/gen-manifests.py`.
4. Commit the regenerated `manifest.json`. To display it, point a slideshow element at that
   feed's manifest (see the hero `data-slideshow` in `site/index.html`).

To update photos later: upload to the bucket, re-run the generator, commit the manifest.

> Tip: if you ever keep local copies of S3 images inside a feed dir, gitignore the image
> files there (`site/assets/img/<feed>/*.jpg` etc.) and keep only `manifest.json` tracked.

## Comedy tags

Two-step, two-tool workflow. You invent short **meta-tag** codes (`kkj`) that map to
reader-facing **normalized** labels ("Knock-Knock Joke") within a category
(location / date / subject / joke), then apply them to videos. The Comedy page reads the same
files (`site/assets/data/comedy-tags.json` + `comedy-clips.json`) and lets visitors filter.
Both tools share `comedy_data.py` and have no dependencies.

**1. Register the vocabulary — `register-tag.py`** (the formal process). Validates the code
format, refuses silent overwrites (`--force` to replace), and only accepts a known category
unless you pass `--new-category`. Prompts interactively for anything you omit.

```bash
python3 scripts/register-tag.py add kkj --label "Knock-Knock Joke" --category joke
python3 scripts/register-tag.py import-csv tags.csv     # bulk upsert; columns: tag,label,category
python3 scripts/register-tag.py add-category location
python3 scripts/register-tag.py list
python3 scripts/register-tag.py retire kkj --strip     # remove a code (and pull it off videos)
```

`import-csv` upserts: rows with a new `tag` are added, rows whose `label`/`category` changed
are updated, unchanged rows are left alone, and codes **not** in the CSV are never removed.
The whole file is validated first (code format, required fields, no duplicate tags), so a bad
row aborts the import without writing anything.

**2. Add videos and apply codes — `comedy-tags.py`** (operational). `add` submits a new
video (from a YouTube id or URL), optionally with tags; its title is fetched automatically
from YouTube (keyless oEmbed) unless you pass `--title`. `tag`/`untag` adjust an existing
one. Both `add` and `tag` reject any code that isn't registered, so every tag on a video
has a label and category for the site.

```bash
python3 scripts/comedy-tags.py add   "https://youtu.be/VIDEOID" --tags kkj chi   # title auto-fetched
python3 scripts/comedy-tags.py add   "https://youtu.be/VIDEOID" --title "Custom Caption"
python3 scripts/comedy-tags.py tag   <VIDEO_ID> kkj chi     # tag an existing video
python3 scripts/comedy-tags.py untag <VIDEO_ID> chi
python3 scripts/comedy-tags.py videos      # list videos + their tags
python3 scripts/comedy-tags.py tags        # list the registry (read-only)
```

## Projects (`projects.py`)

The projects section is data-driven from `site/assets/data/projects.json`. Each project is
git-based: a GitHub repo, an optional web-app link, a status (`in-flight`/`completed`), and an
optional "featured on" date. The projects page shows a highlight of **featured + in-flight**
projects, then a paginated list of **completed** ones (newest first). `featured` = the **5
projects most recently designated** (by their featured date). Marking a project completed
records `completed_at` from the repo's **most recent commit** (via the GitHub API; falls back
to today if unreachable), so the completed list's newest-first order tracks real repo activity.

Each detail page's writeup is the repo's **README, rendered to HTML at build time**. Override
it by dropping `content/projects/<slug>.md` in THIS repo. Projects marked `"page": "custom"`
(e.g. the rebound demo, with its embedded live app) keep a hand-authored page under
`site/projects/` and are skipped by `render`.

```bash
python3 scripts/projects.py add <slug> --name "..." --repo <url> [--blurb ..] [--tags a b] \
        [--webapp <url>] [--status in-flight|completed] [--page generated|custom]
python3 scripts/projects.py feature   <slug>          # designate featured (stamps today)
python3 scripts/projects.py unfeature <slug>
python3 scripts/projects.py status    <slug> completed
python3 scripts/projects.py list
python3 scripts/projects.py render                    # READMEs/overrides -> detail pages
```

`render` needs `markdown` (`pip install -r scripts/requirements.txt`). `serve.sh` and the
deploy steps run it automatically.

## serve.sh

Local dev entrypoint: refreshes manifests, then serves `site/`. `PORT=8080 ./scripts/serve.sh`.
