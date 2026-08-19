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

## serve.sh

Local dev entrypoint: refreshes manifests, then serves `site/`. `PORT=8080 ./scripts/serve.sh`.
