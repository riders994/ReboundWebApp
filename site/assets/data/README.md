# Comedy clips

`comedy-clips.json` drives the Clips grid on `comedy.html`. It's a JSON array; each entry is
a YouTube video. Use either the bare video ID or an object with a title:

```json
[
  "dQw4w9WgXcQ",
  { "id": "dQw4w9WgXcQ", "title": "My Tight Five at the Comedy Cellar" }
]
```

The **video ID** is the part after `watch?v=` in a YouTube URL
(`https://www.youtube.com/watch?v=`**`dQw4w9WgXcQ`**), or after `youtu.be/`.

## Tags

`comedy-tags.json` is the tag registry: a `categories` list plus a `tags` map of your
short **meta-tag** codes to a reader-facing **normalized** label and category, e.g.
`"kkj": { "label": "Knock-Knock Joke", "category": "joke" }`. Each clip in
`comedy-clips.json` may carry a `"tags": ["kkj", "chi"]` list, and the Comedy page renders
a filter bar (grouped by category) so visitors can narrow the videos. Selected tags are
reflected in the URL (`?ctags=kkj,chi`) so a filtered view is shareable.

Don't hand-edit tags — use the scripts, which keep both files consistent. Registering the
vocabulary and applying it to videos are two separate steps:

```bash
# formal process: manage the registry vocabulary (validated)
python3 scripts/register-tag.py add kkj --label "Knock-Knock Joke" --category joke
python3 scripts/register-tag.py list

# operational: submit a new video (id or URL), or tag an existing one
python3 scripts/comedy-tags.py add   "https://youtu.be/VIDEOID" --title "Cellar Set" --tags kkj
python3 scripts/comedy-tags.py tag   <VIDEO_ID> kkj chi
python3 scripts/comedy-tags.py untag <VIDEO_ID> chi
python3 scripts/comedy-tags.py videos    # list videos + their tags
```

Add or reorder entries here, reload the page, and the embeds update. An empty array shows a
"add your clips" note.

Each clip's still is pulled automatically from YouTube's thumbnail
(`https://i.ytimg.com/vi/<id>/hqdefault.jpg`) — you never need to add an image. The grid shows
that still with a play button and only loads the YouTube player when clicked.
