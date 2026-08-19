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

Add or reorder entries here, reload the page, and the embeds update. An empty array shows a
"add your clips" note.

Each clip's still is pulled automatically from YouTube's thumbnail
(`https://i.ytimg.com/vi/<id>/hqdefault.jpg`) — you never need to add an image. The grid shows
that still with a play button and only loads the YouTube player when clicked.
