#!/usr/bin/env python3
"""Manage comedy videos and their tags (operational tool).

Adds new videos, applies/removes registry tag codes on videos in
site/assets/data/comedy-clips.json, and lists what's tagged. Tag codes must first be
registered with register-tag.py — that's the formal process for managing the tag
vocabulary; this tool just applies them. Any flow given an unregistered tag errors out.

Usage:
  comedy-tags.py add    <VIDEO_ID|URL> [--tags kkj chi]   # submit a new video (title auto-fetched)
  comedy-tags.py tag    <VIDEO_ID> kkj chi     # add tags to an existing video
  comedy-tags.py untag  <VIDEO_ID> chi         # remove tags from a video
  comedy-tags.py videos                        # list videos with their resolved tags
  comedy-tags.py tags                          # list the registry (read-only)
"""
import argparse
import sys

import comedy_data as cd


def _require_registered(reg, codes):
    """Exit with an error if any code isn't in the registry (shared by add + tag)."""
    unknown = cd.unregistered(reg, codes)
    if unknown:
        sys.exit("unregistered tag code(s): " + ", ".join(unknown) +
                 "\nRegister them first, e.g.:  register-tag.py add " + unknown[0] +
                 ' --label "..." --category subject')


def cmd_tags(args):
    reg = cd.load_registry()
    if not reg["tags"]:
        print("(no tags registered yet — use: register-tag.py add ...)")
        return
    for cat in reg["categories"]:
        codes = sorted(c for c, t in reg["tags"].items() if t["category"] == cat)
        if not codes:
            continue
        print(f"\n{cat}:")
        for code in codes:
            print(f"  {code:<14} {reg['tags'][code]['label']}")


def cmd_videos(args):
    reg = cd.load_registry()
    clips = cd.load_clips()
    if not clips:
        print("(no videos in comedy-clips.json)")
        return
    for entry in clips:
        c = cd.norm_clip(entry)
        title = c.get("title", "(untitled)")
        labels = [reg["tags"].get(t, {}).get("label", t + "?") for t in c["tags"]]
        print(f"{c['id']}  {title}")
        print(f"    tags: {', '.join(labels) if labels else '(none)'}")


def cmd_add(args):
    reg = cd.load_registry()
    vid = cd.extract_video_id(args.video)
    if not vid:
        sys.exit(f"could not parse a YouTube video id from '{args.video}' "
                 f"(pass the 11-char id, or a full watch / youtu.be / shorts URL)")
    codes = args.tags or []
    _require_registered(reg, codes)  # same guard as the tag flow

    clips = cd.load_clips()
    idx = cd.find_clip(clips, vid)
    if idx >= 0 and not args.force:
        sys.exit(f"video '{vid}' already exists — use `comedy-tags.py tag {vid} ...` to add "
                 f"tags, or --force to overwrite its entry")

    # Title: use --title if given; otherwise fetch it from YouTube (keyless oEmbed).
    title = args.title
    if not title:
        title = cd.fetch_youtube_title(vid)
        if title:
            print(f"  fetched title: {title}")
        else:
            print("  (couldn't fetch title from YouTube — pass --title to set one)")

    entry = {"id": vid}
    if title:
        entry["title"] = title
    entry["tags"] = list(dict.fromkeys(codes))  # dedupe, keep order

    if idx >= 0:
        clips[idx] = entry
        action = "updated"
    else:
        clips.append(entry)
        action = "added"
    cd.save_clips(clips)

    labels = [reg["tags"][c]["label"] for c in entry["tags"]]
    print(f"{action} {vid}" + (f" — {args.title}" if args.title else ""))
    print(f"    tags: {', '.join(labels) if labels else '(none)'}")


def cmd_tag(args):
    reg = cd.load_registry()
    _require_registered(reg, args.codes)
    clips = cd.load_clips()
    idx = cd.find_clip(clips, args.video_id)
    if idx < 0:
        sys.exit(f"video '{args.video_id}' is not in comedy-clips.json — add it there first")
    c = cd.norm_clip(clips[idx])
    for code in args.codes:
        if code not in c["tags"]:
            c["tags"].append(code)
    clips[idx] = c
    cd.save_clips(clips)
    print(f"{args.video_id} tags: {', '.join(c['tags']) or '(none)'}")


def cmd_untag(args):
    clips = cd.load_clips()
    idx = cd.find_clip(clips, args.video_id)
    if idx < 0:
        sys.exit(f"video '{args.video_id}' is not in comedy-clips.json")
    c = cd.norm_clip(clips[idx])
    c["tags"] = [t for t in c["tags"] if t not in args.codes]
    clips[idx] = c
    cd.save_clips(clips)
    print(f"{args.video_id} tags: {', '.join(c['tags']) or '(none)'}")


def main():
    p = argparse.ArgumentParser(description="Apply comedy tags to videos.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tags", help="list the registry (read-only)").set_defaults(func=cmd_tags)
    sub.add_parser("videos", help="list videos with their tags").set_defaults(func=cmd_videos)

    a = sub.add_parser("add", help="submit a new video (optionally with tags)")
    a.add_argument("video", help="YouTube video id or URL")
    a.add_argument("--title", help="caption shown on the site (default: fetched from YouTube)")
    a.add_argument("--tags", nargs="*", default=[], help="registered tag codes to apply")
    a.add_argument("--force", action="store_true", help="overwrite if the video already exists")
    a.set_defaults(func=cmd_add)

    t = sub.add_parser("tag", help="add tags to a video")
    t.add_argument("video_id")
    t.add_argument("codes", nargs="+")
    t.set_defaults(func=cmd_tag)

    ut = sub.add_parser("untag", help="remove tags from a video")
    ut.add_argument("video_id")
    ut.add_argument("codes", nargs="+")
    ut.set_defaults(func=cmd_untag)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
