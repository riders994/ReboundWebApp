"""Shared data access for the comedy tag scripts (register-tag.py, comedy-tags.py).

Central place for the file paths, JSON load/save, and the meta-tag code rule so the two
tools can't drift apart.
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "site" / "assets" / "data"
TAGS_FILE = DATA / "comedy-tags.json"
CLIPS_FILE = DATA / "comedy-clips.json"

# meta-tag codes: lowercase, digits and hyphens, start alphanumeric, <= 24 chars
CODE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,23}$")

# YouTube video ids are 11 chars of [A-Za-z0-9_-]
YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(text):
    """Return the 11-char YouTube id from a bare id or a watch/youtu.be/shorts/embed URL."""
    text = (text or "").strip()
    if YT_ID_RE.match(text):
        return text
    m = re.search(r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})", text)
    return m.group(1) if m else None


def unregistered(reg, codes):
    """Return the subset of codes that are not in the registry."""
    return [c for c in codes if c not in reg["tags"]]


def _load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_registry():
    reg = _load(TAGS_FILE, {"categories": [], "tags": {}})
    reg.setdefault("categories", [])
    reg.setdefault("tags", {})
    return reg


def save_registry(reg):
    _save(TAGS_FILE, reg)


def load_clips():
    return _load(CLIPS_FILE, [])


def save_clips(clips):
    _save(CLIPS_FILE, clips)


def norm_clip(entry):
    """Clips may be a bare id string or an object; return a mutable dict form."""
    if isinstance(entry, str):
        return {"id": entry, "tags": []}
    entry.setdefault("tags", [])
    return entry


def find_clip(clips, video_id):
    for i, entry in enumerate(clips):
        cid = entry if isinstance(entry, str) else entry.get("id")
        if cid == video_id:
            return i
    return -1


def strip_code_from_clips(code):
    """Remove a tag code from every video; return how many were touched."""
    clips = load_clips()
    n = 0
    for i, entry in enumerate(clips):
        c = norm_clip(entry)
        if code in c["tags"]:
            c["tags"] = [t for t in c["tags"] if t != code]
            clips[i] = c
            n += 1
    save_clips(clips)
    return n
