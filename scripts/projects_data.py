"""Shared data access for the projects framework (projects.py).

Central place for the projects.json path, load/save, slug/repo parsing, the "featured =
5 most recently designated" rule, and README fetching, so the CLI and any other tooling
stay consistent.
"""
import datetime
import json
import pathlib
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
PROJECTS_FILE = SITE / "assets" / "data" / "projects.json"
OVERRIDE_DIR = ROOT / "content" / "projects"   # local README overrides live here
DETAIL_DIR = SITE / "projects"                  # generated <slug>.html pages

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,48}$")
FEATURED_LIMIT = 5


def today():
    return datetime.date.today().isoformat()


def load_projects():
    if not PROJECTS_FILE.exists():
        return []
    return json.loads(PROJECTS_FILE.read_text()).get("projects", [])


def save_projects(projects):
    PROJECTS_FILE.write_text(json.dumps({"projects": projects}, indent=2) + "\n")


def find(projects, slug):
    for i, p in enumerate(projects):
        if p.get("slug") == slug:
            return i
    return -1


def parse_repo(url):
    """Return (owner, repo) from a GitHub URL, or (None, None)."""
    m = re.search(r"github\.com[:/]+([^/]+)/([^/#?]+?)(?:\.git)?/?$", url or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def featured_slugs(projects, limit=FEATURED_LIMIT):
    """The `limit` projects most recently designated featured (by featured_at date)."""
    dated = [p for p in projects if p.get("featured_at")]
    dated.sort(key=lambda p: p["featured_at"], reverse=True)
    return {p["slug"] for p in dated[:limit]}


def render_markdown(text):
    import markdown
    return markdown.markdown(
        text, extensions=["fenced_code", "tables", "sane_lists", "toc"], output_format="html5"
    )


def fetch_readme(owner, repo, branch=None):
    """Fetch a repo's README from raw.githubusercontent.com.

    Returns (text, raw_base_url) or (None, None). raw_base_url is the directory URL of the
    README on the resolved branch, so relative image paths can be rewritten to absolute.
    """
    branches = [branch] if branch else ["main", "master"]
    names = ["README.md", "readme.md", "README.markdown"]
    for b in branches:
        for name in names:
            base = f"https://raw.githubusercontent.com/{owner}/{repo}/{b}/"
            try:
                with urlopen(Request(base + name, headers={"User-Agent": "Mozilla/5.0"}), timeout=15) as r:
                    return r.read().decode("utf-8", "replace"), base
            except (HTTPError, URLError, TimeoutError, OSError):
                continue
    return None, None
