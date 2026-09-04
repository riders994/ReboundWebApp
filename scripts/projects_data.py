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


def is_draft(project):
    """Draft projects are held back: not listed anywhere, and no detail page is built."""
    return bool(project.get("draft"))


def published(projects):
    """Everything except drafts — what the site is allowed to show."""
    return [p for p in projects if not is_draft(p)]


def featured_slugs(projects, limit=FEATURED_LIMIT):
    """The `limit` projects most recently designated featured (by featured_at date).

    Drafts are excluded first, so a held-back project cannot sit on a featured slot
    and quietly shrink the highlight row. The JS does the same, in the same order.
    """
    dated = [p for p in published(projects) if p.get("featured_at")]
    dated.sort(key=lambda p: p["featured_at"], reverse=True)
    return {p["slug"] for p in dated[:limit]}


def fetch_latest_commit_date(owner, repo, branch=None):
    """Return the date (YYYY-MM-DD) of the most recent commit, or None on failure.

    Uses the unauthenticated GitHub API (rate-limited but fine for occasional CLI use).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=1"
    if branch:
        url += f"&sha={branch}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}
    try:
        with urlopen(Request(url, headers=headers), timeout=15) as r:
            data = json.load(r)
        return data[0]["commit"]["committer"]["date"][:10]  # "2025-07-31T..." -> "2025-07-31"
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError, OSError):
        return None


def render_markdown(text):
    import markdown
    return markdown.markdown(
        text, extensions=["fenced_code", "tables", "sane_lists", "toc"], output_format="html5"
    )


README_NAMES = ["README.md", "readme.md", "README.markdown"]

# Guessed in this order before spending an API call. Measured against this account's
# repos (Aug 2026): 9 default to "primary", 2 to "master", and none to "main" — so the
# old ["main", "master"] order missed every time and burned three requests doing it.
# "main" stays last because it is GitHub's default for new repos, so a repo created
# tomorrow will still resolve without an API call.
BRANCH_GUESSES = ["primary", "master", "main"]


def fetch_default_branch(owner, repo):
    """The repo's actual default branch, or None. One unauthenticated API call."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}
    try:
        with urlopen(Request(url, headers=headers), timeout=15) as r:
            return json.load(r).get("default_branch")
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError, OSError):
        return None


def _try_branches(owner, repo, branches):
    """Probe branch x README-name combinations on raw.githubusercontent.com.

    Returns (text, base, saw_network_error). A 404 means "not here"; a timeout or DNS
    failure means "we don't know", and the two must not be conflated -- one is a missing
    README, the other is a bad afternoon on the network.
    """
    saw_network_error = False
    for b in branches:
        if not b:
            continue
        base = f"https://raw.githubusercontent.com/{owner}/{repo}/{b}/"
        for name in README_NAMES:
            try:
                with urlopen(Request(base + name, headers={"User-Agent": "Mozilla/5.0"}), timeout=15) as r:
                    return r.read().decode("utf-8", "replace"), base, saw_network_error
            except HTTPError as e:
                if e.code != 404:
                    saw_network_error = True
            except (URLError, TimeoutError, OSError):
                saw_network_error = True
    return None, None, saw_network_error


def fetch_readme(owner, repo, branch=None):
    """Fetch a repo's README from raw.githubusercontent.com.

    Returns (text, raw_base_url, status). raw_base_url is the directory URL of the README
    on the resolved branch, so relative image paths can be rewritten to absolute.

    status is one of:
      "ok"      -- README found
      "missing" -- the repo has no README on its real default branch
      "error"   -- something transient (network, rate limit); do NOT treat as missing

    Branch resolution guesses BRANCH_GUESSES first, because raw.githubusercontent.com is
    a separate host and does not count against the API rate limit. Only if every guess
    misses do we spend one API call to ask GitHub what the default branch actually is,
    then retry -- so an unusual branch ("develop", "trunk") still resolves, it just costs
    one call. That lazy fallback is what keeps a render inside the 60/hour unauthenticated
    budget, which also pays for one commit-date call per project.

    Distinguishing "missing" from "error" is the point of the status. The pages are
    committed and deployed by rsync, so a transient failure silently rewritten as "this
    project has no README" would ship a confident lie.
    """
    tried = [branch] if branch else list(BRANCH_GUESSES)
    text, base, net_error = _try_branches(owner, repo, tried)
    if text is not None:
        return text, base, "ok"

    # An explicit readme_branch is a deliberate choice; don't second-guess it.
    if not branch:
        default = fetch_default_branch(owner, repo)
        if default and default not in tried:
            text, base, net_error_2 = _try_branches(owner, repo, [default])
            net_error = net_error or net_error_2
            if text is not None:
                return text, base, "ok"
        elif default is None:
            # Couldn't even ask -- can't claim the README is missing.
            net_error = True

    return None, None, ("error" if net_error else "missing")
