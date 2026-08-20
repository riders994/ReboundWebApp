#!/usr/bin/env python3
"""Manage the portfolio's projects framework.

projects.json (site/assets/data/projects.json) is the source of truth. Each project is
git-based: it has a GitHub repo, an optional web-app link, a status (in-flight/completed),
and an optional "featured on" date. The projects page shows a highlight of featured +
in-flight projects, then a paginated list of completed ones (newest first).

Each project's detail page writeup is its repo's README, rendered to HTML at build time —
unless you override it by dropping content/projects/<slug>.md in THIS repo (which also lets
a web-app project keep a fully custom, hand-authored page: set its page to "custom").

Usage:
  projects.py add rebounding --name "NBA Rebound Predictor" --repo https://github.com/you/repo \
      --blurb "..." --tags Python D3.js --webapp rebounding.html --status in-flight --page custom
  projects.py feature   <slug>            # designate featured (stamps today's date)
  projects.py unfeature <slug>
  projects.py status    <slug> completed  # or in-flight; completing stamps completed_at
  projects.py list
  projects.py render                      # fetch READMEs / overrides -> generate detail pages

"featured" = the 5 projects most recently designated (by their featured date).
"""
import argparse
import html
import re
import sys

import projects_data as pd


# --------------------------------------------------------------------------- #
# registry commands
# --------------------------------------------------------------------------- #
def cmd_add(args):
    if not pd.SLUG_RE.match(args.slug):
        sys.exit(f"invalid slug '{args.slug}': lowercase letters, digits, hyphens")
    projects = pd.load_projects()
    if pd.find(projects, args.slug) >= 0:
        sys.exit(f"project '{args.slug}' already exists")
    if args.status not in ("in-flight", "completed"):
        sys.exit("status must be 'in-flight' or 'completed'")
    if args.page not in ("generated", "custom"):
        sys.exit("page must be 'generated' or 'custom'")
    project = {
        "slug": args.slug,
        "name": args.name,
        "blurb": args.blurb or "",
        "repo": args.repo,
        "webapp": args.webapp,
        "tags": args.tags or [],
        "status": args.status,
        "featured_at": None,
        "completed_at": pd.today() if args.status == "completed" else None,
        "page": args.page,
        "readme_branch": args.branch,
    }
    projects.append(project)
    pd.save_projects(projects)
    print(f"added {args.slug} [{args.status}, page={args.page}]")


def _get(projects, slug):
    idx = pd.find(projects, slug)
    if idx < 0:
        sys.exit(f"no project '{slug}'")
    return idx


def cmd_feature(args):
    projects = pd.load_projects()
    projects[_get(projects, args.slug)]["featured_at"] = pd.today()
    pd.save_projects(projects)
    featured = pd.featured_slugs(projects)
    print(f"featured {args.slug} (currently featured: {', '.join(sorted(featured))})")


def cmd_unfeature(args):
    projects = pd.load_projects()
    projects[_get(projects, args.slug)]["featured_at"] = None
    pd.save_projects(projects)
    print(f"unfeatured {args.slug}")


def cmd_status(args):
    if args.value not in ("in-flight", "completed"):
        sys.exit("status must be 'in-flight' or 'completed'")
    projects = pd.load_projects()
    p = projects[_get(projects, args.slug)]
    p["status"] = args.value
    if args.value == "completed" and not p.get("completed_at"):
        p["completed_at"] = pd.today()
    pd.save_projects(projects)
    print(f"{args.slug} -> {args.value}" + (f" (completed {p['completed_at']})" if args.value == "completed" else ""))


def cmd_list(args):
    projects = pd.load_projects()
    if not projects:
        print("(no projects)")
        return
    featured = pd.featured_slugs(projects)
    for p in projects:
        marks = []
        if p["slug"] in featured:
            marks.append("★featured")
        marks.append(p["status"])
        if p.get("completed_at"):
            marks.append(f"done {p['completed_at']}")
        print(f"{p['slug']:<20} {p['name']}")
        print(f"    {', '.join(marks)}  |  page={p.get('page')}  repo={p.get('repo')}")


# --------------------------------------------------------------------------- #
# render: build detail pages from READMEs / overrides
# --------------------------------------------------------------------------- #
NAV = """  <nav class="nav">
    <div class="nav__inner">
      <a class="nav__brand" href="../index.html">Rohan Vahalia</a>
      <button class="nav__toggle" aria-label="Menu" aria-expanded="false">☰</button>
      <ul class="nav__links">
        <li><a href="../index.html#about">About</a></li>
        <li><a href="index.html" aria-current="page">Projects</a></li>
        <li><a href="../resume.html">Resume</a></li>
        <li><a href="../blog/index.html">Blog</a></li>
        <li><a href="../comedy.html">Comedy</a></li>
      </ul>
    </div>
  </nav>"""

FOOTER = """  <footer class="footer">
    <div class="wrap footer__inner">
      <span class="muted">© <span data-year>2026</span> Rohan Vahalia</span>
      <ul class="footer__social">
        <li><a href="mailto:r.vahalia@gmail.com">Email</a></li>
        <li><a href="https://github.com/riders994">GitHub</a></li>
        <li><a href="https://www.linkedin.com/in/rvahalia">LinkedIn</a></li>
      </ul>
    </div>
  </footer>"""

STATUS_LABEL = {"in-flight": "In flight", "completed": "Completed"}


def _rewrite_relative(html_text, base):
    """Point relative <img>/<a href> targets at the repo's raw/blob URLs so they resolve."""
    if not base:
        return html_text
    html_text = re.sub(r'(<img[^>]+src=")(?!https?://|data:)([^"]+)"', r'\1' + base + r'\2"', html_text)
    return html_text


def _render_body(project):
    """Return (html_body, source) for a project's writeup."""
    override = pd.OVERRIDE_DIR / f"{project['slug']}.md"
    if override.exists():
        return pd.render_markdown(override.read_text()), "override"
    owner, repo = pd.parse_repo(project.get("repo", ""))
    if not owner:
        return "<p><em>No repo configured and no local override.</em></p>", "empty"
    text, base = pd.fetch_readme(owner, repo, project.get("readme_branch"))
    if text is None:
        return f"<p><em>Could not fetch README from {html.escape(project['repo'])}.</em></p>", "fetch-failed"
    return _rewrite_relative(pd.render_markdown(text), base), "readme"


def _detail_html(project, body):
    name = html.escape(project["name"])
    blurb = html.escape(project.get("blurb", ""))
    status = project.get("status", "in-flight")
    badge = STATUS_LABEL.get(status, status)
    tags = "".join(f'<li class="tag">{html.escape(t)}</li>' for t in project.get("tags", []))
    links = f'<a class="btn btn--ghost" href="{html.escape(project["repo"])}" target="_blank" rel="noopener">View on GitHub →</a>'
    if project.get("webapp"):
        links += f'\n        <a class="btn btn--primary" href="{html.escape(project["webapp"])}">Open the app →</a>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{name} — Rohan Vahalia</title>
  <meta name="description" content="{blurb}">
  <link rel="stylesheet" href="../assets/css/main.css">
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🛠️</text></svg>">
</head>
<body>
  <!-- GENERATED by scripts/projects.py render — edit content/projects/{project['slug']}.md to override. -->
{NAV}

  <header class="wrap section" style="padding-bottom:1rem">
    <p class="eyebrow">Project · <span class="badge badge--{status}">{badge}</span></p>
    <h1>{name}</h1>
    <p class="lead">{blurb}</p>
    <ul class="tags" style="margin-top:1rem">{tags}</ul>
    <div class="project-links" style="margin-top:1.4rem">
        {links}
    </div>
  </header>

  <article class="wrap section prose">
{body}
  </article>

{FOOTER}

  <script src="../assets/js/main.js"></script>
</body>
</html>
"""


def cmd_render(args):
    projects = pd.load_projects()
    generated = [p for p in projects if p.get("page", "generated") == "generated"]
    if not generated:
        print("no 'generated' projects to render (custom pages are hand-authored)")
        return
    try:
        import markdown  # noqa: F401  (used by projects_data.render_markdown)
    except ImportError:
        sys.exit("markdown not installed — run: pip install -r scripts/requirements.txt")
    pd.DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    for p in generated:
        body, source = _render_body(p)
        # indent body under <article>
        body = "\n".join("    " + line if line.strip() else line for line in body.splitlines())
        out = pd.DETAIL_DIR / f"{p['slug']}.html"
        out.write_text(_detail_html(p, body))
        print(f"{p['slug']}: {out.relative_to(pd.ROOT)} ({source})")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Manage the projects framework.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="add a project")
    a.add_argument("slug")
    a.add_argument("--name", required=True)
    a.add_argument("--repo", required=True, help="GitHub repo URL")
    a.add_argument("--blurb", default="")
    a.add_argument("--tags", nargs="*", default=[])
    a.add_argument("--webapp", default=None, help="link to a live web-app interface")
    a.add_argument("--status", default="in-flight", help="in-flight | completed")
    a.add_argument("--page", default="generated", help="generated | custom")
    a.add_argument("--branch", default=None, help="README branch override")
    a.set_defaults(func=cmd_add)

    f = sub.add_parser("feature", help="designate a project featured (stamps today)")
    f.add_argument("slug")
    f.set_defaults(func=cmd_feature)

    uf = sub.add_parser("unfeature", help="remove the featured designation")
    uf.add_argument("slug")
    uf.set_defaults(func=cmd_unfeature)

    st = sub.add_parser("status", help="set a project's status")
    st.add_argument("slug")
    st.add_argument("value", help="in-flight | completed")
    st.set_defaults(func=cmd_status)

    sub.add_parser("list", help="list projects").set_defaults(func=cmd_list)
    sub.add_parser("render", help="generate detail pages from READMEs/overrides").set_defaults(func=cmd_render)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
