#!/usr/bin/env python3
"""Formal registration process for comedy meta-tags.

The single sanctioned way to ADD, RETIRE, or inspect entries in the comedy tag registry
(site/assets/data/comedy-tags.json). Applying tags to a video is a separate, operational
step handled by comedy-tags.py.

A registry entry has:
  code      your short meta-tag: lowercase [a-z0-9-], <= 24 chars   (e.g. "kkj")
  label     the reader-facing normalized name                       (e.g. "Knock-Knock Joke")
  category  one of the registry's known categories                  (e.g. joke)

`add` validates the code, refuses to silently overwrite an existing one, and only accepts a
known category unless you explicitly create it — so the vocabulary stays clean. Run with a
terminal and it will prompt for anything you omit.

Usage:
  register-tag.py add kkj --label "Knock-Knock Joke" --category joke
  register-tag.py add kkj                       # interactive: prompts for label/category
  register-tag.py add pun --label "Pun" --category joke --new-category
  register-tag.py list
  register-tag.py categories
  register-tag.py add-category joke
  register-tag.py retire kkj [--strip]          # remove; --strip also pulls it off videos
"""
import argparse
import sys

import comedy_data as cd


def _prompt(message):
    if not sys.stdin.isatty():
        return None
    try:
        return input(message).strip()
    except EOFError:
        return None


def cmd_add(args):
    reg = cd.load_registry()
    code = args.code

    if not cd.CODE_RE.match(code):
        sys.exit(f"invalid code '{code}': use lowercase letters, digits and hyphens "
                 f"(start alphanumeric, max 24 chars), e.g. 'kkj'")

    label = args.label or _prompt(f"Normalized label for '{code}': ")
    if not label:
        sys.exit("a --label is required (the reader-facing name)")

    known = ", ".join(reg["categories"]) or "none yet"
    category = args.category or _prompt(f"Category [{known}]: ")
    if not category:
        sys.exit("a --category is required")

    if code in reg["tags"] and not args.force:
        cur = reg["tags"][code]
        sys.exit(f"'{code}' already exists -> \"{cur['label']}\" [{cur['category']}]; "
                 f"pass --force to replace it")

    if category not in reg["categories"]:
        create = args.new_category
        if not create:
            ans = _prompt(f"category '{category}' is new — create it? [y/N]: ")
            create = bool(ans) and ans.lower().startswith("y")
        if not create:
            sys.exit(f"unknown category '{category}'. Known: {', '.join(reg['categories']) or '(none)'}. "
                     f"Use --new-category, or: register-tag.py add-category {category}")
        reg["categories"].append(category)
        print(f"(created new category '{category}')")

    verb = "updated" if code in reg["tags"] else "registered"
    reg["tags"][code] = {"label": label, "category": category}
    cd.save_registry(reg)
    print(f"{verb} {code} -> \"{label}\" [{category}]")


def cmd_add_category(args):
    reg = cd.load_registry()
    if args.name in reg["categories"]:
        print(f"category '{args.name}' already exists")
        return
    reg["categories"].append(args.name)
    cd.save_registry(reg)
    print(f"added category '{args.name}'")


def cmd_categories(args):
    reg = cd.load_registry()
    print(", ".join(reg["categories"]) or "(none)")


def cmd_list(args):
    reg = cd.load_registry()
    if not reg["tags"]:
        print("(registry empty)")
        return
    for cat in reg["categories"]:
        codes = sorted(c for c, t in reg["tags"].items() if t["category"] == cat)
        if not codes:
            continue
        print(f"\n{cat}:")
        for code in codes:
            print(f"  {code:<14} {reg['tags'][code]['label']}")


def cmd_retire(args):
    reg = cd.load_registry()
    if args.code not in reg["tags"]:
        sys.exit(f"'{args.code}' is not registered")
    del reg["tags"][args.code]
    used = {t["category"] for t in reg["tags"].values()}
    reg["categories"] = [c for c in reg["categories"] if c in used]
    cd.save_registry(reg)
    print(f"retired {args.code}")
    if args.strip:
        n = cd.strip_code_from_clips(args.code)
        print(f"stripped {args.code} from {n} video(s)")


def main():
    p = argparse.ArgumentParser(description="Formal registration for comedy meta-tags.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="register a new meta-tag (validated)")
    a.add_argument("code")
    a.add_argument("--label", help="reader-facing normalized label")
    a.add_argument("--category", help="e.g. location, date, subject, joke")
    a.add_argument("--new-category", action="store_true", help="allow creating the category if it's new")
    a.add_argument("--force", action="store_true", help="overwrite an existing code")
    a.set_defaults(func=cmd_add)

    ac = sub.add_parser("add-category", help="create a category")
    ac.add_argument("name")
    ac.set_defaults(func=cmd_add_category)

    sub.add_parser("categories", help="list categories").set_defaults(func=cmd_categories)
    sub.add_parser("list", help="list the registry").set_defaults(func=cmd_list)

    r = sub.add_parser("retire", help="remove a meta-tag from the registry")
    r.add_argument("code")
    r.add_argument("--strip", action="store_true", help="also remove it from all videos")
    r.set_defaults(func=cmd_retire)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
