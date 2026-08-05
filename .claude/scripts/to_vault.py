"""Copy CHEATSHEET.md and PATTERNS.md into the Obsidian vault as conforming notes.

The two documents are written to be lint-stable in the vault already — headings are
pre-set to the Title Case the Obsidian Linter applies, they carry no ( ) or " that
capitalize-headings would miscount, and prose is unwrapped so paragraph-blank-lines
leaves it alone. Everything left over is syntax GitHub and Obsidian cannot share:

  - YAML frontmatter, which the vault requires and a repo doc should not carry
  - [PATTERNS.md](PATTERNS.md)      -> [[Python Code Patterns]]
  - [DP](#dynamic-programming)      -> [[#Dynamic Programming|DP]]

Frontmatter already on the target note is preserved, so hand-edited keys such as
rating: survive a re-copy. Code blocks are passed through untouched.

Usage:  python .claude/scripts/to_vault.py [--vault PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_VAULT = Path.home() / "Downloads" / "hope"
NOTE_DIR = "70 - Learn/Code"

NOTES = {
    "CHEATSHEET.md": "Python Code Cheatsheet",
    "PATTERNS.md": "Python Code Patterns",
}

DEFAULT_FRONTMATTER = """---
author: sourav
tags: [learn/code]
type: article
status: completed
difficulty: beginner
rating:
completed:
url: https://github.com/souravas/code
created: {today}
---"""

FENCE = re.compile(r"^\s*```")
HEADING = re.compile(r"^#{1,6} (.*)$")
# github-slugger: lowercase, drop punctuation and symbols, spaces to hyphens
SLUG_STRIP = re.compile(r"[ -⁯⸀-⹿\\'!\"#$%&()*+,./:;<=>?@\[\]^`{|}~·‐-―]")


def slug(heading: str, seen: dict[str, int]) -> str:
    s = SLUG_STRIP.sub("", heading.strip().lower()).replace(" ", "-")
    n = seen.get(s, 0)
    seen[s] = n + 1
    return f"{s}-{n}" if n else s


def heading_by_slug(text: str) -> dict[str, str]:
    """Map every GitHub anchor in a document back to its heading text."""
    out: dict[str, str] = {}
    seen: dict[str, int] = {}
    infence = False
    for line in text.split("\n"):
        if FENCE.match(line):
            infence = not infence
            continue
        if infence:
            continue
        m = HEADING.match(line)
        if m:
            out[slug(m.group(1), seen)] = m.group(1)
    return out


def wikilink(target: str, label: str, drop_alias: bool) -> str:
    if drop_alias or label == target.lstrip("#").split("#")[-1]:
        return f"[[{target}]]"
    return f"[[{target}|{label}]]"


def convert_links(text: str, own: str, anchors: dict[str, dict[str, str]]) -> str:
    """Rewrite markdown links to wikilinks, leaving fenced code untouched."""

    def one(m: re.Match[str]) -> str:
        label, dest = m.group(1), m.group(2)
        file, _, anchor = dest.partition("#")
        source = file or own
        if source not in NOTES:
            return m.group(0)
        note = NOTES[source]
        # "CHEATSHEET.md" / "CHEATSHEET.md -> Lists" as link text is a repo-ism
        drop = label.startswith(source)
        if not anchor:
            return wikilink(note, label, drop)
        head = anchors[source].get(anchor)
        if head is None:
            print(f"  ! unresolved anchor {dest!r}", file=sys.stderr)
            return m.group(0)
        target = f"#{head}" if not file else f"{note}#{head}"
        return wikilink(target, label, drop)

    out, infence = [], False
    for line in text.split("\n"):
        if FENCE.match(line):
            infence = not infence
            out.append(line)
            continue
        out.append(line if infence else re.sub(r"\[([^\]\[]*)\]\(([^)]+)\)", one, line))
    return "\n".join(out)


def existing_frontmatter(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n.*?\n---", text, re.S)
    return m.group(0) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dest_dir = args.vault / NOTE_DIR
    if not dest_dir.is_dir():
        print(f"no such folder: {dest_dir}", file=sys.stderr)
        return 1

    sources = {name: (REPO / name).read_text(encoding="utf-8") for name in NOTES}
    anchors = {name: heading_by_slug(text) for name, text in sources.items()}

    from datetime import date

    for name, note in NOTES.items():
        target = dest_dir / f"{note}.md"
        front = existing_frontmatter(target) or DEFAULT_FRONTMATTER.format(
            today=date.today().isoformat()
        )
        body = convert_links(sources[name], name, anchors).strip("\n")
        out = f"{front}\n\n{body}\n"
        if args.dry_run:
            print(f"{target}: {len(out.splitlines())} lines "
                  f"({'existing' if existing_frontmatter(target) else 'default'} frontmatter)")
            continue
        target.write_text(out, encoding="utf-8", newline="\n")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
