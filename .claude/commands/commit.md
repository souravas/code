---
description: Commit all pending changes, one commit per file, with Add/Update/Remove messages
argument-hint: "[path]... (optional — defaults to all pending changes)"
allowed-tools: Bash(git status:*), Bash(git add:*), Bash(git commit:*), Bash(git show:*), Bash(git log:*), Bash(git diff:*)
---

Pending changes:

!`git status --porcelain`

Commit each of these files, one commit per file. If `$ARGUMENTS` is non-empty, limit the run to
those paths. If there are no changes, say so and stop.

## Steps

For each changed file, work out the message (below), then commit only that file:
`git add <path>` then `git commit -m "<message>" -- <path>`. For a rename, stage and commit both the
old and the new path together. When done, show `git log --oneline` for the new commits.

## Message format

Present tense, sentence case (capitalize the first word only), no scope prefix, no trailing period.

`<name>` is the filename with the `NN_` ordering prefix and the extension stripped, underscores
turned into spaces: `09_dynamic_programming/28_target_sum.py` → `target sum`.

| Change | Message | Example |
| --- | --- | --- |
| New file | `Add <name>` | `Add reconstruct binary tree` |
| **Empty placeholder now solved** | `Add <name>` | `Add target sum` |
| Modified file that already had code | `Update <name>` | `Update num ways to decode` |
| Deleted file | `Remove <name>` | `Remove trie` |
| Renumbered (same base name) | `Rename <name>` | `Rename lowest common ancestor` |
| Renamed to a different name | `Rename <old> to <new>` | `Rename max area to container water` |

**The placeholder rule matters here.** Around 50 `algomonster` files are committed as empty stubs, so
solving one shows up in `git status` as `M`, not `??` — but it is an `Add`. Before writing `Update`
for a modified `.py`, check what was there before:

```
git show HEAD:<path> | wc -c     # 0 → it was an empty placeholder → "Add <name>"
```

For files that aren't solutions, use the document's own name rather than the derived one:
`Update Cheatsheet`, `Update Patterns`, `Update README`, `Update CLAUDE.md`, `Update commit command`.

## Rules

- One file per commit — never group. The exception is a mechanical change that sweeps the whole repo
  (line-ending normalization, a reformat): that is one commit, e.g. `Normalize line endings to LF`,
  and confirm with the user before making it.
- Never add a `Co-Authored-By` trailer or any other trailer.
- Commit only. Do not push unless explicitly asked.
- Never force-add gitignored files (`__pycache__/`, `.venv/`, `desktop.ini`).
- `.gitattributes` stores everything as LF. If git warns about line endings, fix the file — do not
  change the git config to silence it.
