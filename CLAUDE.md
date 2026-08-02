# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal coding-interview prep repo: Python 3.14, standard library only. No build system, no
dependencies, no test framework, no linter config. Solutions are worked through by hand and read
back later for review, so clarity of the solution itself is the product.

Two top-level solution trees plus three standalone documents: `CHEATSHEET.md` and `PATTERNS.md` (the
reference pair), and `README.md`, a short index describing those two.

```
algomonster/  01_sorting  02_binary_search  03_two_pointers  04_depth_first_search
              05_backtracking  06_breadth_first_search  07_graph  08_heap
              09_dynamic_programming  10_advanced_data_structures  11_miscellaneous
neetcode/     arrays_hashes  backtracking  binary_search  heap  linked_list
              math_geometry  sliding_window  stack  trees  two_pointers
```

## Running code

`neetcode/` files import shared node classes by absolute package path
(`from neetcode.trees.tree_node import TreeNode`). There are no `__init__.py` files — this works
only via implicit namespace packages with the repo root on `sys.path`, so run them as modules from
the repo root:

```
python -m neetcode.trees.invert_tree        # works
python neetcode/trees/invert_tree.py        # ModuleNotFoundError
```

`algomonster/` files have no cross-file imports (each defines whatever `Node`/`ListNode` it needs
inline) and no `__main__` blocks — nothing runs on import. To exercise one, import it in a throwaway
script or `python -c`; do not add demo blocks to existing algomonster files.

## The two trees have deliberately different conventions

Match the convention of whichever tree you are editing — do not unify them.

**`algomonster/NN_topic/NN_problem.py`** — follows the AlgoMonster course syllabus; the numeric
prefixes are curriculum order and are load-bearing for navigation.

- Module-level `snake_case` functions with type hints (`def rob(nums: list[int]) -> int:`).
- Node classes (`class Node`, `class ListNode`) are declared inline in each file that needs one.
- A second approach for the same problem goes in a sibling function suffixed `_improved` or
  `_optimized` (e.g. `rob` / `rob_improved`), keeping the naive version visible for comparison.
- No `class Solution`, no `__main__` block.

**`neetcode/topic/problem.py`** — mirrors LeetCode submissions.

- `class Solution` with `camelCase` methods matching the exact LeetCode signature.
- Shared `TreeNode` (`neetcode/trees/tree_node.py`) and `ListNode` (`neetcode/linked_list/list_node.py`)
  are imported, not redefined.
- Multiple approaches become numbered method variants on the same class (`groupAnagrams1`,
  `groupAnagrams2`).
- Some files end with an `if __name__ == "__main__":` block that instantiates `Solution()` and prints
  a sample call — this is the only informal test harness in the repo.

`algomonster/11_miscellaneous/` is the only directory still holding **intentionally empty
placeholders** — 23 of its 26 files are 0 bytes (all but `01_intervals.py`, `02_merge_intervals.py`,
`03_insert_interval.py`). Those filenames were pre-created from the syllabus and not yet solved; they
are not broken or truncated, so leave them alone unless asked to solve one. Every other directory is
fully solved, including `09_dynamic_programming/` (48 files) and `10_advanced_data_structures/` (15).

## Code style

- No docstrings anywhere. Comments are rare and reserved for the non-obvious insight behind a step
  (e.g. why the smaller side advances in trapping-rain-water), never for restating the code.
- Built-in generics (`list[int]`, `dict[str, int]`) over `typing.List`; `Optional` is used where a
  LeetCode signature calls for it.
- Memoization defaults to `from functools import cache`. `lru_cache` is not banned — it survives in
  `09_dynamic_programming/37_interval_dp.py` and in the PATTERNS.md memoization template. Hand-rolled
  `memo = {}` dicts also appear where the point is showing the mechanism.
- The whole tree is black-clean under default settings (88 cols, double quotes, trailing commas,
  4-space indent) — keep it that way. Black is neither on `PATH` nor a project dependency; the VS Code
  extension bundles the only copy, which runs fine against Python 3.14:

  ```
  $env:PYTHONPATH = (Get-Item "$env:USERPROFILE\.vscode\extensions\ms-python.black-formatter-*\bundled\libs").FullName
  python -m black .
  ```

## The reference documents

`CHEATSHEET.md` is **syntax only** — one-liners, stdlib APIs, idioms, no algorithms. `PATTERNS.md` is
**algorithm templates** — full adaptable implementations. New material belongs on one side of that
line; when adding a section to either, update its `📋 Table of Contents` anchor list at the top.

## Git

`.gitattributes` forces LF in the repo and on checkout for `.py`/`.md`/`.json`, on Windows included —
don't let a tool reintroduce CRLF. 22 working-tree files (DP 41–48 and most of
`10_advanced_data_structures/`) are nonetheless CRLF on disk; git normalizes them to LF on commit, so
`git diff` stays clean and the warning git prints when touching them is benign.

Commits are one file each, subject-only, present tense, sentence case — capitalize the first word
only: `Add target sum`, `Update word break`, `Rename lowest common ancestor`.
`.claude/commands/commit.md` (the `/commit` command) is the authoritative spec: it carries the full
message table and the rule that solving an empty placeholder is `Add`, not `Update`, even though git
reports it as `M`.
