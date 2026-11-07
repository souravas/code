# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal coding-interview prep repo: Python 3.14, standard library only. No build system, no
dependencies, no test framework, no linter config. Solutions are worked through by hand and read
back later for review, so clarity of the solution itself is the product.

Two solution trees, every directory fully solved (no empty or placeholder files):

```
algomonster/  01_sorting  02_binary_search  03_two_pointers  04_depth_first_search
              05_backtracking  06_breadth_first_search  07_graph  08_heap
              09_dynamic_programming  10_advanced_data_structures  11_miscellaneous
neetcode/     arrays_hashes  backtracking  binary_search  heap  linked_list
              math_geometry  sliding_window  stack  trees  two_pointers
```

`README.md`, the GitHub landing page, only links to <https://souravas.com/code>, which holds the
reference documents and the overview of this repo; see [The reference documents](#the-reference-documents).

## Running code

`neetcode/` files import shared node classes by absolute package path
(`from neetcode.trees.tree_node import TreeNode`). There are no `__init__.py` files — this works
only via implicit namespace packages with the repo root on `sys.path`, so run them as modules from
the repo root:

```
python -m neetcode.trees.invert_tree        # works
python neetcode/trees/invert_tree.py        # ModuleNotFoundError
```

`algomonster/` files have no cross-file imports — each defines whatever `Node`/`ListNode` it needs
inline. Most define functions only, so importing them does nothing; to exercise one, import it in a
throwaway script or `python -c`. Do not add demo blocks to algomonster files that lack them.

Leave these exceptions as they are:

- **Seven files execute on import** via bare top-level statements (not `__main__` guards), and all
  seven print: all four of `01_sorting/`, plus `02_binary_search/01_binary_search.py`,
  `02_binary_search/07_peak_mountain_array.py` and `03_two_pointers/16_product_of_array.py`.
- **Two files raise the recursion limit on import**: `09_dynamic_programming/24_divisor_game.py` and
  `32_perfect_squares.py` call `sys.setrecursionlimit` at module level.
- **Six files in `09_dynamic_programming/` are comment-only lesson notes**, carrying the syllabus
  explanation for a topic with no coded exercise of its own: `07_grid.py`, `25_knapsack_dp.py`,
  `33_0_1_knapsack.py`, `41_topological_sort_dp.py`, `46_bitmask.py`, `47_bitmask_dp.py`. They are
  deliberate, not stubs.

## The two trees have deliberately different conventions

Match the convention of whichever tree you are editing — do not unify them.

**`algomonster/NN_topic/NN_problem.py`** — follows the AlgoMonster course syllabus; the numeric
prefixes are curriculum order and are load-bearing for navigation. Two names repeat inside
`04_depth_first_search/` because the syllabus revisits them — `09`/`14_reconstruct_binary_tree.py`
and `13`/`16_lowest_common_ancestor.py` (13 is the BST-ordering variant, 16 the general binary
tree). Match on the numeric prefix, not the name.

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
  `groupAnagrams2`), or a plain helper method where the split is by role (`evalRPN` / `resolve`).
- Design problems are a bare class with the LeetCode-specified name, not a `Solution` wrapper —
  `KthLargest`, `MinStack`, `LRUCache`.
- Five files end with an `if __name__ == "__main__":` block that prints a sample call —
  `arrays_hashes/{contains_duplicate,group_anagrams,two_sum,valid_anagrams}.py` and
  `backtracking/n_queens.py`. This is the only informal test harness in the repo.

Some files carry a second import mid-file (e.g. `from functools import cache` reappearing before an
`_improved` variant that needs it). That mirrors how the solution was worked through and is left
alone rather than hoisted.

## Code style

- No docstrings anywhere. Comments are rare and reserved for the non-obvious insight behind a step
  (e.g. why the smaller side advances in trapping-rain-water), never for restating the code.
- Built-in generics (`list[int]`, `dict[str, int]`) over `typing.List`, though a few older files
  still import it; `Optional` is used where a LeetCode signature calls for it.
- Memoization defaults to `from functools import cache`; `lru_cache` appears but isn't the default.
  Hand-rolled `memo = {}` dicts appear where the point is showing the mechanism, usually as the
  naive half of a naive/`_improved` pair.
- `from math import inf` is preferred over `float("inf")` in newer files; both are present. Return
  types are annotated `int | float` where a function can return `inf`.
- The whole tree is black-clean under default settings (88 cols, double quotes, trailing commas,
  4-space indent) — keep it that way. Black is neither on `PATH` nor a project dependency; the VS Code
  extension bundles the only copy, which runs fine against Python 3.14:

  ```
  $env:PYTHONPATH = (Get-Item "$env:USERPROFILE\.vscode\extensions\ms-python.black-formatter-*\bundled\libs").FullName
  python -m black .
  ```

## The reference documents

They don't live in this repo. `CHEATSHEET.md` (**syntax only** — one-liners, stdlib APIs, idioms,
no algorithms), `PATTERNS.md` (**algorithm templates** — full adaptable implementations) and the
overview of this repo are one page, <https://souravas.com/code>. The only copy is `code/` in the site
repo (`souravas/souravas.github.io`, checked out at `../souravas.github.io`), and that repo's
`CLAUDE.md` has the rules for editing them. The overview, `code/README.md` there, lists each
directory's file count and the exceptions above, so update it when solutions are added, removed or
moved.

Between them the pair is meant to cover **every technique that appears anywhere in the two solution
trees**, so a newly solved problem that introduces a technique not yet documented should be reflected
there. They are references, not a per-problem index: templates are generalized and named after the
pattern, and a problem only earns its own named subsection when it demonstrates something the
surrounding template does not.

## Git

`.gitattributes` forces LF in the repo and on checkout for `.py`/`.md`/`.json`, on Windows included —
don't let a tool reintroduce CRLF. The index is all LF, but a working tree written before
`.gitattributes` existed (like the original Windows checkout) can still hold CRLF files. Git
normalizes them on commit, so the line-ending warning for those is benign.
`git ls-files --eol | grep 'w/crlf'` lists them; deleting one and restoring it with
`git checkout -- <path>` rewrites it as LF.

`__pycache__/` is gitignored; ignore it when counting files.

Commits are one file each, subject-only, present tense, sentence case — `Add target sum`,
`Update word break`, `Rename lowest common ancestor` — with no trailers. `.claude/commands/commit.md`
(the `/commit` command) is the authoritative spec.
