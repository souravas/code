# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal coding-interview prep repo: Python 3.14, standard library only. No build system, no
dependencies, no test framework, no linter config. Solutions are worked through by hand and read
back later for review, so clarity of the solution itself is the product.

Two top-level solution trees (245 solution files) plus three standalone documents: `CHEATSHEET.md`
and `PATTERNS.md` (the reference pair), and `README.md`, a short index describing those two.

```
algomonster/  01_sorting(4)   02_binary_search(8)   03_two_pointers(19)
              04_depth_first_search(16)  05_backtracking(16)  06_breadth_first_search(5)
              07_graph(22)  08_heap(8)  09_dynamic_programming(48)
              10_advanced_data_structures(15)  11_miscellaneous(25)
neetcode/     arrays_hashes(9)  backtracking(2)  binary_search(5)  heap(4)
              linked_list(9)  math_geometry(2)  sliding_window(6)  stack(6)
              trees(11)  two_pointers(5)
```

Every directory is fully solved — there are no empty or placeholder files left anywhere in the tree.

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

Seven early files are the exception and **execute on import** via bare top-level statements (not
`__main__` guards): all four of `01_sorting/`, plus `02_binary_search/01_binary_search.py`,
`02_binary_search/07_peak_mountain_array.py`, and `03_two_pointers/16_product_of_array.py`. Leave
those as they are.

Six files in `09_dynamic_programming/` are **comment-only lesson notes** — they carry the syllabus
explanation for a topic that has no coded exercise of its own, so they contain no executable
statements: `07_grid.py`, `25_knapsack_dp.py`, `33_0_1_knapsack.py`, `41_topological_sort_dp.py`,
`46_bitmask.py`, `47_bitmask_dp.py`. They are deliberate, not stubs.

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
- Five files end with an `if __name__ == "__main__":` block that instantiates `Solution()` and prints
  a sample call — `arrays_hashes/{contains_duplicate,group_anagrams,two_sum,valid_anagrams}.py` and
  `backtracking/n_queens.py`. This is the only informal test harness in the repo.
- Design problems are a bare class with the LeetCode-specified name, not a `Solution` wrapper —
  `KthLargest`, `MinStack`, `LRUCache`.

Some files carry a second import mid-file (e.g. `from functools import cache` reappearing before an
`_improved` variant that needs it). That mirrors how the solution was worked through and is left
alone rather than hoisted.

## Code style

- No docstrings anywhere. Comments are rare and reserved for the non-obvious insight behind a step
  (e.g. why the smaller side advances in trapping-rain-water), never for restating the code.
- Built-in generics (`list[int]`, `dict[str, int]`) over `typing.List`; `Optional` is used where a
  LeetCode signature calls for it.
- Memoization defaults to `from functools import cache` (38 files). `lru_cache` is not banned, but
  `09_dynamic_programming/37_interval_dp.py` is now its only live use. Hand-rolled `memo = {}` dicts
  also appear where the point is showing the mechanism, usually as the naive half of a
  naive/`_improved` pair.
- `from math import inf` (9 files) is preferred over `float("inf")` in newer files; both are present.
  Return types are annotated `int | float` where a function can return `inf`.
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

Between them the pair is meant to cover **every technique that appears anywhere in the two solution
trees**, so a newly solved problem that introduces a technique not yet documented should be reflected
there. They are references, not a per-problem index: templates are generalized and named after the
pattern, and a problem only earns its own named subsection when it demonstrates something the
surrounding template does not.

Both files are read on a phone and skimmed under time pressure — keep code blocks self-contained,
keep the prose between them short, and put the reason a step is non-obvious in a comment on that
step rather than in a paragraph above it.

### They are mirrored into an Obsidian vault

`python .claude/scripts/to_vault.py` writes both files into `70 - Learn/Code/` of the Obsidian vault
at `~/Downloads/hope`, adding the frontmatter and turning every markdown link into a wikilink. Only
that residue is done at copy time — everything else here is already written the way the vault's
Obsidian Linter would rewrite it on save, so lint is a no-op. That is why these two files break the
prose conventions the rest of this repo follows, and re-imposing them silently undoes the copy:

- **Prose is one paragraph per line, never hard-wrapped** — unlike this file. The vault's
  `paragraph-blank-lines` rule reads each wrapped line as its own paragraph and puts a blank line
  between them, shattering a sentence into fragments. Leave the long lines alone.
- **Headings are already in the Linter's Title Case**, which is not ordinary Title Case: its
  289-word minor-word list applies anywhere in the line, so `Two Pointers across Two Arrays` and
  `Linear DP — the Stairs Family` are correct and will be reverted if "fixed". A capital you type is
  frozen, so an acronym or a product name is safe as written.
- **No `(`, `)` or `"` in a heading** — the rule miscounts word boundaries after them and
  miscapitalizes the rest of the line. Attach the gloss with an em dash or a comma instead, the way
  `Kadane's Algorithm — Maximum Subarray` and `Unbounded Knapsack — Coin Change II, Number of Ways`
  do. Parens in body text and in code are unaffected.
- A blank line always separates a paragraph from a list that follows it.
- Backticks in a heading are load-bearing where they wrap a lowercase keyword — they are the only
  reason the scope heading in `CHEATSHEET.md` still reads `global` and `nonlocal` rather than
  `Global` and `Nonlocal`. The cost is that a heading containing backticks is a fragile wikilink
  target on the vault side, so prefer a heading without them when the words are not keywords.

Heading text feeds three things at once — the GitHub anchor slug, the `📋 Table of Contents` entry,
and the vault wikilink — so renaming one means fixing every `](#…)` that points at it in both files.

## Git

`.gitattributes` forces LF in the repo and on checkout for `.py`/`.md`/`.json`, on Windows included —
don't let a tool reintroduce CRLF. 44 working-tree files are nonetheless CRLF on disk (DP 41–48,
all of `10_advanced_data_structures/` bar one, and most of `11_miscellaneous/`); git normalizes them
to LF on commit, so `git diff` stays clean and the warning git prints when touching them is benign.
`git ls-files --eol | grep 'w/crlf'` lists them.

`__pycache__/` is gitignored and appears under most `neetcode/` directories plus
`algomonster/10_advanced_data_structures/` — ignore it when counting files.

Commits are one file each, subject-only, present tense, sentence case — capitalize the first word
only: `Add target sum`, `Update word break`, `Rename lowest common ancestor`.
`.claude/commands/commit.md` (the `/commit` command) is the authoritative spec and carries the full
message table. Its rule that solving an empty placeholder counts as `Add` rather than `Update` is
now dormant — no empty placeholders remain — but the file is still the reference for everything else.
