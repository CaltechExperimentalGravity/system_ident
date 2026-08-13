---
name: pull-before-concluding-absent
description: Always git fetch/pull before concluding that a file or feature does not exist
metadata:
  type: feedback
---

Before saying "X does not exist in this repo," run `git fetch` (or `git pull --ff-only`) and
check `git log HEAD..origin/main`. This clone is synced across several machines, so the local
working tree is routinely many commits behind work done elsewhere.

**Why:** on 2026-08-12 I enumerated every local file, concluded "there is no slide deck in this
repo," and built a new one from scratch. The user said "maybe do a git pull and check again" —
local `main` was **7 commits behind** and `origin/main` already had `talks/ligo-sysid.qmd`, a
full reveal.js deck with `talks/figs/`, `make_figs.py`, and `talk.scss`. The wasted work was
entirely avoidable with one fetch.

**How to apply:** fetch first whenever the task is "improve/update/fix the existing X" and X
isn't where expected. A confident absence claim about repo contents requires a fresh fetch
behind it. Related: [[never-search-outside-repo]] — widen in git history, never on the disk.
