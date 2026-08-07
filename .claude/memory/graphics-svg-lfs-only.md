---
name: graphics-svg-lfs-only
description: "HARD RULE — every plot is SVG, every committed graphic lives in Git LFS"
metadata: 
  node_type: memory
  type: feedback
---

**HARD RULE (user, 2026-06-18):** every plot/figure committed to this repo is **SVG**
(vector), and **no graphics live outside Git LFS** unless absolutely necessary.

**Why:** SVG is the vector format the user wants for all plots; binary graphics bloat
the git history, so they belong in LFS, not inline in the repo.

**How to apply:**
- Generate plots as `.svg`, not `.png`/`.jpg`. matplotlib: `savefig(..., format="svg")`.
  plotly: `write_image(..., format="svg")` (needs `kaleido`) or export the figure to SVG.
- Track graphics with Git LFS: `.gitattributes` with
  `*.svg filter=lfs diff=lfs merge=lfs -text` (and the same for any `*.png`/`*.jpg` that
  are *absolutely necessary* — e.g. an Open-Graph social card, which social platforms
  won't render as SVG). Run `git lfs install` once per clone.
- When adding or regenerating any figure, default to SVG+LFS; never commit a new PNG
  plot. If a raster format is genuinely unavoidable, call it out and still put it in LFS.
- Existing PNGs (docs/examples/thumbnails/*.png, docs/assets/og-card.png) predate this
  rule — migrate them to SVG+LFS when touched. Related: [[stay-on-pintelon-schoukens]].
