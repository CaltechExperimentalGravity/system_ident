---
name: no-heavy-compute-in-slide-generation
description: "HARD RULE — build decks from docs/_freeze and existing caches; cache-first generators, never recompute what already ran"
metadata:
  node_type: memory
  type: feedback
---

# HARD RULE: never run heavy computation to build slides

Slide/deck generation is a **presentation** job, not a measurement job. Do **not** launch
campaigns, twin runs, or Monte-Carlo sweeps to produce a figure for a talk.

**Reuse what the repo already computed**, in this order:
1. `docs/_freeze/examples/*/execute-results/html.json` — executed pages, committed. The
   Plotly figure JSON is in there; rebuild an SVG from it in seconds
   (`plotly.io.from_json`) and parse the numbers out of the same file. This is how
   `talks/make_figs.py` gets the DARM drift panels instead of re-driving a ~20 min campaign.
2. Existing caches / artifacts: `experiments/rtsfreerun/srm_campaign_cache_*.npz`,
   `talks/figs/sos_recovery.json`, an experiment's own log or saved SVG.
3. Only if none exists: compute **once**, write the result to a cache next to the figure,
   and guard the code so a re-run is a no-op.

**Cache-first is not optional.** Every generator group must early-out when its figures AND
its numbers are already on disk — *before* importing anything expensive (importing
`srm_modal_demo` alone builds the compiled `x1hsts6dof` model). A full regeneration with
nothing changed must be seconds. Only an explicit `--force` recomputes.

**Never recompute something you already ran this session.** (Cost of learning this: the
compiled-twin HSTS rung was run twice, ~20 min wasted, plus a ~20 min DARM drift campaign
started for results that were already frozen in the repo.)

Rendering the deck itself must execute **no** code — numbers reach the slides through a
generated `_variables.yml` + `{{< var … >}}`, never a Jupyter kernel.
