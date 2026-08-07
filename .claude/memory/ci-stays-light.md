---
name: ci-stays-light
description: "system_ident CI must stay light — heavy/slow tests and renders run locally, not on every push"
metadata: 
  node_type: memory
  type: project
  modified: 2026-08-05T19:35:17.915Z
---

For `system_ident`, CI is deliberately light: one Python, no docs stack in the test
job, no coverage upload, `pytest -m "not slow"`. Heavy work runs locally — the full
suite (`pytest` with no marker filter) and any expensive example render.

**Why:** project decision (2026-08-05), taken while adding the first pytest job to
CI: CI stays light, heavy tests run only locally. Pushes should stay fast enough to
not be a bottleneck; the exhaustive check is the developer's job before pushing.

**How to apply:** when a test or docs page gets slow enough to dominate a CI job,
mark it `@pytest.mark.slow` (or `freeze: true` for an example) rather than letting
CI absorb it — but only when something else already covers that path in CI, and say
so in the marker's comment. Before proposing any CI speedup, measure the per-step
timings first (`gh api .../actions/jobs/<id>`): on this repo the install steps were
4–38 s while the real costs were `quarto render` and `pytest`, so dependency or
conda-env caching would have bought nothing.
