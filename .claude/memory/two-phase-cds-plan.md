---
name: two-phase-cds-plan
description: CDS work is two phases; we are on Phase 1 (RTSfreerun). Do NOT discuss Phase 2 (real hardware) until explicitly told.
metadata: 
  node_type: memory
  type: project
---

The CDS effort has **two phases**:

- **Phase 1 — RTSfreerun** (the compiled-CDS digital twin: `x1hsts` / `x1hsts6dof`,
  `backends/rtsfreerun_adapter.py`, example 07, the A1→A4 rtsfreerun spec). **This is the
  current focus.**
- **Phase 2 — real CDS hardware** — will use **pyepics, pyawg, cdsutils** (from
  git.ligo.org). Not started.

**HARD RULE:** stay on Phase 1. Do **not** talk about, plan, or speculate on Phase 2 / real
hardware **until the user explicitly says we are moving to real hardware** — and not before.
The roadmap report's "real-CDS / AWG↔NDS / CDSBackend" items are Phase 2 — out of scope for now.
See [[dont-guess-ask]].
