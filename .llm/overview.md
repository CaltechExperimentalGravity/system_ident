# .llm/ — system_ident agent working notes

Working notes for agent sessions (mirrors the `.llm/` convention in the companion
`digital_twin/` repo). Committed, but not part of the package or the published docs.
`.claude/NOTES.md` holds the broader project log; this `.llm/` focuses on the **digital-twin /
RTSfreerun integration** work.

## TL;DR
`system_ident` is a Pintelon–Schoukens optimal-excitation system-ID package for LIGO suspensions
(leakage-free FRF, ML fit at the Cramér–Rao bound, first-class closed loop, coupled-MIMO plants).
The current thread: make it **CDS-aware by driving the companion digital twin**
(`digital_twin/`, the rtsfreerun-compiled CDS models) with realistic noise — via one adapter
backend.

## Pages
- [roadmap.md](roadmap.md) — **master execution roadmap** (tracks A–C: RTSfreerun twin demos,
  MIMO joint ID, refinement bake-off; each tagged local / twin-box). Start here to decide what to
  run next.
- [digital-twin.md](digital-twin.md) — summary of `digital_twin/twin/`: the `mdl` API, channel
  conventions, the noise model, the shipped models, and a real `x1hsts` suspension scenario.
- [rtsfreerun-integration.md](rtsfreerun-integration.md) — the `RTSfreerunBackend` adapter design,
  local mock-testing strategy, config/CLI wiring, and the iteration roadmap (smoke → SISO → closed
  loop → MIMO).
- [pintelon-schoukens-mimo-fit.md](pintelon-schoukens-mimo-fit.md) — **literal P&S procedure for the
  joint MIMO parametric fit (step 2)**, scraped from `docs/SysID-Pintelon.pdf` with exact eq/page
  citations: common-denominator model, SML equation-error cost, Gauss–Newton, SK starting values,
  CRB, `f0/Q` propagation, validation — plus the mapping onto our twin and corrections to the
  pre-book design assumptions. **Read before writing the step-2 spec.**
- [ps-book/README.md](ps-book/README.md) — reference index into Pintelon & Schoukens: chapter map,
  question → chapter table, primary papers. The book itself is copyrighted and is **not** in the
  repo.

## Environment notes
- Run everything via `conda run -n sysid …` (this repo's env). For the RTSfreerun path we build the
  compiled RTS model(s) **into the `sysid` env** (separate repos — we don't install `system_ident`
  into the twin env); see README §"Run against the RTSfreerun digital twin" + the
  `rtsfreerun-env-strategy` memory. On the twin box, `x1hsts` is built into `sysid` and the
  real-model test runs (no longer skipped); the mock still covers machines without a built model.
