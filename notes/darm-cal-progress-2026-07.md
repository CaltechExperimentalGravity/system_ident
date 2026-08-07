# DARM calibration — progress & handoff (July 2026)

Status snapshot so context can be cleared and the work resumed cleanly. Goal: apply the single
Pintelon–Schoukens pipeline (periodic multisine → leakage-free reference FRF → ML fit toward the
Cramér–Rao bound) to the LIGO **DARM** readout — the slowest recurring calibration — on a
representative closed-loop twin, raising fidelity toward a real instrument. Phase-1 twin only.

## Done

- **Static single-loop calibration** — `src/system_ident/darm.py`, page
  `docs/examples/08-darm-calibration.qmd`, tests `tests/test_darm.py` (~15). Sensing `C`
  (cavity pole + delay), three-stage actuation `κ` (UIM/PUM/TST), derived servo `D`, response
  `R(f)=(1+G)/C` with a **measured** CRB envelope, and a swept-sine head-to-head. Pre-existing.

- **Round 1 — track a drifting parameter** (this cycle) —
  `src/system_ident/darm_tv.py`, page `docs/examples/13-darm-drift-tracking.qmd`, glue
  `docs/darm_tv_demo.py`, tests `tests/test_darm_tv.py` (7).
  - **Method:** the plant drifts far slower than one record, so each measurement is *locally
    stationary*. Take leakage-free P&S **snapshots** of a drifting ESD strength `κ(t)` with Pcal
    as the ruler (sensing `C` cancels in H_stage/H_pcal → snapshot is drift-immune), then fit
    `κ(t)=Σ cₖ bₖ(t)` in a Legendre time-basis by weighted LS (a **Lataire–Pintelon** basis
    expansion, two-step form). Coefficient covariance `(BᵀWB)⁻¹` = CRB on `κ(t)` and `κ̇(t)`.
  - **Result:** a **5 %/hour** drift on TST recovered **within an honest CRB** (pull z≈0.5),
    **resolvable by a computed ~12×** (bound, not eyeball). Static limit (no drift) degrades
    gracefully to flat-at-truth. Per-snapshot σ≈1 % of κ; tracking σ≈0.4 % of κ.
  - Key API: `darm.with_params(**overrides)`, `darm.drift_profile(...)`,
    `darm_tv.snapshot_kappa / track_kappa / basis_matrix / fit_tv (→TVFit.predict) / resolvability`.

- **Housekeeping:** relabeled the period-variance noise estimator honestly (was mislabeled
  "LPM" in `notes/darm-calibration-via-pns.md` §3); fixed a latent `darm ↔ darm_adapter`
  circular import that only bit a fresh `import darm_tv` (the render path) — fixed by importing
  `.darm` first; smoke test guards it.

- **Roadmap published** at the foot of `docs/examples/08-darm-calibration.qmd`.

## Decisions locked (from the planning interview — carry into the next round)

- Scope is **fidelity-first**; the wall-clock *campaign design* is a later round.
- Drift is treated as **genuinely slowly-time-varying**, not shortened-record snapshots-only.
  Ground-truth generation → **GP / correlated wander**; estimator workhorse → **Lataire–Pintelon
  basis expansion** (round 1 used a deterministic profile as the placeholder truth).
- The SRC-detuning optical-spring pole is to be parametrized on the **physical detuning φ(t)**, so
  the DARM pole crosses **real → complex** analytically (not on Q_s or pole locations).
- Noise fix = **relabel only** this round (no LPM implementation).
- Whitening/DAC chain, ESD limits, real-CDS/Foton import, MIMO: **out** for now.

## Next (roadmap, in order — also in example 08)

1. Realistic **GP drift** (stochastic wander, physical correlation time; score statistically).
2. **SRC-detuning optical spring** — replace the single real sensing pole with an optical biquad
   in `φ`; DARM pole migrates real → complex. (User note: RSE ~395 Hz real pole → complex; ± detuning
   phase; sign/unstable-spring regime not critical yet.)
3. **Several parameters drifting jointly** (optical gain, cavity-pole/detuning, stage κ_i).
4. **One-shot Lataire–Pintelon** fit (coefficients direct to the record) vs the two-step form.
5. **Continuous TDCF lines** — always-on κ(t) tracking between sweeps.
6. **Actuation / DAC chain** — whitening/de-whitening, coil driver, pre-injection saturation
   check; ESD charge/voltage limits.
7. **Real filters + MIMO** cross-couplings, toward an actual instrument state.
   Live-CDS / hardware injection is a separate, later phase.

## Blockers / repo state

- **GitHub org billing is locked** (est. ~1 week from 2026-07-09). It disables **both** Actions
  (CI red regardless of code — runners never acquire the job; the real reason shows only in job
  *annotations*) **and** Git LFS (a push carrying a new SVG is rejected: "exceeded its LFS
  budget"). Diagnostic tip: when CI logs 404/BlobNotFound, read the job annotations.
- Consequence: the **example-13 gallery thumbnail** (`docs/examples/thumbnails/13.svg`, LFS) is
  parked in a **deferred local commit** — it will push on the next `git push` once LFS unlocks.
  All non-LFS work (code, both pages, roadmap, tests) is on `origin/main`.
- **Verification gate while CI is down:** `conda run -n sysid pytest` and
  `cd docs && conda run -n sysid python -m quartodoc build && conda run -n sysid quarto render docs`.
  Everything passes locally (pytest incl. the 7 new TV tests; full 48-page site renders).
- Regenerate the thumbnail if needed: `docs/make_assets.py::thumb_13`.

## Pointers

- Design + citations: `notes/darm-calibration-via-pns.md`.
- P&S closed-loop / time-varying method notes: `.llm/pintelon-schoukens-mimo-fit.md`,
  `.llm/engineering-practices.md`, `.llm/ps-book/README.md`.
