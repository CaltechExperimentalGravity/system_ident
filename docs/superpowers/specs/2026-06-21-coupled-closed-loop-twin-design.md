# Generic coupled + closed-loop twin + nonparametric MIMO recovery

**Status:** approved design, pre-implementation
**Date:** 2026-06-21
**Arc:** Step 1 of "joint MIMO identification of closed-loop systems" — the headline SEI/ISC
commissioning need (see `.llm/roadmap.md` audit, 2026-06-21).
- **Step 1 (this spec):** the generic coupled+closed-loop Python twin + nonparametric recovery.
- **Step 2 (next spec):** the joint parametric fit — shared modal poles across elements + MIMO
  Cramér–Rao bound.
- **Step 3 (final, named follow-on):** demonstrate the same recovery + joint fit on the **RTSfreerun
  compiled twin** (`x1hsts6dof`, real L1-MC2 loops). Because recovery runs through the
  `ChannelBackend` API, the generic twin and the existing `backends/rtsfreerun_adapter` feed the
  *same* code — the design below keeps recovery **backend-agnostic** so step 3 is a demonstration, not
  a rewrite.

Phase 1 (RTSfreerun) only — real hardware is Phase 2, explicitly out of scope.

## 1. Goal

Build a **dimension-generic, coupled, closed-loop suspension twin**, and verify that the existing
leakage-free **reference-based FRF recovers the open-loop coupled plant `G` through the live diagonal
control loops** (including the input/output decoupling matrices), **nonparametrically** (per matrix
element), matched to the known analytic plant.

This is the testbed the joint parametric MIMO fit (step 2) will run on — proven before we trust a fit
on its data. It serves SUS, SEI, ASC, and LSC: diagonal per-DoF control of a cross-coupled plant is
how LIGO loops are run today, and the residual in-loop coupling (below) is exactly what a commissioner
cannot get from today's SISO-per-DoF tooling.

## 2. Topology (the real LIGO damping-loop structure)

```
sensors (n_sens) ─► M_in (n_dof × n_sens, CONSTANT) ─► diagonal C_d(s) (n_dof, one per DOF)
   ▲                                                          │
   │                                                          ▼
coupled plant G(s) (n_sens × n_act) ◄── actuators (n_act) ◄── M_out(s) (n_act × n_dof, FREQ-DEPENDENT)
```

- **`M_in`** — sensor→DOF decoupling, **constant** (frequency-independent) matrix.
- **`C_d(s)`** — `n_dof` **diagonal** controllers (one SISO loop per control DOF). The loops are
  diagonal *by design*; the coupling lives in the plant.
- **`M_out(s)`** — DOF-control→actuator decoupling, **frequency-dependent** matrix of filters,
  designed to diagonalize the loop in either the **Euler** or the **eigenmode** basis — a
  **configurable** choice (which paradigm is optimal for IFO performance is an open study; the twin
  must let you *select* the basis, not bake one in).
- **`G(s)`** — the coupled **actuator→sensor** plant; this is what we identify.

**Dimensions are three independent integers** `n_sens`, `n_dof`, `n_act`:
- **SUS / SEI:** `n_sens = n_dof = n_act` (square — e.g. 6: L/P/Y/R/V/T).
- **ISC / LSC:** they may differ (rectangular `M_in`, `M_out`, and `G`).

A scalar matrix on a frequency-dependent coupled plant diagonalizes only approximately, so **residual
in-loop cross-coupling remains** — present in the twin, and recovered correctly by the FRF below.

## 3. Components (each independently testable)

1. **`MIMOSuspension`** — the coupled plant `G(s)`, `n_sens × n_act`, built from a shared normal-mode
   expansion (generalize `plant.coupled_suspension` past its current 2×2 L/P) with configurable
   coupling. Square and rectangular both supported. Carries an analytic oracle (`G.eval(freq)` →
   the `n_sens × n_act` complex tensor) to score recovery against.
2. **`CoupledLoop`** — holds `G(s)`, diagonal `C_d(s)`, constant `M_in`, frequency-dependent
   `M_out(s)` (with an Euler/eigenmode basis selector). Assembles the closed loop as **one MIMO state
   space via python-control** (`control.interconnect` + `control.minreal`). Exposes: the analytic
   open-loop `G`, the closed-loop signal maps needed by the backend, and a stability check.
3. **`MIMOTwinBackend`** — a **new** backend on the existing `ChannelBackend` API (inject / read /
   ramp_down). Injects the P&S periodic multisine at each actuator (per-DoF EXC, 3 s Tukey ramp),
   reads the **response tensor** (`n_sens` sensor channels) and the per-actuator **plant-input
   monitors** (true actuator input = injected drive + the control contribution fed back through
   `M_out`), under coupled **process + sensor noise**. Simulated **time-domain** with python-control
   (`forced_response` / `input_output_response`) so the in-band slow ringdowns (~20 s at 0.67 Hz) are
   physical, not a frequency-domain approximation.

**Boundary decision:** this is a **new** backend; `TwinBackend` and its SISO `coupling⊥controllers`
guard (`twin.py:113`) are **left untouched** (same choice as `DARMBackend`). Cleaner boundaries than
retrofitting the SISO polynomial closed-loop into MIMO.

## 4. Recovery (the step-1 deliverable)

Drive each actuator `j` with the multisine; read the response tensor `Y_i` (`i` over `n_sens`) and the
plant-input monitors `X_j` (`j` over `n_act`). Form the **leakage-free reference-based FRF tensor**
per pair, `H_ij = mean_p(Y_i)/mean_p(X_j)`, reusing `SysIDLoop._estimate_tf_periodic` (no new
estimation method). `H_ij` recovers the **open-loop** `G_ij` despite `M_in`, `C_d`, and `M_out` —
controller + decoupling cancelled — including the residual in-loop coupling. Verify `H_ij` matches the
analytic `G_ij` to tolerance.

**Drive sequencing:** **v1 drives one actuator at a time** (sequential — the cleanest input
separation, `n_act` campaigns). Simultaneous **uncorrelated** multisines across actuators (faster,
MIMO-orthogonal in one pass) is deferred to **v3**.

## 5. Dependencies

- **python-control** — already a declared dependency (`pyproject.toml`), present (0.10.2).
- **slycot — required.** Present in the `sysid` env (0.6.1); **add it to the declared deps**
  (`pyproject.toml`). CI must install it (conda-forge wheel / PyPI wheel) — the build uses slycot-backed
  python-control ops (interconnection, `minreal`). No pure-Python fallback; slycot is a hard dependency.

## 6. Testing

Run everything via `conda run -n sysid python -m pytest`. CI runs the small cases; the 6-DoF is a
heavier/marked check.

- **Construction / stability:** the assembled closed loop is stable (discrete poles inside the unit
  circle), `minreal` yields a clean realization.
- **Square sanity (2/2/2):** `M_in = M_out = I`, diagonal `C_d`, coupled `G` → the reference FRF
  recovers `G` to tight tolerance (loops cancel).
- **Square realistic (2/2/2):** non-trivial constant `M_in` + a representative frequency-dependent
  `M_out(s)` (a real Euler **or** eigenmode decoupler) + coupling → recover `G` to tolerance under
  coupled process+sensor noise; confirm the **residual in-loop coupling** is present and the recovery
  still returns the true `G`.
- **Non-square (3 sensors / 2 DOFs / 2 actuators):** recover the rectangular `G` (`n_sens × n_act`) to
  tolerance — exercises the rectangular `M_in`/`M_out`/`G` path.
- **Honest uncertainty:** the per-bin FRF variance is genuinely estimated (period-to-period variance,
  not floored); coherence/CRB sane. Tolerances set from real runs, never loosened to pass.
- **6-DoF instantiation (square, L/P/Y/R/V/T):** a marked/slower test or experiment confirming the
  same recovery at full SUS/SEI scale.

## 7. Out of scope (this spec)

- The **joint parametric fit** — one model with shared modal poles across all elements + MIMO CRB.
  That is step 2, its own spec.
- The **RTSfreerun demonstration** — running the recovery + joint fit on the compiled `x1hsts6dof`
  twin. That is step 3 (its own follow-on); this spec only ensures recovery stays backend-agnostic so
  it transfers.
- The **damping-paradigm study** (Euler vs eigenmode optimum) — the twin *supports* the basis knob;
  running the study is later work.
- **Real hardware / CDS** (pyepics/pyawg/cdsutils) — Phase 2, not to be touched until the user says so
  ([[two-phase-cds-plan]]).

## 8. Hard rules honored

- One P&S pipeline; reuse `_estimate_tf_periodic` + `multisine_from_psd`; no new estimation method.
- `conda run -n sysid` for all execution ([[use-conda-run-sysid-env]]).
- Any plot SVG + Git LFS; data-driven y-limits ([[graphics-svg-lfs-only]]).
- Trunk-based, push to main ([[trunk-based-push-to-main]]); don't silently reverse user changes
  ([[never-silently-reverse-user-commands]]); this spec was interview-driven, not guessed
  ([[dont-guess-ask]]). Phase 1 only ([[two-phase-cds-plan]]).
