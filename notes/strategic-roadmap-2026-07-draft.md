# system_ident — Strategic Roadmap (DRAFT for review)

> **STATUS: DRAFT — 2026-07-03.** For review by Rana. Nothing here is committed policy.
> Built on `.llm/roadmap.md`, `notes/roadmap-and-engineer-questions.md`, `CLAUDE.md`,
> `.llm/pintelon-schoukens-mimo-fit.md`, and a full code+experiment+docs audit
> (three parallel audits, 2026-07-03). Supersedes nothing until accepted.

This is a **single Pintelon–Schoukens (P&S) optimal-excitation** pipeline for LIGO
suspensions: periodic multisine → leakage-free reference FRF → ML fit → Cramér–Rao-driven
excitation. The roadmap keeps that spine and refuses divergent methods.

---

## 0. The one-paragraph verdict

The P&S **core is solid and the twins are faithful** — the genuine headline is RTSfreerun
A1–A4 plus the rank-1 joint MIMO modal fit closing through the *real* L1-SRM production
dampers. But the project has a **two-tier maturity split** that is now the central risk:
the **SISO path (`loop.py`)** is a mature, config-driven, iterative, CRB-in-the-loop,
watchdog-guarded pipeline, while the **MIMO path (`mimo_fit.py` + experiment scripts)** is a
**single-shot, script-only, oracle-seeded** research prototype with no config integration and
no MIMO Fisher/CRB tool. The recent session's hard-won lessons (fabricated ADC removed; flat
drives forbidden; the "doublet" is spatial, not a resolution limit; the fit needs
oracle-anchoring to not go degenerate) all point at the **same root cause**: the MIMO fit is
**under-hardened**, and some prose still mislabels under-modeling / parameterization artifacts
as "physical limits" — the exact trap `CLAUDE.md`'s feasibility gate exists to stop. **The
right work is not more twin polish.** It is (A) making the MIMO fit trustworthy *without the
oracle*, (B) auditing twin fidelity and productionizing the MIMO iterative loop, and (C) a
concrete, question-driven bridge to real CDS.

**Update (2026-07-03, later):** the central risk is now **largely retired** — the MIMO fit is
**blind** (find_modes, no oracle), has a **first-class MIMO Fisher/CRB**, and **iterates**.
Progress snapshot below.

## 0b. Progress snapshot (2026-07-03)

| Item | State | Where |
|---|---|---|
| ADC fabrication removed; doublet resolved spatially; uncertainty-aware (prior-robust) drive; feasibility-gate prose purged | ✅ done | `e8d8358`, `bccfa79`, `68124fe`, `2fdd9d8` |
| **Top-1** data-driven mode-finder + BLIND flagship adoption | ✅ done | `mimo_fit.find_modes`; `ba8ee01`, `479f239` |
| **Top-2** first-class MIMO Fisher/CRB + DONE criterion | ✅ done | `mimo_fit.mimo_fisher_matrix`/`mimo_parameter_covariance`/`modal_frac_uncertainty`; `62598de` |
| **Top-3** drive α-vs-concentration: derived-α floor + sparse Fisher-placed lines | ✅ capability (adoption = wire sparse lines through the campaign + regen) | `design/pintelon.select_excited_lines`, `floor_energy_frac`; `f24acca`, `68124fe` |
| **Top-4** MIMO iterative loop | ◑ engine + `u_decay` + `combine` hook done; **config integration, SRM inverse-variance combine, hard-scenario blind run remain** | `mimo_iterate.iterate_mimo`; `0759cba`, `0f0e0ec` |
| **Top-5** twin-fidelity ledger + guard test | ✅ done | `notes/twin-fidelity-ledger.md`, `tests/test_twin_fidelity.py`; `d865b8b` |
| **A5** `recover_open_loop` conditioning guard + `assemble_campaign` resolution guard | ✅ done (adaptive `_choose_transient` for MIMO remains) | `mimo_loop`, `mimo_campaign`; `81937a4` |

**Open next:** #3 sparse-line adoption (campaign regen); #4 config.py integration + SRM combine
impl + hard-scenario blind validation; per-mode-Q twin realism; the blind `[5b]` doublet
(per-block fit robustification); **Phase C is still gated on the operator questions.**

---

## 1. What is SOLID (protect it; don't re-litigate)

- **P&S SISO pipeline** — `loop.py`, `model.py`, `fisher.py`, `excitation.py`, `config.py`,
  `design/pintelon.py`. Prior-robust first pass → point-optimal later passes; inverse-variance
  multi-pass accumulation (`_accumulate`); adaptive transient drop (`_choose_transient`);
  leakage-free reference FRF `H = mean(Y)/mean(X)` (cancels the controller in closed loop);
  Fisher/CRB wired into both design and the stop criterion; validated bit-for-bit vs the legacy
  `sysIDlib` engine. **This is the reference architecture the MIMO path must converge to.**
- **The rank-1 modal *model*** — `mimo_modal.py::Rank1ModalModel` has a clean analytic Jacobian,
  frequency normalization, and honest pole extraction. The model object is not the problem.
- **Twin fidelity of the *plant*** — `rtsfreerun_adapter.py` correctly reconstructs the virtual
  `PLANT_IN` with the right feedback sign (`FEEDBACK_COEFF=-1` for the "+−" coil sum); the
  compiled `x1hsts6dof` runs the same C numerics a real front-end runs; oracle cross-checks
  (SS + independent SOS) in `rtsfreerun_oracle.py`.
- **The fabricated ADC is genuinely gone** — verified: no `quantize`/`adc_bits`/`adc_range`
  remains in `src/` or `experiments/`. No other fabricated noise source found.
- **Design tooling** — `design/resolution.py::recommend_resolution` sizes `nperseg/df/n_transient`
  from the prior ringdown (df ≤ Δf/few-bins, T ≳ Q/f0). Matches the "df/T are knobs" doctrine.
- **Safety scaffolding** — watchdog (saturation + RMS), ramp-in/out outside the measured record,
  safe-state handoff, dashboard STOP.

## 2. What is FRAGILE / half-built (the real work)

1. **MIMO fit is oracle-seeded, oracle-anchored, oracle-scored (TOP RISK).**
   In `run_srm6dof_modal.py`: `prior_init` seeds poles *directly from the analytic oracle
   modes* (f0 **and** Q=50), `fit_modal` anchors with `PRIOR_WEIGHT=1e12` toward those same
   oracle f0's, and `score_fit` grades against the same oracle. The flagship "11/13 modes,
   median Q-err 0.4%" therefore shows a *truth-seeded, truth-anchored* fit **stays put** — not
   blind identifiability. On live hardware there is **no oracle to seed/anchor to.**
   *(Mitigation already in place, and correct: the CRB uses the DATA-only Jacobian
   `FitResult.jac`, so the reported error bars are honest, not tightened by the 1e12 anchor.)*
   The honest cold-start evidence is `run_srm6dof_iterate.py` (far +30% prior, **no anchor**) —
   it is under-emphasized and should become the primary MIMO demo.
2. **`peak_pick_modes` collapses on fine-df data.** `mimo_fit.py::peak_pick_modes` uses
   `scipy.signal.find_peaks` on total power; on the 0.004-Hz grid it reads sidelobes and
   piles modes onto adjacent bins of the strongest peak. This is *why* the script fell back to
   oracle-seeding. A robust, **data-driven mode-finder + model-order selection** is the missing
   keystone.
3. **No MIMO Fisher/CRB in the library.** `fisher.py` is SISO/`TFModel`-only. The MIMO CRB is
   assembled by hand in scripts from `Rank1ModalModel.jacobian`. The feasibility-gate discipline
   (compute the bound!) has no first-class MIMO tool.
4. **MIMO measurement is single-shot and un-hardened.** `mimo_campaign.py::assemble_campaign`
   does a **fixed** transient drop (no `_choose_transient`), no excited-line masking, no
   synchronous-bin assertion, hardcoded ramp; `mimo_loop.py::recover_open_loop` inverts X per
   bin with **no conditioning guard** (ill-conditioned near resonance).
5. **No config/loop integration for MIMO.** `config.py` builds only `TFModel` priors + the SISO
   estimator/designer. There is **no** `Rank1ModalModel` prior builder, no MIMO estimator in the
   registry, no iterative MIMO loop. The MIMO path lives entirely in `experiments/` scripts.
6. **Drive α-vs-concentration tension (unresolved).** `FLOOR_FRAC=0.05` applied across **all**
   ~in-band lines makes ~95% of the energy floor → near-flat, defeating concentration and the
   iteration payoff. See §5 for the principled fix.
7. **Framing that violates the feasibility gate.** `srm_modal_report.md` and the
   `distinct_oracle_modes` docstring still call the doublet/triplet collapse a *"physical
   resolution limit."* `CLAUDE.md` explicitly lists this as a **parameterization** failure — and
   the code itself already disproves it (`fit_block_decoupled` resolves the doublet spatially).
   The "0.004 Hz resolution knee" prose is likewise a soft limit, not a physical one. **Purge or
   quantify.**
8. **Statistical story gaps.** Noise estimator is the **period-variance** estimator but is
   sometimes labeled **LPM** (a different technique); coherence is *floored* not measured on a
   clean twin; **no model-order-selection** procedure; 2-period records underflow to a fabricated
   σ floor (`1e-9`).

## 3. Twin-fidelity ledger (what's physical vs assumed)

| Component | Where | Status |
|---|---|---|
| Seismic ASD (`ligo-india`, microseism 3e-6@0.15Hz, NLNM floor) | `srm6dof_loop.py::_seismic_at_m1_asd` | **Physical** — matches digital_twin `noise.py`; *but* referred to the coil node `DRIVE_EXC`, not the suspension point (approx, honestly labeled) |
| gnd→M1 via `HSTS_GND_TF` × ISI transmissibility | same | **Physical** — real `.mat` residues + real ISI fn; P has zero seismic (no gnd→M1 pitch path — legitimate) |
| OSEM/BOSEM readout 1e-10 m/√Hz, 1 Hz knee | `srm6dof_loop.py::bosem_noise_spec` | **Plausible/physical level**; injected at damper *sensor node*, not through an in-loop quantized sensor (documented compromise) |
| Actuator: 30000-count coil limit | asserted in report | **Not enforced in code** (no saturate in SRM campaign); peak drive ~0.068 counts |
| Uniform **Q=50 on every mode** | inherited from `hsts_full.mat` state space | **Idealization** — real HSTS has per-mode Q; makes the "Q recovery" story easier (all targets identical & known) |
| 16-bit ±1mm ADC quantizer | REMOVED (commit e8d8358) | **Gone — verified** |
| `cds.py` real-hardware backend | `backends/cds.py` | **Pure stub** — every method `raise NotImplementedError`; honest LIMITATIONS docstring |
| Adapter seismic ASD is a *copy* of the twin's | `rtsfreerun_adapter._seismic_asd` | Values match now; **drift risk** — reconcile periodically |

---

# THE 3-PHASE PLAN

Phases are **capability tiers, run mostly in order but with overlap**, not calendar quarters.
Phase A is pure simulation hardening; Phase B is twin-based trust/transfer; Phase C is the
real-CDS bridge (now explicitly in scope — enumerated as questions, not guesses).

---

## PHASE A — Simulation: harden the P&S pipeline (DO THIS FIRST)

**Goal.** Make the MIMO fit trustworthy *without an oracle*, resolve the drive tradeoff, and
close the maturity gap between the SISO loop and the MIMO path — all in pure simulation where
the truth is known and iteration is cheap.

### A-priorities (ranked)
1. **Data-driven mode-finder + model-order selection (the keystone).** Replace/augment
   `mimo_fit.py::peak_pick_modes` so it does **not** collapse on fine-df data and needs **no**
   oracle seed. Approach: stabilization-diagram / order-sweep over `Rank1ModalModel` n_modes with
   an information criterion (AIC/MDL) or whiteness-gated residual test; cluster candidate poles
   across orders; reject spurious sidelobe poles by pole–residue stability, not by knowing the
   answer. Deliverable: `run_srm6dof_iterate.py`-style cold start recovers ≥11/13 modes with the
   **anchor OFF** and **seed from a broad prior**, not the oracle.
2. **First-class MIMO Fisher/CRB.** Add a MIMO Fisher to `fisher.py` (or a sibling) built from
   `Rank1ModalModel.jacobian` weighted by the campaign covariance `Cz`, yielding a per-mode
   `(f0,Q)` uncertainty and a `_frac_uncertainty`-analogue DONE criterion. This is what lets the
   feasibility gate apply to MIMO with real numbers instead of by-hand scripting.
3. **Resolve the drive α-vs-concentration tradeoff** (see §5 for the derivation). Make the floor
   apply to a **chosen sparse excited-line set** placed by the dispersion function near the
   informative modes — not spread across every in-band bin — and make α **derived** from a target
   floor-energy fraction so it doesn't collapse to flat as the line count grows.
4. **Purge feasibility-gate-violating prose.** Rewrite `srm_modal_report.md` and the
   `distinct_oracle_modes`/resolution docstrings: the doublet is **spatial** (resolved by
   `fit_block_decoupled`), the triplet's within-plane pair is a **parameterization** case to fix
   (shared-pole collapse), not a physical limit; either resolve the 1.51 Hz within-plane pair
   (per-plane multi-mode fit / higher order) or state the SNR·N ≳ (Γ/Δf)⁴ number that would beat
   it. No "resolution knee" language without a CRB behind it.
5. **Harden `mimo_campaign.py` / `mimo_loop.py`** to SISO parity: adaptive `_choose_transient`,
   full-energy/ramp-period trimming, excited-line masking, synchronous-bin assertion, and a
   conditioning guard in `recover_open_loop`.

### Already solid (reuse, don't rebuild)
- `Rank1ModalModel` (analytic Jacobian), `fit_block_decoupled` (spatial doublet), the SISO
  `_choose_transient`/`_accumulate`/`_estimate_tf_periodic` as the porting template.

### Gaps / risks
- The mode-finder is genuinely hard on tight clusters; budget for it. But per the feasibility
  gate, **prove** any residual "can't resolve" with the super-resolution bound before calling it
  a limit — parametric ML super-resolves.
- Uniform Q=50 makes A look easier than reality; validate the mode-finder on a **per-mode-Q**
  synthetic (see Phase B) before trusting it.

### Concrete next actions (files/functions)
- `src/system_ident/mimo_fit.py`: new `find_modes_stabilized(G, freq, ...)` (order-sweep +
  clustering); demote `peak_pick_modes` to a helper.
- `src/system_ident/fisher.py`: `mimo_fisher_matrix(model, exps, freq)` +
  `mimo_parameter_covariance`.
- `src/system_ident/design/pintelon.py`: sparse excited-line selection + derived-α floor
  (`floor_energy_frac` param replacing raw `floor_frac` at the call sites).
- `experiments/rtsfreerun/run_srm6dof_iterate.py`: promote to the **primary** MIMO demo
  (anchor OFF, broad-prior seed); regenerate `srm_modal_report.md` from it.
- Tests: cold-start recovery without oracle; MIMO CRB vs Monte-Carlo (extend
  `test_mimo_fit.py::test_crb_matches_monte_carlo` to the sweep-selected order).

---

## PHASE B — RTSfreerun digital twin: make ID trustworthy & transferable

**Goal.** Turn the demo into a *validated, productionized* twin-based identification: audit the
twin's physics, productionize the MIMO iterative loop, validate against the oracle blind, and
add per-mode-Q realism so the method isn't tuned to an idealization.

### B-priorities (ranked)
1. **Physical-fidelity audit of the twin (institutionalize it).** The fabricated ADC proved we
   need a standing check. Deliverable: a short `twin-fidelity-ledger.md` (seed from §3 above) that
   lists every noise/disturbance/actuator/sensor component, its source, and whether it is
   *physical / referred-approximation / idealization*, plus a test that fails if a noise source
   appears that isn't traceable to the twin. Specifically resolve: seismic referral point
   (coil-node vs suspension-point), and whether an in-loop quantized sensor
   (`READOUT_NOISE` `cdsFilt` rebuild, as `x1hstsdamped` has) is worth building for a true in-loop
   OSEM sensor.
2. **Productionize the MIMO iterative loop.** Mirror `SysIDLoop` for the MIMO path: prior-robust
   first pass → MIMO dispersion reallocation → inverse-variance accumulation of `(Ybar,Ubar,Cz)`
   across passes → refit (Phase-A mode-finder) → MIMO CRB check → repeat until a **per-mode CRB
   target**. Wire it into `config.py` (a `Rank1ModalModel` prior builder + a MIMO estimator in the
   registry + watchdog/safe-handoff). This is the SISO/MIMO unification.
3. **Blind oracle validation.** Score the productionized loop against the analytic oracle with the
   **seed and anchor OFF** and the operator NOT told the modes — the credibility test the current
   flagship skips. Report f0/Q/σ and CRB-vs-empirical calibration.
4. **Per-mode-Q realism.** Replace the uniform Q=50 with a per-mode Q (from HSTS design values /
   measured loss) in the twin/oracle so the mode-finder and CRB are exercised on a non-idealized
   plant. Removes the "all targets identical & known" crutch.
5. **Statistical story fixes.** Rename/relabel the period-variance estimator consistently (stop
   calling it LPM unless the real LPM is implemented for the few-period regime); measure coherence
   instead of flooring it; add σ-calibration (predicted vs empirical) and a residual
   whiteness/Gaussianity test.

### Status & findings (B core — done 2026-07-03)
The loop ENGINE is built and validated; two findings reshape the remaining priorities.

- **Built:** `src/system_ident/mimo_iterate.py::iterate_mimo` — the callback-based
  estimate→redesign→re-measure loop (prior-robust first pass → point-optimal from the trusted
  fitted modes → stop when `modal_frac_uncertainty` < target), unit-tested with mocks (no
  campaigns). `run_srm6dof_iterate.py` drives it on the twin with a fully BLIND fit
  (`find_modes` + `mimo_parameter_covariance` + `modal_frac_uncertainty`). Remaining for
  priority 2: `config.py` integration + cross-pass inverse-variance accumulation.
- **Finding 1 — iteration is marginal on the perfect-prior, high-SNR twin.** Far-prior end-to-end
  run (+30% wrong prior): pass 0 (robust) already hits 12/13 within 1% at frac-unc 1.85e-5;
  pass 1 (point-optimal) gives 1.62e-5 — only **1.1× tighter**. Iteration's value is real only
  when pass 0 *can't* cope (far prior, scarce SNR, or a much smaller floor). So: **make the loop's
  validation scenario a genuinely hard one** (far prior AND scarce SNR / low budget), not this
  idealized twin — else the loop looks pointless.
- **Finding 2 — point-optimal redesign can OVER-concentrate.** Pass 1 narrowed recovery 12→10
  modes within 1% while the worst-case CRB improved: concentrating the budget on the fitted modes
  under-drives weakly-coupled ones. So `frac_unc` (worst-case CRB) is **not a complete quality
  proxy**. Redesign refinement (fold into priority 2): keep a **broader floor** on later passes
  (don't collapse to pure point-optimal), and add a **recovery-breadth / coverage** term to the
  DONE criterion alongside the worst-case CRB. Same α-vs-concentration tension as Phase-A §5 —
  resolve them together.

### Already solid (reuse)
- `rtsfreerun_adapter.py` (PLANT_IN reconstruction), `rtsfreerun_oracle.py` (oracle + SOS
  cross-check), the caches, the closed-loop reference-based recovery (diagonal FRF → oracle at
  2e-4).

### Gaps / risks
- The compiled 6-DoF model exposes no ground/ISI port and no in-loop readout-noise chain — the
  referrals are approximations; the audit must state their validity range.
- Housekeeping: two stale PNGs (`hsts_recovery.png`, `hsts6dof_recovery.png`) violate the
  SVG+LFS rule; ~90 MB of untracked `.npz` caches + an orphaned doublet cache/SVG; **no**
  experiment plots are tracked in git/LFS at all. Clean up + LFS the SVGs.

### Concrete next actions (files/functions)
- ✅ `src/system_ident/mimo_iterate.py::iterate_mimo` (loop engine, callback-based) — DONE.
- `src/system_ident/mimo_iterate.py`: add cross-pass inverse-variance accumulation of
  `(Ybar,Ubar,Cz)`; a broader-floor late-pass redesign + a coverage term in the DONE criterion
  (findings 1–2 above).
- `src/system_ident/config.py`: `build_priors` emits `Rank1ModalModel`; register a MIMO
  estimator/designer; a `RunConfig`-driven wrapper around `iterate_mimo` with the watchdog.
- `experiments/rtsfreerun/`: a HARD-scenario blind-validation script (far prior + low budget,
  seed/anchor OFF); per-mode-Q twin variant.
- `notes/twin-fidelity-ledger.md`; `tests/`: a "no untraceable noise source" guard test.

---

## PHASE C — Live CDS (real hardware): the bridge, question-first

> **Previously Phase-2 / off-limits; now EXPLICITLY in scope.** Per `CLAUDE.md` and the
> "don't guess" rule, this phase **does not fabricate any CDS/LIGO operational specific.** It
> enumerates the binding unknowns as short, direct questions for the operators/user and identifies
> the minimal real-hardware bridge. **Do not write hardware code until these are answered and the
> user green-lights it.**

### What's real-hardware-ready vs twin-only
- **Ready-ish / designed:** `backends/base.py` ABC (clean), the periodic-multisine synthesis
  (`excitation.py`), the leakage-free estimator (physics is hardware-agnostic).
- **Stub:** `backends/cds.py` — every method `NotImplementedError`; assumes awg/cdsutils +
  nds2 (lazy-imported, absent). Its docstring already enumerates the real gaps.
- **Twin-only:** everything that reconstructs `PLANT_IN` from compiled-model internals; the
  referred-noise injection; the oracle.

### The minimal bridge (once questions are answered)
1. Implement `CDSBackend.inject/read/ramp_down` against awg/nds2 for **one** DoF on the compiled
   RTSfreerun model first (same API), then real hardware.
2. A **timing/decoherence monitor**: verify integer-period synchronous capture survives AWG↔NDS
   clocking; detect per-period phase walk of the drive lines (the leakage-reintroduction failure).
3. A **DAC-frame, filter-aware drive designer** + **pre-injection** worst-case-peak saturation
   check at the DAC node (the watchdog is currently post-hoc/per-segment — a saturating drive
   reaches the coil before breach is seen).
4. One **head-to-head vs diaggui/awggui** on a shared measurement (equal wall-clock, equal
   in-band RMS) — the comparison a controls engineer actually trusts.

### KEY OPEN QUESTIONS FOR THE OPERATORS/USER (do not guess — answer before coding)
*(These consolidate `notes/roadmap-and-engineer-questions.md` §A–I; ask the highest-leverage
ones first.)*
1. **Channels.** For a given suspension/DoF: which channel do we inject on (AWG excitation
   point), which is the **after-controller drive monitor** that is the true FRF input `X`, and
   which is the readback `Y`? (A wrong/one-cycle-delayed `X` silently biases toward the
   closed-loop response.)
2. **Actuator range & units.** What is the binding DAC/coil limit in counts per stage (30000?),
   and the counts↔physical (N, m) calibration per stage? Is the top stage an **ESD** with
   bias-voltage/charge constraints the generic saturation check won't know?
3. **Injection mechanism & safety.** How is a multisine safely injected on a *live/locked*
   suspension — via awg? What are the watchdog/abort hooks, ramp requirements for a high-Q mode,
   and does snapshot/restore need to capture **filter-module state/history** (SDF,
   `fm_clear_history`), not just the drive array?
4. **Timing & synchronization.** Front-end 16384 Hz vs analysis `fs`; do GPS timestamps let us
   align the capture to integer periods; is there measurable AWG↔NDS fractional-sample drift over
   a multi-minute record?
5. **Damping-loop state during ID.** Do we identify with the production dampers **engaged**
   (reference-based recovery cancels them — our demonstrated mode) or disabled? Stability margin
   with a comb near the UGF?
6. **Lock / Guardian.** Is ID done in observing, or only non-observing time? Do we need Guardian
   awareness (request state, abort on lock-loss)? In-band footprint acceptable?
7. **Interop.** Do engineers need the fit exported as a **Foton ZPK/SOS** filter string to drop
   into a damping bank? (Foton export was previously removed.)
8. **Provenance.** What manifest must a delivered fit carry (config, prior, seed, GPS span, model
   order, σ, oracle) for traceability/reproduction on their IFO state?

### Concrete next actions (files/functions)
- `backends/cds.py`: implement against the compiled RTSfreerun model first (twin-in-the-loop),
  gated on Q1–Q4 answers.
- New: DAC-frame designer (whitening/coil-driver Foton model between designed PSD and injected
  counts) + pre-injection peak check hooking `safety.py`.
- New: decoherence monitor (per-period drive-line phase walk).
- `loop.py`/`config.py`: Foton export, provenance manifest on `LoopResult`.

---

## 4. TOP-5 PRIORITIES (ruthlessly ordered, cross-phase)

1. ✅ **[A] Data-driven mode-finder + order selection** so the MIMO fit works **without the
   oracle** — **DONE** (`find_modes` + blind flagship adoption).
2. ✅ **[A] First-class MIMO Fisher/CRB** — **DONE** (`mimo_fisher_matrix` /
   `mimo_parameter_covariance` / `modal_frac_uncertainty`; in `mimo_fit.py`, not `fisher.py`).
3. ◑ **[A] Resolve the drive α-vs-concentration tradeoff** — derived-α floor + sparse
   Fisher-placed lines (`select_excited_lines`) **DONE**; sparse-line pipeline adoption pending.
4. ◑ **[B] Productionize the MIMO iterative loop + config integration**, then **blind**
   validation — loop engine + `u_decay` + `combine` **DONE**; config integration, the SRM
   inverse-variance combine, and the hard-scenario blind run remain.
5. ✅ **[A/B] Purge "physical limit" prose** and institutionalize the **twin-fidelity ledger +
   guard test** — **DONE** (`notes/twin-fidelity-ledger.md`, `tests/test_twin_fidelity.py`).

*(Phase C is high-value but gated on the user answering §Phase-C questions; do not start
hardware code first.)*

## 5. Appendix — the drive α-vs-concentration fix (proposed)

**Problem.** A meaningful per-line floor `α·peak` guarantees every multisine component carries
iterable power, but applied across **all N in-band bins** the floor energy `≈ N·α·peak·df`
dominates the fixed budget `Px_tot` once N is large (~2000 lines) → the drive is ~95% floor →
near-flat → concentration and the iteration payoff vanish. This is the tension in commit 68124fe.

**Root cause.** The floor is applied to the **dense rfft grid**, but a P&S multisine only needs
power on a **chosen sparse line set**. Flooring bins that shouldn't be excited at all is what
flattens the drive.

**Proposed fix (two parts).**
1. **Choose the excited-line set first** (sparse), placing lines by the dispersion function
   near the informative modes (and a thin covering comb for prior robustness) — *then* apply the
   floor only across that set. Unexcited bins stay at zero, not the floor.
2. **Derive α from a target floor-energy fraction**, not a fixed 0.05. Pick
   `floor_energy_frac ≈ 0.1–0.2` of `Px_tot`; with `L` chosen lines and peak `p`,
   set `α = floor_energy_frac · Px_tot / (L · p · df)` (clipped). Then the floor is a *fixed
   small share* of the budget regardless of `L`, and the Fisher-shaped peaks keep the rest.

This preserves "power on every excited line so it can iterate" (the P&S-correct requirement)
**without** collapsing to flat, and makes the knob principled rather than a magic 0.05. Validate
on `run_srm6dof_iterate.py`: concentration ratio and per-line SNR should both stay high while
every excited line remains above the iterate-able floor.
