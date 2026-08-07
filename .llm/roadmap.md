# system_ident — master execution roadmap

The single execution-ready roadmap for what's left. Written so a **fresh Claude session on another
machine** (e.g. the twin box) can pick up any track and run it without re-deriving
context. Companion to [overview.md](overview.md),
[digital-twin.md](digital-twin.md), [rtsfreerun-integration.md](rtsfreerun-integration.md). The
broader project log is `.claude/NOTES.md`.

---

## How to use this file (read first)

**Environment & guardrails — non-negotiable:**
- Run **everything** through `conda run -n sysid …`. Never invoke Python outside the `sysid` env.
  On the twin box the demos additionally need the `twin` env (where the rtsfreerun model is built);
  see [digital-twin.md](digital-twin.md) "Install / run".
- **If a library is missing or broken, stop and ask the user.** Do not hunt for substitutes or
  improvise an alternative. (Standing user instruction.)
- **FRF excitation is ALWAYS the Pintelon–Schoukens multisine** — `excitation.multisine_from_psd`.
  Gaussian noise (`excitation.timeseries_from_asd`) is **background/disturbance only**, never the
  identification drive. Any new measurement code must preserve this; it was audited repo-wide.
- The grid constraint everywhere: **`T_fft = nperseg/fs > 3/f_min`** (enough cycles of the slowest
  line to resolve it). Drop the first `n_transient` period(s) as settling.

**Pick a track by its blocker tag:**
- 🟢 **local** — runnable now in `sysid` on any machine (incl. this Mac).
- 🔵 **twin-box** — needs a built rtsfreerun model (`twin` env, Linux/CDS box).

**Per-track shape** (every track below has all seven): Goal · Why · Status/prereqs · Steps ·
Verify · Definition of done · Blocker.

---

## Status snapshot (2026-06-21)

**Done & green** (full suite **148 passed, 1 skipped** locally — incl. **13 RTSfreerun real-model
tests** that skip in CI where the compiled model isn't built):
- P&S optimal-excitation pipeline: prior-robust → point-optimal multisine design, leakage-free
  periodic-DFT FRF (ratio-of-averages), ML fit (`GMLEstimator`), Fisher/CRB uncertainty, safety
  watchdog + safe handoff, dashboard.
- **Closed-loop first-class** — config-declarable injection point + drive-monitor channel recovers
  the open-loop plant. (`configs/closed_loop_demo.yml`, `tests/test_closed_loop_config.py`.)
- **Track A (RTSfreerun A1→A4) — DONE and shipped as example 07.** A1 wiring, A2 open-loop SISO,
  A3 closed-loop (all six real L1-MC2 dampers; open-loop plant recovered <0.1% diagonal), A4
  open-loop 6×6 cross-coupling tensor vs the SS oracle — on the compiled `x1hsts`/`x1hsts6dof`
  numerics under the twin's own seismic+readout noise. (Results log below; `experiments/rtsfreerun/`.)
- **DARM calibration via P&S — example 08** (closed-loop twin: sensing C, 3-stage actuation, R(f)+CRB).
- **Methods page** `tutorial/why-optimal-excitation.qmd` (real head-to-head vs broadband/swept;
  honest phasing/DAC-frame treatment).
- Docs audit (`notes/docs-audit-2026-06-20.md`) + landing-page honesty fixes + reference
  pre-render hook; noise estimator relabeled period-variance (not "LPM").
- Quarto docs + GitHub Pages CI.

**Phase context:** CDS is two phases — **Phase 1 = RTSfreerun (current)**, Phase 2 = real hardware
(pyepics/pyawg/cdsutils), **not to be worked or discussed until the user says so**. Everything below
is Phase 1.

---

## Audit — through the SEI / ISC commissioning lens (2026-06-21)

> **Guiding light:** what must a **SEI** (seismic isolation / suspension damping) or **ISC**
> (length/angular sensing & control) commissioning engineer believe before trusting this on *their*
> subsystem? Findings tied to repo evidence; direction to be set by the interview that follows.

**Solid / already demonstrated (don't re-chase):**
- The pipeline runs against the **real compiled HSTS front-end numerics**, not a Python toy: it
  recovers the open-loop drive→sensor plant under the twin's own `ligo-india` seismic + BOSEM
  readout noise (A2), **through the production L1-MC2 damping loops closed on all six DOFs** (A3,
  controller cancelled, <0.1% on the diagonal incl. pitch), and the open-loop **6×6 cross-coupling
  tensor** matches the state-space oracle (A4, L↔P / R↔Y ~0.1–0.2%). This is a direct **SEI
  suspension-damping** result.
- ISC-relevant plants exist already: Fabry–Pérot cavity (ex 03), DARM through its servo (ex 08),
  2×2 coupled (ex 06).
- Honest uncertainty (CRB + per-bin period-variance noise model), closed-loop cancellation
  first-class, drive-safety watchdog, leakage-free vs broadband/swept shown with real numbers.

**Gaps they will hit first (prioritized):**

1. **No joint MIMO identification — [headline].** A real suspension (SEI) and ASC / multi-cavity
   (ISC) are *strongly coupled, and the coupling is what they tune*. Today `SysIDLoop` is strictly
   SISO-per-DoF (`loop.py:144` loop over `dofs`, per-DoF `models`/`info`); the coupled matrix is
   recovered only **nonparametrically per-pair** (ex 06, A4) — **no joint fit with shared modal
   poles, no MIMO CRB, no MIMO estimator** (`estimators/` has none). A commissioner cannot get a
   coupled *model* with per-element uncertainty. (This is Track B; A4's joint fit depends on it.)
2. **The real operating point — coupled AND in-loop — is unrepresentable even in simulation.**
   `TwinBackend` forbids coupling together with controllers (`twin.py:113`, deliberate guard). A3
   (in-loop) and A4 (coupled) were demonstrated *separately*; the **closed-loop off-diagonal
   coupling** — exactly what a damped, cross-coupled suspension or an ASC loop presents — is, in this
   roadmap's own words, "the genuine MIMO/Track-B story," and is **not** identified. (The compiled
   rtsfreerun model can do both at once; the generic twin + loop cannot.)
3. **Generality beyond what's shown.** SEI is more than one suspension — the **ISI/HEPI seismic
   platforms** (sensor-correction, blend filters) aren't demonstrated; ISC's **coupled ASC** isn't.
   The method is plant-agnostic, but a commissioner wants to see *their* plant. *Which* to add is an
   interview question.
4. **Twin validation is real but not CI-reproducible.** The 13 RTSfreerun tests pass against the
   compiled model where it's built (the twin box), but are skipif-gated out of CI
   (`test_rtsfreerun_6dof.py:30`) and ex 07 is `freeze`. Honest (the model can't live in public CI),
   but a third party reproduces only by building the model into their env.
5. **Track C (refinement-efficiency bake-off) still unwritten** — `refinement_sweep.py` exists, no
   verdict.

**Out of scope now:** real hardware / CDS (Phase 2); absolute-scale, leakage, SISO closed-loop
recovery, and the noise model (all already demonstrated).

---

## Track A — RTSfreerun twin-box demos 🔵 twin-box · *DONE (A1–A4; shipped as example 07)*

> **Goal.** Run the existing P&S pipeline against the *compiled* digital-twin suspension models under
> the twin's own realistic seismic + readout noise, in four escalating iterations, and confirm we
> recover the known plant each time.
>
> **Why.** This is the "CDS-aware" demonstration: identify a real front-end's drive→sensor plant with
> the *same C numerics a real front-end runs*, no hardware. It's the bridge between the twin-validated
> simulation and the eventual control-room hardware backend.
>
> **Status / prereqs.** Adapter + mock + config/CLI all shipped (see snapshot). What's left is purely
> *running it on the twin box*. Prereqs on that box: `twin` env with a built model
> (`model=x1hsts pip install ./rtsfreerun`, see [digital-twin.md](digital-twin.md)), and
> `pip install -e <system_ident>` into that env so `system_ident run … --rtsfreerun` resolves.
> **Get exact channel names from the model, not from memory:** read
> `digital_twin/twin/scenarios/hsts.yaml` (and `hsts_damped.yaml`, the 6dof scenario) and confirm the
> `_EXC`/`_OUT` port names against the built `.mdl`. The demo config
> `src/system_ident/configs/rtsfreerun_demo.yml` is a **skeleton** — its names came from `hsts.yaml`;
> verify before trusting. `mdl.sample_rate` (≈16384 Hz) is the source of truth for rate; the adapter
> decimates to `measurement.fs`.

**The four iterations** (each is a separate refinement pass; fill in its Results stub as you go):

### A1 — Wiring smoke 🔵
- **Steps.** Smallest possible round-trip on `x1hsts` (or `x1sysexample`): build the model, inject a
  short multisine at one `_EXC`, `read` the matching `_OUT`, confirm (a) output length is exactly
  `nperseg*n_periods` after decimation, (b) the injected lines show up at the right bins, (c) rate
  handling (16384 → `measurement.fs`) is integer-decimation clean. This mirrors
  `tests/test_rtsfreerun_backend.py::test_inject_read_frf_recovers_plant` but on the real `mdl`.
- **Verify.** A 1-DoF script or a trimmed config; assert the FRF input/output bins line up. No
  plant-recovery accuracy bar yet — just plumbing.
- **DoD.** Round-trip works on the real model; rates and lengths are exact; noise off.

### A2 — Open-loop SISO suspension 🔵
- **Steps.** `x1hsts`, one DoF. Inject the multisine at the actuator (`COIL_DRIVER_EXC`), read the
  sensor (`READOUT_NOISE_OUT`), with the twin's noise **on**: `seismic ligo-india` → `ISI_RESIDUAL_EXC`
  and `bosem` → `READOUT_NOISE_EXC` (already in `rtsfreerun_demo.yml`). FRF input X = the
  after-actuator drive monitor (`COIL_DRIVER_OUT`, the `drive:` channel). Recover the drive→sensor
  plant and compare to the analytic `HSTS_DRV_TF_A/B` the scenario was built from.
- **Verify.** `system_ident run src/system_ident/configs/rtsfreerun_demo.yml --rtsfreerun --yes`
  (start with `max_iter` small). Check recovered f0/Q/gain against the foton ZPK that defines
  `HSTS_DRV_TF` (load it via the twin's `foton_loader`/scenario). Tune `px_total` to stay under
  `COIL_DRIVER_LIMIT` (30000).
- **DoD.** f0/Q within tolerance of the analytic plant under realistic noise; uncertainty target met.

### A3 — Closed-loop 🔵
- **Steps.** `x1hstsdamped` / `hsts_damped.yaml`: engage the damping bank, then use the **first-class
  closed-loop mode** — set the `drive:` channel to the after-controller drive monitor and `injection_point`
  appropriately (see `backends/twin.py` closed-loop wiring + `configs/closed_loop_demo.yml` for the
  config shape). The reference-based FRF cancels the controller and recovers the **open-loop** plant.
- **Verify.** Same recovery check as A2, but loop closed; confirm the open-loop plant comes back
  (controller cancelled), not the suppressed closed-loop response.
- **DoD.** Open-loop plant recovered with the damping loop engaged.

### A4 — MIMO / coupled 🔵 (depends on Track B for the *joint fit*)
- **Steps.** `x1hsts6dof`: multi-DoF + cross-coupling campaign. Drive each input, form the full
  output×input FRF tensor, compare to the analytic suspension matrix. The nonparametric per-pair FRF
  works today (as `tests/test_twin_mimo.py` does by hand); the *parametric joint fit* needs Track B.
- **Verify.** Per-pair FRFs match the analytic `(out,in)` elements; once Track B lands, the joint
  matrix fit shares poles across elements.
- **DoD.** Coupled matrix identified on the twin and matched to the analytic plant.

> **Results log (twin box, 2026-06-18 — x1hsts built into the `sysid` env):**
> - **A1 — DONE.** Channel names confirmed against the built `.mdl`
>   (`COIL_DRIVER_EXC`/`_OUT`, `READOUT_NOISE_OUT`, `ISI_RESIDUAL_EXC`,
>   `READOUT_NOISE_EXC`). The analytic oracle (yaml ZPK, plane-`f`→`s` via −2π,
>   `backends/rtsfreerun_oracle`) agrees with the model's realized SOS to **1.6e-6**;
>   noise-off FRF through the adapter matches the oracle to **0.6% median** (coh
>   0.99996); exact length, clean ×64 decimation.
> - **A2 — DONE (P&S optimal excitation).** Open-loop SISO recovery under
>   `seismic ligo-india` + `bosem` via the `run_siso_passes` campaign (broad
>   prior-robust pass 1 from the perturbed prior → point-optimal refinement), the
>   SAME campaign the double-pendulum example uses. Fractional uncertainty falls
>   **0.228 → 0.046 → 0.034** over 3 passes; recovered plant vs oracle **~0.2% median
>   FRF err**; all five modes recovered to <0.2% in f0. Drive peak ~9–10k ≪ 30000.
>   Gated by `tests/test_rtsfreerun_real_model.py`; demo `experiments/rtsfreerun/run_hsts.py`.
> - A3 — _in progress, RE-SCOPED to `x1hsts6dof` + real L1-MC2 loops_ (2026-06-18).
>   Built `x1hstsdamped` (single-DOF) but it is **superseded**: its damping bank
>   (`MC1_M1_DAMP_L`) ships unconfigured and `hsts_damped.yaml` `init:` never sets it.
>   Per user ("use the real filters, close the loops on the full 6DOF — not
>   half-assed"), A3's real closed loop lives on **`x1hsts6dof`** (already built),
>   driven exactly as the twin's canonical example
>   `digital_twin/twin/examples/sus_hsts_6dof/{lib.py,run_rtsfree.py}`. **Reproduced
>   in the `sysid` env** (probe `.claude/probe_6dof_damping.py`): all 6 real **L1 MC2**
>   dampers (`SUS-MC2_M1_DAMP_<dof>` from `aligo_filter_files/l1/L1SUSMC2.txt`, FM list
>   from the L1 SDF, per-DOF CAL on free slot **FM1**) close stably and damp (closed/
>   open RMS 0.75–0.87 broadband). **A3-core proven** (`.claude/probe_a3_core.py`):
>   with loops CLOSED, reference-based FRF `READOUT_L /(u+MC2_M1_DAMP_L_OUT)` recovers
>   the **open-loop** plant (median 5.6% vs 3.8% open, crude single-pass) → controller
>   cancelled; the p90 tail (~39%) is **MIMO cross-coupling bias** (the real A4 story).
> - **A4 — works (open-loop MIMO tensor).** On `x1hsts6dof` (real plant via
>   `lib.load_plant_continuous`→`ss_set_abcd`), the leakage-free reference-FRF tensor
>   `READOUT_i/PLANT_IN_j` (2-pass accumulated) matches the SS oracle
>   (`orc.state_space_frf`): **diagonal anti-resonances <0.1%**, dominant physical
>   couplings **L↔P, R↔Y ~0.1–0.2%**; weak off-diagonals are noise-floor. Helper:
>   `experiments/rtsfreerun/hsts6dof_loop.py` (`measure_tensor`). Backend gained an
>   additive `plant_inputs=` virtual channel (drive + Σ feedback probes) so the
>   reference FRF works through a closed loop, `run_siso_passes` unchanged.
> - **A3 — DONE (closed-loop diagonal recovery, all six real loops closed).** With the
>   real L1-MC2 dampers engaged, the reference FRF `READOUT_d / PLANT_IN_d` recovers the
>   **open-loop** plant diagonal to **<0.1% (all six DOFs, incl. pitch)** — controller
>   cancelled. The A2-style optimal-excitation **parametric** campaign also recovers it
>   (L: frac 0.045→0.012, median 0.15%/peak 1.6%, all 5 modes; R median 0.14%).
>   **Root-cause of the earlier "peaks 50–110% off" dead-end: a SIGN BUG** in the
>   plant-input reconstruction. The composite's `COIL_DRV_SUM_<d>` is a **`"+-"`** sum
>   (drive **minus** the delayed `MC2_M1_DAMP_<d>` feedback, per `gen_x1hsts6dof.py`);
>   I had reconstructed `drive + damp_out`. Wrong sign is negligible off-resonance but
>   dominates at the damped modes — looked like an ill-conditioning/coupling wall, was
>   one sign. NOT flat-vs-optimal, NOT a delay (LOOP_DELAY 1-cycle, negligible), NOT
>   Track B. Backend `plant_inputs` gained `feedback_coeff` (−1 here).
> - **A4 — DONE (open-loop MIMO tensor).** Diagonal anti-resonances <0.1%, dominant
>   couplings L↔P, R↔Y ~0.1–0.2% vs the SS oracle. Off-diagonal closed-loop coupling is
>   the genuine MIMO/Track-B story; the per-pair tensor (spec A4) is the open-loop job.
>
> **Key findings (read before A3/A4):**
> - The HSTS **drive→sensor plant is the order-10 `HSTS_DRV_TF_A` cascade** (FM1–FM5;
>   `HSTS_DRV_TF_B`=identity): **5 modes at 0.67/1.01/1.52/2.81/3.78 Hz, all Q≈50**,
>   with interleaved near-cancelling zeros. The old skeleton prior (0.9 Hz/Q 10) was
>   **wrong** — corrected in `configs/rtsfreerun_hsts.yml`.
> - The **bare model has no filters**; the scenario `init:` must be applied
>   (`apply_scenario_init`, replicating `orchestrator._apply_init`). The adapter now
>   takes a `rtsfreerun.scenario:` path and applies it.
> - **We do P&S optimal excitation — NOT flat.** The order-10 plant's near-cancelling
>   pole/zero pairs make the Fisher **rank-deficient**, which made `dispersion()` /
>   `inv(info)` blow up. Fix: `fisher.safe_inverse` (pinv fallback **only** when the
>   Fisher is singular; full-rank still inverts exactly, so the sysIDlib bit-for-bit
>   tests are untouched). With that, prior-robust→optimal excitation + CRB run cleanly
>   and refine pass over pass. **Do not regress to flat excitation.**
> - `run_siso_passes` gained an `x_ch=` arg (FRF input = the after-actuator **drive
>   monitor** `COIL_DRIVER_OUT`, distinct from the injection `COIL_DRIVER_EXC`) — the
>   reference-based estimate that drops the coil driver / a closed loop out of the
>   recovered plant. The double-pend call (no `x_ch`) is unchanged.
> - One model per process: clear filter history (`mdl.fm_clear_history(*modules)`)
>   before each rung/campaign, else carryover biases the next measurement.

**Blocker:** 🔵 twin-box (built rtsfreerun model). Adapter/config/CLI are done.

---

## Track B — MIMO joint identification 🟢 local · *new feature*

> **Goal.** Identify the *coupled* suspension matrix — fit the full output×input transfer matrix with
> shared normal-mode poles and the off-diagonal cross-coupling — instead of independent SISO fits.
>
> **Why.** The forward model is already coupled (`plant.coupled_suspension`, twin off-diagonal paths),
> but the identification path throws the coupling away. Real suspensions cross-couple; recovering the
> matrix (and its shared modes) is the honest MIMO result. Needs **no hardware** — the twin already
> emits correct coupled data, so the test oracle exists.
>
> **Status / prereqs.** Absent. Today `SysIDLoop.run` is strictly SISO-per-DoF: it iterates
> `for d in dofs` and fits one `TFModel` per DoF (`loop.py:122-124`, `loop.py:157-178`,
> `loop.py:218-222`), with per-DoF `accum`/`info` dicts. The estimator interface is SISO
> (`estimators/base.py`: `fit(freq, H_meas, H_err, model) -> TFModel`). The nonparametric per-pair FRF
> already works via the static `SysIDLoop._estimate_tf_periodic` (used four times by hand in
> `tests/test_twin_mimo.py`). Config can't even *enable* coupling simulation:
> `config.build_twin_backend` never passes `coupling=` (`config.py` ~144-160). And coupling is
> mutually exclusive with closed-loop controllers (`twin.py:111-115`, deliberate guard) — so plan MIMO
> as **open-loop only** for now.

- **Steps.**
  1. **Config plumbing** — let `config.build_twin_backend` pass a `coupling:` dict through to
     `TwinBackend` so a MIMO twin is declarable (today it's reachable only by constructing
     `TwinBackend` in code, as the test does).
  2. **Loop MIMO mode** — add a path that, per pass, drives each input DoF and forms the full
     `(out, in)` FRF tensor by reusing `_estimate_tf_periodic` on every channel pair; accumulate
     inverse-variance per element across passes (generalize the per-DoF `accum`).
  3. **Joint estimator** — fit the matrix with **shared poles** across elements (modal expansion:
     `plant.coupled_suspension`, `plant.py:76-127`, esp. the shared normal-mode poles ~86-97). Each
     element shares the denominator; numerators (residues) differ. Likely a new estimator alongside
     `GMLEstimator` rather than overloading the SISO `fit` signature.
  4. **MIMO Fisher/CRB** — extend `fisher_matrix` to the matrix case so uncertainty is reported per
     element / per shared pole.
- **Verify.** New tests against a `coupling=`-enabled `TwinBackend` (oracle = the analytic coupled
  matrix). Recover both diagonal anti-resonance notches and the notch-free off-diagonals; shared poles
  must come back common across elements. `conda run -n sysid python -m pytest -q` stays green.
- **DoD.** A config-declarable coupled twin is identified jointly; recovered matrix + shared poles
  match the analytic plant within tolerance; feeds Track A4.

**Blocker:** 🟢 local (twin simulation suffices). Substantial new code, not a stub fill-in.

---

## Track C — Refinement-efficiency bake-off 🟢 local · *closes a loose end*

> **Goal.** Quantify *when the recursive Bayesian-MAP refinement actually pays off* vs the
> prior-ignoring `broadband_ls` mode — the one open thread left from the prior bake-off.
>
> **Why.** The prior bake-off already concluded (`experiments/prior_bakeoff/FINDINGS.md`): local prior
> tweaks do **not** solve cold-start (0/7 across 2142 campaigns), and `broadband_ls` solves cold-start
> prior-independently. FINDINGS explicitly flags the *refinement-efficiency* comparison as "the
> meaningful next bake-off." `refinement_sweep.py` is that follow-on but has no written verdict yet.
>
> **Status / prereqs.** `experiments/prior_bakeoff/refinement_sweep.py` exists and runs the **real**
> `SysIDLoop` with a *good* prior (±5–20%) at low SNR, measuring passes-to-target for `broadband_ls`
> vs `bayesian` (at several prior strengths) vs `hybrid`. Just needs running + writing up. (Note: some
> estimator names in that experiment — `bayesian`/`hybrid` — predate estimator-set changes; reconcile
> against the current `estimators/` registry before running, and adjust if removed.)

- **Steps.** Reconcile estimator names with the current registry; run the sweep in `sysid`; tabulate
  passes-to-target by mode × prior strength × SNR; append a verdict section to `FINDINGS.md`.
- **Verify.** `conda run -n sysid python experiments/prior_bakeoff/refinement_sweep.py` produces a
  clean table; conclusion is reproducible across seeds.
- **DoD.** `FINDINGS.md` gains a written refinement-efficiency verdict (when/whether Bayesian beats
  broadband_ls, and at what prior strength), closing the bake-off.

**Blocker:** 🟢 local.

---

## Ordering & parallelism

1. **Now, on the twin box:** Track A in order (A1 → A2 → A3 → A4). This is the user's stated priority.
2. **In parallel, anywhere (`sysid`):** Track B (unblocks A4's joint fit), Track C (bake-off).

Keep this file current — tick DoDs, fill the A-results log, and note any channel-name corrections
discovered against the real models.
