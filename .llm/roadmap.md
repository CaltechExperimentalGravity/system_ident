# system_ident — master execution roadmap

The single execution-ready roadmap for what's left. Written so a **fresh Claude session on another
machine** (e.g. the Linux/twin box) can pick up any track and run it without re-deriving context.
Companion to [overview.md](overview.md), [digital-twin.md](digital-twin.md),
[rtsfreerun-integration.md](rtsfreerun-integration.md). The broader project log is
`.claude/NOTES.md`; the live 40m SOS deployment plan is `notes/40m-sos-campaign-handoff-2026-07.md`.

---

## How to use this file (read first)

**Environment & guardrails — non-negotiable:**
- Run **everything** through `conda run -n sysid …`. Never invoke Python outside the `sysid` env.
- **If a library is missing or broken, stop and ask the user.** Do not hunt for substitutes or
  improvise an alternative. (Standing user instruction.)
- **FRF excitation is ALWAYS the Pintelon–Schoukens multisine** — `excitation.multisine_from_psd`.
  Gaussian noise (`excitation.timeseries_from_asd`) is **background/disturbance only**, never the
  identification drive.
- **Never hand-roll what python-control provides** (state-space, `c2d`, FRF, `feedback`, `tf`/`zpk`).
- The grid constraint everywhere: **`T_fft = nperseg/fs > 3/f_min`**; size a campaign from the prior
  with `design.resolution.recommend_resolution`, not by guessing. Drop the first `n_transient`
  period(s) as settling.
- Before calling anything a limit: **compute the CRB / SNR / actuator headroom first** (`CLAUDE.md`
  feasibility gate, `.llm/engineering-practices.md`).

**Pick a track by its blocker tag:**
- 🟢 **local** — runnable now in `sysid` on any machine.
- 🔵 **twin-box** — needs a built rtsfreerun model (Linux box).
- 🔴 **gated** — needs operator answers / Phase-2 authorisation. Do not start.

**Per-track shape** (every track below has all seven): Goal · Why · Status/prereqs · Steps ·
Verify · Definition of done · Blocker.

---

## Status snapshot (2026-08-11)

**Method: done and demonstrated.** The science half of this repo is in good shape and should not be
re-chased:

- P&S SISO pipeline: prior-robust → point-optimal multisine design, leakage-free periodic-DFT FRF
  (ratio-of-averages), ML fit (`GMLEstimator`), Fisher/CRB, safety watchdog + safe handoff,
  live dashboard, CLI + YAML config.
- **Closed-loop is first-class** — config-declarable injection point + drive-monitor channel
  recovers the open-loop plant (`configs/closed_loop_demo.yml`, verified end-to-end this session).
- **Track A (RTSfreerun A1→A4) — DONE**, on the compiled `x1hsts`/`x1hsts6dof` numerics under the
  twin's own seismic + BOSEM noise, through the production L1-MC2 damping loops (example 07).
- **Rank-1 modal MIMO joint fit — DONE as a library** (`mimo_modal`, `mimo_fit`, `mimo_campaign`,
  `mimo_iterate`): shared modal poles, per-mode rank-1 residues, SML/IQML fit, MIMO Fisher/CRB,
  block-decoupled fits for spatial doublets, data-driven order selection (`find_modes`). Proven on
  the SRM 6-DoF (example 10), reduced quad open + closed (11, 12), and the analytic 40m SOS
  (`sos_campaign`, recovery within the CRB).
- **DARM calibration via P&S** — examples 08 and 13 (sensing + hierarchical actuation, cal-line
  Fisher design, drift tracking), with the `provenance` gate against shipping invented numbers.
- 40m SOS deployment: **Stage 0 and Stage 1 DONE** (analytic 6-DoF SOS plant + OSEM projection
  layer + pyctl MIMO recovery within CRB). Next is Stage 2 (rtsfree composite) — see
  `notes/40m-sos-campaign-handoff-2026-07.md`.

**Phase context:** CDS is two phases — **Phase 1 = RTSfreerun / in-process twins (current)**,
Phase 2 = real hardware (awg/nds2/pyepics/cdsutils), **not to be worked or discussed until the user
says so**. Everything below except Track L Stage 5 is Phase 1.

---

## Audit — code functionality (2026-08-11)

> **Lens:** not "is the method right" (it is, and it is gated by tests) but **"can anyone other than
> the author drive it?"** Every finding below was checked against the code this session; file:line
> references are the evidence.

### What is actually wired end-to-end

`system_ident run <config.yml> --twin | --rtsfreerun` runs one stack: YAML → `RunConfig` →
`TwinBackend`/`RTSfreerunBackend` + `GMLEstimator` + `PintelonSchoukensDesigner` + `Watchdog` →
`SysIDLoop.run` → per-DoF `TFModel` + fractional uncertainty → (optional) live dashboard.
Verified this session: `configs/twin_demo.yml` reaches the target in 2 DoF; `configs/closed_loop_demo.yml`
recovers the open-loop plant through a closed loop. (But see finding 9 — with those configs'
defaults the *printed uncertainty* is not a measured quantity.)

### Reachability map — the headline

The package is **three estimation stacks and several application stacks; exactly one is drivable
from the config/CLI/dashboard.**

| Stack | Core modules | Config/CLI | `SysIDLoop` + watchdog | Dashboard |
|---|---|---|---|---|
| SISO polynomial (`TFModel`) | `loop`, `config`, `cli`, `fisher`, `design/pintelon`, `estimators/gml`, `backends/{twin,rtsfreerun_adapter}` | ✅ | ✅ | ✅ |
| SISO physical (`ResonatorModel`) | `resonator`, `resonator_design` | ❌ | ❌ | ❌ |
| MIMO rank-1 modal | `mimo_modal`, `mimo_fit`, `mimo_campaign`, `mimo_iterate`, `mimo_loop`, `backends/{mimo_twin,reduced}` | ❌ | ❌ | ❌ |
| Applications | `sos_campaign`, `sos_closed`, `darm*`, `reduced_plant`, `closed_loop_id`, `osem` | ❌ | ❌ | ❌ |

Everything outside row 1 is reachable only by writing a bespoke Python script. That is why
`README.md` tells a user to run `python experiments/rtsfreerun/run_hsts.py` for the flagship demo
instead of the CLI.

### Findings

1. **The campaign wrapper — the thing every result in this repo actually uses — is not in the
   package.** `run_siso_passes` / `param_sigmas` / `frac_history` / `physical_value` live in
   `docs/sysid_campaign.py:28`. Nine test files `sys.path.insert(…/"docs")` to import it
   (`tests/test_rtsfreerun_real_model.py:31`, `tests/test_reduced_quad.py:17`, …), as do
   `experiments/rtsfreerun/*` and every example page. **An installed wheel cannot reproduce any
   headline result.** Scale: 4423 LOC in `docs/*.py` + 3090 LOC in `experiments/` against 8143 LOC
   of package — roughly half the working code is outside the shipped artifact.
2. **The MIMO stack cannot be driven by the orchestrated loop, even in principle.**
   `MIMOTwinBackend` and `ReducedPlantBackend` do not override `snapshot_state`/`restore_state`
   (verified by introspection), so `Watchdog.snapshot()` — the first call in `SysIDLoop.run`
   (`loop.py:140`) — raises `NotImplementedError` from `backends/base.py:64`. The safe-handoff
   contract is satisfied by 2 of 4 usable backends.
3. **No run produces an artifact.** Nothing under `src/` writes a result file. `LoopResult.models`
   (the fitted plants) is discarded by `cli._report`, which prints one float per DoF
   (`cli.py:147-159`). There is no saved FRF, covariance/CRB, designed drive, config hash, git SHA,
   seed record, or provenance dump — so nothing is re-analysable, comparable across runs, or
   deliverable to a commissioner.
4. **The CLI is one verb with one meaningful knob.** Only `run` exists. `--estimator` selects
   between `"gml"` and `"ml"`, which are two names for the *same* class (`config.py:31-34`), so it
   is effectively a no-op. There is no way to preview a design, compute a CRB, lint a config,
   re-fit saved data, or select a backend other than the two booleans. `cli.py:89` calls
   `_confirm(twin=True)` unconditionally, so an `--rtsfreerun` run also prints "about to inject on
   the twin".
5. **Half the twin's realism is unreachable from YAML.** `TwinBackend` implements `coupling=`,
   `saturate=`, and `response_delay_samples=`, but `config.build_twin_backend` never passes any of
   them (`config.py:144-160`) — so MIMO cross-coupling, actuator saturation, and transport delay
   can only be exercised by constructing the backend in Python. (Note the deliberate guard at
   `twin.py:110-115`: coupling and controllers are mutually exclusive, so the *real* operating
   point — coupled **and** in-loop — is not representable in the generic twin at all.)
6. **Config validation is shallow and unknown keys are silent.** `REQUIRED` (`config.py:37-44`)
   checks section presence only. `strategy.prior_uncertainty` (`loop.py:119`),
   `strategy.n_design_iter`, `measurement.n_transient` (`loop.py:109`), `measurement.n_segments`,
   `measurement.t_ramp`, `run.excitation_mode` are all read by the loop/backends but are neither
   required, validated, nor present in the shipped configs — a typo silently reverts to a default
   and the run looks fine. There is no schema version.
7. **`configs/three_dof_twin.yml` cannot be run by any flag** — it has no `twin.plant`, so
   `--twin` fails with `config error: twin runs need a 'twin.plant' resonance spec`, and it has no
   `rtsfreerun` section either. It ships inside the wheel (`pyproject.toml` package-data).
8. **Three parallel Fisher/design implementations, one loop that only accepts one of them.**
   `fisher.dispersion` + `design/pintelon.optimal_excitation` (TFModel, gauge-fixed) vs
   `resonator_design.{dispersion,optimal_excitation,prior_robust_excitation}` (physical, gauge-free)
   vs `mimo_fit.mimo_fisher_matrix` (modal SML). `SysIDLoop._frac_uncertainty` hard-codes
   `model.n_num` (`loop.py:475`), which `ResonatorModel` does not have — so the orchestrated loop is
   locked to the *coefficient* parameterisation that is documented to fail at 6-DoF scale
   (`.claude/memory/rank1-modal-mimo-fit.md`).
9. **Period bookkeeping: the reported uncertainty can be fiction, and the shipped demo config is a
   case of it.** Three independent trims reduce the periods actually averaged — the ramp-taper trim
   (`loop.py:419-424`), the adaptive transient drop `_choose_transient` (`loop.py:428`, which can
   drop *more* than the configured `n_transient`), and the configured drop itself — but
   `_fisher_time_factor` (`loop.py:110`) is computed from the **configured** `n_transient` alone.
   Reproduced with `configs/twin_demo.yml`'s own numbers (`n_segments: 4`, `t_ramp: 3`, 64 s
   periods): 4 periods → 2 survive the ramp trim → 1 survives the transient drop. With
   `P_eff == 1` the period-variance estimator has no scatter to measure, so `var_H = 0` and `H_err`
   collapses to its `1e-9·|H|` floor (`loop.py:460`) — weights ~1e18 on every excited bin — while
   the Fisher is evaluated for **192 s of data when 64 s was averaged**. The demo's printed
   "max fractional uncertainty = 3.5e-4" is therefore not a measured uncertainty. There is no guard:
   `_estimate_tf_periodic` raises only if the *raw* period count is < 2, never if the *effective*
   count falls to 1.
10. **The dashboard is SISO-only and needs the public internet.** `render_html` loads Plotly from
    `https://cdn.plot.ly/...` (`dashboard/server.py:63`) — an air-gapped control-room machine
    renders a blank page. The snapshot schema carries one DoF, one model curve, one coherence trace
    (`loop._emit`, `dashboard/ws.SNAPSHOT_FIELDS`): no CRB band, no per-DoF view, no MIMO matrix, no
    drive-headroom gauge, no run metadata, and nothing is persisted.
11. **Drive safety is post-hoc only.** `Watchdog.check` inspects peak drive and output RMS *after* a
    segment has been injected (`safety.py:96-140`). There is no pre-injection, DAC-frame,
    filter-aware worst-case-peak check — the one thing that matters before a first injection on
    anything expensive. (Already identified as a Stage-4 bridge piece in the 40m handoff note.)
12. **Stale pointers.** `.llm/rtsfreerun-integration.md` (and, before this rewrite, this file)
    referenced `src/system_ident/configs/rtsfreerun_demo.yml`, which does not exist — the real file
    is `configs/rtsfreerun_hsts.yml`. Fixed in this pass.
13. **The A3/A4 validation gate does not run in a full-suite run — it errors.** On a machine with
    `x1hsts6dof` built, `conda run -n sysid python -m pytest -q` gives **316 passed, 2 skipped,
    8 errors**; the 8 are every test in `tests/test_rtsfreerun_6dof.py`. They **pass in isolation**
    (`pytest tests/test_rtsfreerun_6dof.py` → 8 passed). Root cause, reproduced directly:
    rtsfreerun allows **one model instance per process**, and `x1hsts6dof` is constructed
    independently by two test modules — `test_reduced_vs_compiled_twin.py` → `srm6dof_loop.SRM6DOF`
    (which sorts earlier) and `test_rtsfreerun_6dof.py` → `hsts6dof_loop.HSTS6DOF`. The second
    construction returns a dead model: `AttributeError: 'NoneType' object has no attribute
    'get_model_rate_Hz'` (`x1hsts6dof/model.py:67`). `conftest.py` already solves exactly this for
    `x1hsts` with a session-scoped `x1hsts_model` fixture and a docstring saying "a second
    `x1hsts.x1hsts()` fails" — the 6dof path never got the same treatment. **Effect: the gate
    protecting the headline Track A closed-loop result silently does not execute in the run everyone
    treats as authoritative.** Fix: a session-scoped `x1hsts6dof_model` fixture, with `SRM6DOF` /
    `HSTS6DOF` taking an injected `mdl=` instead of constructing their own.

**Not defects (deliberate, do not "fix"):** `backends/cds.py` raising `NotImplementedError` is the
Phase-2 gate; `experiments/` scripts being one-off is fine *once* the reusable part is in the
package; `docs/sysid_plots.py` is presentation-only and belongs in `docs/`.

---

## Track A — RTSfreerun twin-box demos 🔵 · *DONE (A1–A4; shipped as example 07)*

Kept for its results log and hard-won channel/wiring facts. Nothing to do.

> **Results (twin box, 2026-06-18):**
> - **A1.** Channel names confirmed against the built `.mdl` (`COIL_DRIVER_EXC`/`_OUT`,
>   `READOUT_NOISE_OUT`, `ISI_RESIDUAL_EXC`, `READOUT_NOISE_EXC`). The analytic oracle (yaml ZPK,
>   plane-`f`→`s` via −2π, `backends/rtsfreerun_oracle`) agrees with the model's realized SOS to
>   **1.6e-6**; noise-off FRF matches the oracle to **0.6% median**; exact length, clean ×64
>   decimation.
> - **A2.** Open-loop SISO under `seismic ligo-india` + `bosem` via the `run_siso_passes` campaign:
>   fractional uncertainty **0.228 → 0.046 → 0.034** over 3 passes; **~0.2% median** FRF error vs the
>   oracle; all five modes to <0.2% in f0. Drive peak ~9–10k ≪ the 30000-count coil limit.
> - **A3.** With the real L1-MC2 dampers engaged on all six DOFs, the reference FRF
>   `READOUT_d / PLANT_IN_d` recovers the **open-loop** plant diagonal to **<0.1%** (incl. pitch).
>   The earlier "peaks 50–110% off" wall was a **SIGN BUG** in the plant-input reconstruction:
>   `COIL_DRV_SUM_<d>` is a `"+-"` sum, so `X = drive − damp_out`. Backend `plant_inputs` gained
>   `feedback_coeff` (−1 here). **LESSON: when every method fails identically at the resonances,
>   suspect a shared reconstruction/sign bug before indicting the method.**
> - **A4.** Open-loop 6×6 tensor vs the SS oracle: diagonal anti-resonances <0.1%, dominant physical
>   couplings L↔P and R↔Y ~0.1–0.2%.
>
> **Key facts to re-read before touching the HSTS models:** the drive→sensor plant is the order-10
> `HSTS_DRV_TF_A` cascade (5 modes at 0.67/1.01/1.52/2.81/3.78 Hz, all Q≈50, interleaved
> near-cancelling zeros); the bare model has **no filters** until the scenario `init:` is applied;
> the near-cancelling pole/zero pairs make the Fisher rank-deficient, which is what
> `fisher.safe_inverse` exists for — **do not regress to flat excitation**; one model per process, so
> clear filter history (`mdl.fm_clear_history`) between rungs.

---

## Track B — MIMO joint identification 🟢 local · *math DONE; integration NOT*

> **Correction to the previous roadmap.** Track B was written as "Absent — substantial new code".
> The *estimator* has since landed and is proven: shared-pole rank-1 modal model
> (`mimo_modal.Rank1ModalModel`), SML/IQML fit with analytic Jacobian
> (`mimo_fit.MIMOModalEstimator`), MIMO Fisher/CRB + per-mode uncertainty
> (`mimo_fisher_matrix`, `modal_uncertainty`, `modal_frac_uncertainty`, `frf_band`), data-driven
> order selection (`find_modes`), block-decoupled fits for spatial doublets (`fit_block_decoupled`),
> the robust per-actuator campaign with covariance-of-the-mean (`mimo_campaign.assemble_campaign`),
> and an estimate→redesign→re-measure driver (`mimo_iterate.iterate_mimo`). Verified on the SRM
> 6-DoF, the reduced quad (open and closed loop), and the analytic 40m SOS within the CRB.

**What is left of Track B is not math — it is integration, and it is now Track I.** The remaining
gaps, precisely: no config schema can declare a MIMO run; `SysIDLoop` has no MIMO mode; two MIMO
backends fail the safe-handoff contract; `iterate_mimo` re-fits only the latest pass (no cross-pass
inverse-variance accumulation, unlike the SISO loop's `_accumulate`); and the coupled **and**
in-loop operating point is blocked in the generic twin by `twin.py:110-115`.

**Blocker:** 🟢 local. **Superseded by Track I.**

---

## Track C — Refinement-efficiency bake-off 🟢 local · *still open, still unwritten*

> **Goal.** Quantify when recursive refinement actually pays off vs the prior-ignoring
> `broadband_ls` mode — the one open thread from the prior bake-off.
>
> **Why.** `experiments/prior_bakeoff/FINDINGS.md` concluded that local prior tweaks do not solve
> cold-start (0/7 across 2142 campaigns) and flagged refinement-efficiency as "the meaningful next
> bake-off". `refinement_sweep.py` is that follow-on and has no verdict.
>
> **Status / prereqs.** Script exists. Its `bayesian`/`hybrid` estimator names **predate the
> estimator-set cull** — the registry is now `{gml, ml} → GMLEstimator` only (`config.py:31`), so
> reconcile before running or it will fail at import/lookup.

- **Steps.** Reconcile estimator names against the current registry; run the sweep in `sysid`;
  tabulate passes-to-target by mode × prior strength × SNR; append a verdict to `FINDINGS.md`.
- **Verify.** `conda run -n sysid python experiments/prior_bakeoff/refinement_sweep.py` produces a
  clean table; the conclusion is stable across seeds.
- **DoD.** `FINDINGS.md` gains a written refinement-efficiency verdict, closing the bake-off.

**Blocker:** 🟢 local.

---

# Future work — the product half

> These are the tracks the audit says matter now: **sysID core, the interface, and the wrappers.**
> Docs are explicitly *not* the priority — where a track touches docs it is only to stop the docs
> from being the load-bearing code.

## Track D — Promote the campaign wrapper into the package 🟢 local · *highest leverage*

> **Goal.** `from system_ident.campaign import run_siso_passes` works from a plain `pip install`,
> and `docs/`, `experiments/`, and `tests/` all import it from there.
>
> **Why.** Audit finding 1: the multi-pass P&S refinement driver behind every headline result lives
> in `docs/sysid_campaign.py` and is reached by `sys.path` surgery from nine test files. That makes
> the docs directory load-bearing library code, makes the wheel unable to reproduce its own results,
> and blocks Tracks E/F/I (nothing can build on a wrapper that is not importable).
>
> **Status / prereqs.** None — pure refactor, no new math. `docs/sysid_campaign.py` already reuses
> library internals verbatim (`SysIDLoop._estimate_tf_periodic`, `_accumulate`, `_frac_uncertainty`,
> `fisher_matrix`, `optimal_excitation`/`prior_robust_excitation`), so this is a move, not a rewrite.

- **Steps.**
  1. New `src/system_ident/campaign.py`: `run_siso_passes`, `physical_value`, `frac_history`,
     `param_sigmas`, `param_convergence_series`. Import nothing from `docs/`.
  2. Move the small model helpers `sysid_campaign` currently re-exports from `sysid_plots`
     (`f0_q`, `modes`, `dc_gain`) into `system_ident.model` (or `resonator`), where they belong;
     `sysid_plots` imports them back for plotting.
  3. Promote the private loop internals the campaign depends on to a documented surface: make
     `estimate_tf_periodic`, `accumulate`, `frac_uncertainty` public module-level functions in
     `loop.py` with the underscore names kept as thin aliases (so nothing in `tests/` or the twin
     work breaks).
  4. `docs/sysid_campaign.py` becomes a two-line re-export shim (keeps every `.qmd` and the frozen
     `_freeze` cache working); delete the `sys.path.insert(…/"docs")` lines from the nine test files.
  5. Export the MIMO surface from `system_ident/__init__.py` too (`Rank1ModalModel`,
     `MIMOModalEstimator`, `assemble_campaign`, `find_modes`, `recover_open_loop`,
     `iterate_mimo`) — today `__init__` exports four names and none of the research surface.
- **Verify.** `conda run -n sysid python -m pytest -q` green with **no** test importing from
  `docs/`; `pip install .` into a scratch env, then `python -c "from system_ident.campaign import
  run_siso_passes"` and a 1-DoF campaign runs outside the repo tree.
- **DoD.** Zero `sys.path` hacks in `tests/`; the wheel reproduces a headline campaign; `docs/` holds
  only presentation code.

**Blocker:** 🟢 local.

## Track E — Run artifacts, manifest, and replay 🟢 local

> **Goal.** Every run writes a self-describing result directory that can be re-loaded, compared, and
> handed to someone else; a saved run can be re-fit offline without re-measuring.
>
> **Why.** Audit finding 3: today a campaign's entire output is one printed float per DoF and the
> fitted models are dropped on the floor. Nothing is reproducible after the process exits, nothing
> is comparable run-to-run, and a commissioner has nothing to take away. This is also the
> prerequisite for a delivered-fit manifest at hardware time.
>
> **Status / prereqs.** Track D (needs an importable campaign object to serialise). `provenance.py`
> already exists and is the right place to source the "where did this number come from" section.

- **Steps.**
  1. Define `RunArtifact` (dataclass + `save(dir)`/`load(dir)`): per-DoF fitted model (num/den **and**
     physical f0/Q/gain), parameter covariance + per-parameter σ, the measured FRF with per-bin
     errors and coherence, the designed excitation PSD per pass, `IterationRecord` history, the
     safety report, and the raw drive/response time series (optional, `--save-timeseries`).
  2. Manifest sidecar (JSON): config as loaded **plus** a content hash, git SHA + dirty flag,
     `system_ident.__version__`, env (numpy/scipy/control versions), seed, backend identity (for
     rtsfreerun: model name + scenario path), UTC timestamps, and the `provenance` registry dump.
  3. `--out DIR` on `run`; default `runs/<timestamp>-<config-name>/`. Bulk arrays as `.npz`,
     metadata as JSON — no pickle.
  4. `system_ident replay <run-dir>`: re-fit from the saved FRF with a different model order /
     parameterisation / estimator, without touching a backend. This is what makes an expensive
     hardware measurement worth keeping.
- **Verify.** New `tests/test_artifact.py`: run `twin_demo.yml` with `--out`, reload, assert the
  reloaded model reproduces the run's fractional uncertainty bit-for-bit and that `replay` on the
  saved FRF reproduces the same fit.
- **DoD.** `system_ident run … --out r/ && system_ident replay r/` round-trips; the manifest names
  every input needed to reproduce the run.

**Blocker:** 🟢 local.

## Track F — CLI verbs and operator UX 🟢 local

> **Goal.** A CLI that supports the actual workflow — *decide whether to measure, then measure, then
> read the result* — instead of only the middle step.
>
> **Why.** Audit finding 4. The repo's own `CLAUDE.md` feasibility gate says "compute the bound
> before you claim a limit", but there is no command that computes a bound. The advertised path to
> every non-trivial demo is `python experiments/…py`.
>
> **Status / prereqs.** Track D (campaign API) and Track E (artifacts) for `report`/`replay`.

- **Steps.**
  1. `system_ident design <config.yml>` — build the prior, design the drive, and print/plot the
     excitation ASD, the per-line drive amplitudes, the peak-vs-actuator-limit headroom, and the
     **predicted** CRB after N passes. No injection. This is the pre-flight the feasibility gate
     wants, and it is what tells an operator whether the run is worth doing.
  2. `system_ident crb <config.yml> [--snr …] [--duration …]` — the bound alone, as a table:
     σ(f0)/f0, σ(Q)/Q, σ(gain) per DoF vs drive amplitude and record length. Answers "what would it
     take?" with numbers.
  3. `system_ident validate <config.yml>` — strict schema lint (Track G), plus the physics checks:
     `T_fft > 3/f_min`, `recommend_resolution` vs the configured `nperseg`, band vs prior modes,
     drive budget vs `safety.actuator_sat`. Non-zero exit on failure so CI can gate configs.
  4. `system_ident report <run-dir>` — a standalone HTML/SVG summary from a Track-E artifact.
  5. `system_ident list-configs` / `--print-config` (fully resolved config after overrides).
  6. Fix the paper cuts: `_confirm` must name the real backend (`cli.py:89`); either give
     `--estimator` real choices or drop it (`config.py:31-34`); add `--max-iter`, `--dof`,
     `--n-periods` overrides; make `--seed` default to `None` (fresh entropy, recorded in the
     manifest) rather than a silent `0`.
  7. Delete or fill `configs/three_dof_twin.yml` (finding 7).
- **Verify.** Extend `tests/test_step8_cli.py`: every verb has an exit-code test; `validate` fails a
  deliberately under-resolved config and passes the shipped ones; `design` on `twin_demo.yml` prints
  a headroom number and injects nothing (assert the backend's `inject` was never called).
- **DoD.** `system_ident --help` shows `run design crb validate report replay list-configs`; a new
  user can go prior → predicted CRB → measurement → saved report without writing Python.

**Blocker:** 🟢 local.

## Track G — Config schema v2 + backend registry 🟢 local

> **Goal.** One declarative schema that can express every run this repo can actually perform, that
> rejects typos, and that selects a backend by name.
>
> **Why.** Audit findings 5, 6, 7: the twin's coupling/saturation/delay knobs are unreachable from
> YAML, silently-ignored keys make misconfiguration invisible, and backend choice is two booleans.
>
> **Status / prereqs.** Independent of D/E, but Track F's `validate` verb consumes it.

- **Steps.**
  1. **Strict validation.** Reject unknown keys with a "did you mean" hint; declare every key the
     loop/backends read, with types, units, defaults, and a one-line meaning — including the
     currently-undeclared `strategy.prior_uncertainty`, `strategy.n_design_iter`,
     `measurement.n_transient`, `measurement.n_segments`, `measurement.t_ramp`,
     `run.excitation_mode`. Add `schema_version` and fail loudly on a future version.
  2. **Backend registry.** `backend: {name: twin|mimo_twin|reduced|rtsfreerun|cds, …}` replacing the
     `--twin`/`--rtsfreerun` flags (keep the flags as deprecated aliases). Each backend registers a
     `from_config` and declares its required section.
  3. **Expose the twin's realism knobs**: `twin.coupling`, `twin.saturate`,
     `twin.response_delay_samples` → `build_twin_backend`. Then revisit the
     coupling-XOR-controllers guard (`twin.py:110-115`): the compiled model does both at once, and
     coupled-and-in-loop is the real operating point. Either lift the guard with a correct MIMO
     closed-loop solve (`mimo_loop.CoupledLoop` already does exactly this with python-control) or
     document why the generic twin will never do it and point at `MIMOTwinBackend`.
  4. **Model parameterisation as a config choice**: `strategy.model: tf | resonator | rank1_modal`
     (depends on Track H).
  5. **Campaign auto-sizing**: `measurement.auto_resolution: true` → call
     `design.resolution.recommend_resolution` on the prior modes and *derive* `nperseg`/`n_transient`,
     printing what it chose and why. Today that module exists and the shipped configs ignore it.
- **Verify.** `tests/test_config_schema.py`: a typo'd key raises with a suggestion; every shipped
  config validates; a `coupling:` config produces a twin whose off-diagonal FRF is non-zero; the
  registry builds all four in-process backends.
- **DoD.** Every capability in `backends/` and `loop.py` is declarable in YAML; no key is read that
  the schema does not declare.

**Blocker:** 🟢 local.

## Track H — One model protocol, one Fisher, one designer 🟢 local · *sysID core*

> **Goal.** `SysIDLoop`, `fisher`, and `design` work against a single model protocol, so the
> physical (`ResonatorModel`) and modal (`Rank1ModalModel`) parameterisations are first-class in the
> orchestrated loop rather than living in parallel copies.
>
> **Why.** Audit finding 8. There are three implementations of the same P&S math and the loop can
> only use the one that is documented to fail at 6-DoF (`memory/rank1-modal-mimo-fit.md`:
> "common-denominator B/A fails at 6-DoF — numerators absorb pole error"). Every future estimator
> currently costs a fourth copy of `dispersion`/`optimal_excitation`. This is the single change that
> makes the good estimators reachable from the product.
>
> **Status / prereqs.** The protocol already exists de facto — `resonator_design`'s docstring names
> it: `.params`, `.jacobian(freq)`, `.eval(freq)`, `.with_params(theta)`. What is missing is that
> the loop and `fisher.py` assume the TFModel gauge (`model.n_num`).

- **Steps.**
  1. Formalise `ModelProtocol` (typing `Protocol`) with `params`, `jacobian`, `eval`,
     `with_params`, plus **`gauge_indices`** (empty for gauge-free models, `[n_num]` for `TFModel`)
     and **`frac_uncertainty(cov)`**. Implement on `TFModel`, `ResonatorModel`, `Rank1ModalModel`.
  2. Collapse `resonator_design.py` into `fisher.py` + `design/pintelon.py` by dispatching on
     `gauge_indices` — keep `resonator_design` as a deprecation shim for one release. Assert
     bit-for-bit equality with the legacy `sysIDlib` oracle tests (`test_step2_validation.py`,
     `test_step3_validation.py`) before and after; those are the regression gate.
  3. Replace `SysIDLoop._frac_uncertainty`'s `model.n_num` with the protocol method (`loop.py:475`),
     so a `ResonatorModel` prior runs the loop unchanged.
  4. `config.build_priors` grows `strategy.model` (Track G step 4) and can emit a `ResonatorModel`.
  5. **Fix the period bookkeeping** (finding 9) — do this one first, it is a correctness bug, not a
     refactor: have `_estimate_tf_periodic` return `P_eff` (periods actually averaged) and scale
     `T_tot` by that, not by the configured `n_transient`; **raise** when `P_eff < 2` (no noise
     estimate is possible) instead of silently returning the `1e-9` error floor; and either raise
     `configs/twin_demo.yml`'s `n_segments` or shrink its `t_ramp` so the shipped demo actually
     averages several periods. Re-run every example/experiment afterwards — reported uncertainties
     will change, and the honest ones are the new ones.
- **Verify.** `twin_demo.yml` with `strategy.model: resonator` reaches the same target; the legacy
  bit-for-bit oracle tests still pass; a new test asserts the reported CRB matches an empirical
  Monte-Carlo spread over ≥100 seeds within ~10% (the honest check that the bound is not optimistic).
- **DoD.** One `fisher_matrix`, one `optimal_excitation`, three model classes, all loop-drivable.

**Blocker:** 🟢 local.

## Track I — MIMO as a first-class orchestrated loop 🟢 local · *sysID core; supersedes Track B*

> **Goal.** A coupled multi-DoF plant is identified end-to-end from a config — designed drive,
> safety-gated injection, per-actuator campaign, joint rank-1 modal fit with shared poles, MIMO CRB,
> convergence on the worst per-mode fractional uncertainty, safe handoff — with the same operator
> experience as a SISO run.
>
> **Why.** This is what a SEI/ISC commissioner actually needs (the coupling is what they tune), the
> estimator half is done and proven, and every 6-DoF result in this repo currently requires a
> hand-written 300–850 line script (`experiments/rtsfreerun/run_srm6dof_modal.py` is 845 lines).
>
> **Status / prereqs.** Tracks D (campaign API), G (schema/registry), H (model protocol). The
> pieces to compose already exist: `assemble_campaign` → `find_modes` → `init_residues` →
> `MIMOModalEstimator.fit` → `mimo_parameter_covariance` → `modal_frac_uncertainty` →
> `iterate_mimo`.

- **Steps.**
  1. **Close the backend contract** (finding 2): implement `snapshot_state`/`restore_state` on
     `MIMOTwinBackend` and `ReducedPlantBackend`. Add a contract test parameterised over **every**
     `ChannelBackend` subclass so a new backend cannot silently skip the safe handoff, and fix the
     stale "lands in build step 5" message in `backends/base.py`.
  2. **`MIMOSysIDLoop`** (sibling of `SysIDLoop`, same `backend/estimator/designer/watchdog`
     construction): per pass, design from the current modes, drive each actuator in turn through the
     watchdog, assemble the campaign, fit jointly, compute the MIMO CRB, emit a snapshot, stop on
     `modal_frac_uncertainty <= target` or the iteration budget, and always finish through
     `watchdog.abort`.
  3. **Cross-pass accumulation** — `iterate_mimo` currently re-fits only the latest pass (its own
     docstring flags this). Give it the SISO loop's inverse-variance combination over passes: the
     natural form here is accumulating the per-line `(Ybar, Ubar, Cz)` sufficient statistics, not the
     ratio FRF. Verify that N accumulated passes beat 1 pass of N× the length only where the design
     changed (otherwise it is just averaging).
  4. **MIMO excitation design** — today the MIMO campaigns hand `assemble_campaign` a drive PSD
     designed by the SISO machinery. Extend the dispersion iteration to the MIMO Fisher
     (`mimo_fisher_matrix`) so the drive is optimal for the *joint* parameter set, and choose the
     excited-line set with `select_excited_lines` — which exists and is unit-tested
     (`tests/test_prior_robust.py:86`) but is called by **no** measurement path, SISO or MIMO.
  5. **Block structure in config** — `fit_block_decoupled`'s `blocks` (the {L,P,V}/{T,R,Y} plane
     decomposition that resolves spatial doublets) must be declarable in YAML, since it is physics
     the user knows and the fitter cannot guess.
  6. Retire the duplicated harness code in `experiments/rtsfreerun/*6dof*.py` and `sos_campaign.py`
     down to config + a thin call.
- **Verify.** A `configs/mimo_twin_demo.yml` recovers a coupled 3×3 twin's shared poles within the
  CRB from the CLI; re-run the SOS Stage-1 gate (`tests/test_sos_sysid.py`) through the new loop and
  reproduce its within-CRB result; the backend-contract test covers all backends.
- **DoD.** `system_ident run configs/mimo_twin_demo.yml` performs a joint MIMO identification with
  per-mode CRB bars and a safe handoff; no headline MIMO result needs a bespoke script.

**Blocker:** 🟢 local. The largest track here; sequence it after D/G/H.

## Track J — Live dashboard v2 🟢 local

> **Goal.** An operator view that works in a control room, shows uncertainty, and can display a MIMO
> run.
>
> **Why.** Audit finding 10: CDN-only Plotly (blank page when air-gapped), SISO-only schema, no CRB,
> nothing persisted.
>
> **Status / prereqs.** Track E (artifacts) for persistence; Track I for the MIMO schema. The hub /
> websocket / STOP plumbing is sound and tested — this is the view layer.

- **Steps.** Vendor Plotly into `dashboard/static/` (no network at render time); add the CRB band
  around the fitted FRF and a per-parameter σ panel; add a drive-headroom gauge (peak vs
  `safety.actuator_sat`) — the number an operator watches; per-DoF tabs and a MIMO
  magnitude-matrix view driven by an extended snapshot schema; show run metadata (config name, git
  SHA, seed, elapsed, pass number); write each snapshot into the Track-E run directory so the live
  view and the saved report are the same data.
- **Verify.** `tests/test_step9_dashboard.py` extended: page renders with **no external URLs** (assert
  no `http` in the HTML); snapshot validation accepts a MIMO payload; a run with `--out` leaves a
  replayable snapshot log.
- **DoD.** The dashboard renders fully offline, shows uncertainty, and handles a MIMO run.

**Blocker:** 🟢 local.

## Track K — Pre-injection drive safety (DAC-frame) 🟢 local

> **Goal.** Refuse or scale a drive **before** it is injected, based on its worst-case peak in the
> DAC frame after the anti-imaging / whitening / actuation filter chain.
>
> **Why.** Audit finding 11: the watchdog only reacts after a segment has already gone out. The
> repo's own crest-factor memory (`crest-factor-lives-at-the-dac`) says the plant-referred crest is
> not the DAC crest — so a drive that looks safe in plant units can clip at the DAC. Named as a
> Stage-4 bridge piece in the 40m handoff. On the twin this costs nothing; on hardware it is the
> difference between a rehearsal and an incident.
>
> **Status / prereqs.** `safety.py` (post-hoc checks + ramp + handoff) and the backends' `saturate`
> knob exist. Needs the per-channel filter chain to be declarable (Track G).

- **Steps.** Add `Watchdog.check_drive(drive, fs, channel)` — propagate the designed multisine
  through the declared DAC-frame filter chain, take the true worst-case peak (not the RMS×crest
  estimate), compare against the channel's DAC/coil limit with a configurable margin, and either
  refuse with a clear message or rescale the budget and say by how much. Call it in
  `SysIDLoop._measure_dof` *before* `backend.inject`, and expose it as the headroom number in
  `system_ident design` (Track F) and the dashboard gauge (Track J).
- **Verify.** A test where a drive passes the plant-referred check but clips after a whitening
  filter is caught pre-injection; the existing `tests/test_step5_safety.py` behaviour is unchanged
  when no filter chain is declared.
- **DoD.** No drive reaches a backend without a pre-injection headroom verdict.

**Blocker:** 🟢 local.

## Track L — 40m SOS deployment ladder 🔵 twin-box → 🔴 gated

> **Goal.** Identify a real 40m SOS as a 6-DoF MIMO system, via the endorsed safe path.
>
> **Why / status / steps / verify / DoD.** Owned by `notes/40m-sos-campaign-handoff-2026-07.md` —
> read that, not a summary. Current position: **Stage 0 (analytic 6-DoF SOS plant + OSEM projection)
> and Stage 1 (pyctl MIMO recovery within CRB) are DONE**; **Stage 2 (rtsfree composite
> `gen_x1sos6dof.py` + the nominal `OPT_CTRL_SUS*` foton damping banks + the analytic↔rtsfree
> validation gate) is next** and needs the Linux box. Stage 3 = scored fault/drift test-situation
> library. Stage 4 = fill `CDSBackend` behind a pluggable transport (real awg+nds2 **or** a twin
> transport) plus **two** bridge pieces: a timing/decoherence monitor, and the pre-injection DAC
> check (**that is Track K — build it there, not twice**). The delivered-fit manifest is **Track E**.
> Stage 5 is 🔴 **gated**: enumerate operator questions only, write no live-injection code.
>
> **Foton export is OUT OF SCOPE — resolved 2026-08-11 (user).** "The FOTON stuff is left for
> someone else to do." The 40m handoff note listed a third Stage-4 bridge piece, "Foton ZPK/SOS
> export"; that item is **struck**. Do not write a Foton exporter, do not re-propose one, and do not
> treat a delivered fit as needing to land in a foton bank — this repo's deliverable is the
> identified model + its manifest (Track E). *Reading* foton banks stays fine and is unaffected:
> the twin's `apply_foton_bank` / `readFilter` path is how the real L1-MC2 and `OPT_CTRL_SUS*`
> damping filters get loaded, and Track A/L depend on it. The prohibition is on **export**.
> See `.claude/memory/no-foton-export.md`.

**Blocker:** 🔵 twin-box for Stage 2–4; 🔴 gated for Stage 5.

---

## Ordering & parallelism

0. **Before anything else — two small, self-contained defects:** (a) the shared-model test
   isolation bug (finding 13) — the Track-A validation gate currently does not execute in a full
   suite run; (b) Track H step 5, the period bookkeeping / `P_eff` bug (finding 9) — every
   uncertainty number this repo prints depends on it.
1. **Then, in order — it unblocks everything else:** **D** (campaign into the package) →
   **G** (schema + backend registry) → **H** (one model protocol / one Fisher).
2. **Then, the two big product tracks in parallel:** **E** (artifacts + replay) with **F** (CLI
   verbs), and **I** (MIMO loop). F depends on E for `report`/`replay`; I depends on D/G/H.
3. **Then:** **J** (dashboard v2), **K** (pre-injection DAC safety — build it before Stage 4 needs
   it, not during).
4. **Anytime, independent:** **C** (refinement bake-off write-up).
5. **On the Linux box, on its own schedule:** **L** (40m SOS Stage 2 →).

Keep this file current: tick DoDs, correct the audit findings as they are fixed, and record any
channel-name or wiring correction discovered against a real model.
