# `system_ident` roadmap review + the questions a LIGO controls engineer will ask

**Date:** 2026-06-21  **Reviewer:** skeptical practitioner pass (read-only)  **Branch:** `main` @ `387c2c4`
**Scope:** the P&S pipeline (`src/system_ident/`), the twins (`backends/twin.py`,
`backends/rtsfreerun_adapter.py`, `darm.py`), the docs (`docs/tutorial/*`, `docs/examples/*`),
and the roadmap (`.llm/roadmap.md`, `notes/darm-calibration-via-pns.md`,
`notes/docs-audit-2026-06-20.md`).

This note is feedstock for deciding what to build, test, and document next. It is deliberately
hard on the gaps; the craft is genuinely good (see the docs audit), so the leverage now is in
the things that stand between a clean twin suite and a controls engineer trusting it on a real
suspension.

---

## 1. Roadmap assessment — solid / partial / claimed-vs-verified

### Solid and demonstrated

- **The P&S measurement core is real and coherent, end to end.** Optimal design by the
  dispersion fixed point (`design/pintelon.py:optimal_excitation`, port of
  `sysIDlib.get_opt_exc_Pxx`), periodic Schroeder multisine synthesis
  (`excitation.multisine_from_psd`, `excitation.py:117`), the leakage-free reference-based
  ratio-of-averages FRF with a per-bin error bar (`loop.py:_estimate_tf_periodic:383`), the
  Sanathanan–Koerner→ML estimator (`estimators/gml.py`), and Fisher/CRB
  (`fisher.py:fisher_matrix`, `dispersion`). The orchestration (`loop.SysIDLoop`) closes the
  loop with a prior-robust first pass (`design/pintelon.py:prior_robust_excitation`) and
  inverse-variance cross-pass accumulation (`loop.py:_accumulate:309`). Parts are validated
  bit-for-bit against the legacy `sysIDlib` oracle (`tests/test_step2_validation.py`,
  `test_step3_validation.py`). This is the package's strongest asset.

- **The twin is more than a toy.** `TwinBackend` models MIMO cross-coupling, closed-loop
  controllers via an exact 6-numerator closed-loop construction (`twin.py:_build_closed_loop:247`),
  integer-sample response delay, and a hard actuator clip. The closed-loop reference FRF
  cancelling the controller is demonstrated (`docs/tutorial/why-optimal-excitation.qmd` §6,
  `docs/examples/05`).

- **The compiled-CDS demonstration (RTSfreerun A1–A4) is the headline and it is real.**
  `.llm/roadmap.md` Track A results log: A2 SISO recovery on the compiled `x1hsts` plant (5 modes,
  ~0.2% median FRF error under the twin's seismic+readout noise), A3 closed-loop diagonal recovery
  through the **real L1-MC2 dampers** (all six DOFs <0.1%, controller cancelled), A4 open-loop MIMO
  tensor vs a state-space oracle. This is the most convincing thing in the repo: it runs the same C
  numerics a front-end runs, scored against an analytic oracle (`backends/rtsfreerun_oracle.py`),
  and it caught a real sign bug in the plant-input reconstruction (`COIL_DRV_SUM` is a `"+-"` sum;
  `feedback_coeff=-1`, `rtsfreerun_adapter.py:128`).

- **DARM-calibration twin + a fair head-to-head harness.** `darm.py` builds a representative
  closed-loop DARM (sensing pole+delay, three actuation stages, derived servo) and provides
  `multisine_response_sigma` vs `swept_sine_response_sigma` with honest wall-clock accounting,
  plus the `recover_response`/`fit_sensing`/`recover_actuation` deliverables. The accompanying
  note (`notes/darm-calibration-via-pns.md`) is unusually candid about where P&S does *not* help.

- **The methods page is honest about its own limits.** `why-optimal-excitation.qmd` §2.1, §5, §9
  repeatedly refuse to claim a crest/peak-drive advantage and flag the DAC-frame gap. That
  intellectual honesty is exactly what earns a skeptic's attention.

### Stubbed / partial / not yet built

- **No hardware backend.** `backends/cds.py:CDSBackend` raises `NotImplementedError` on every
  method ("lands in build step 8"). There is **zero** real-CDS validation: no `awg`/`cdsutils`
  injection, no `nds2` readback, no GPS-timestamp/period-boundary alignment. Everything below
  "real hardware" in this review is therefore untested by construction. The module's own docstring
  is admirably explicit about this (`cds.py:8-32`).

- **No MIMO joint identification (Track B).** `SysIDLoop.run` is strictly SISO-per-DoF: it loops
  `for d in dofs` and fits one `TFModel` each (`loop.py:155-176`, `loop.py:221`); the estimator
  interface is SISO (`estimators/base.py`). The per-pair FRF tensor is computed *by hand* in
  `tests/test_twin_mimo.py` and `experiments/rtsfreerun/hsts6dof_loop.py`, but there is no joint
  fit with shared modal poles, no MIMO Fisher/CRB, and config can't even enable coupling
  (`config.build_twin_backend` never passes `coupling=`). Coupling is *mutually exclusive* with
  controllers in the twin (`twin.py:113-117`), so MIMO is open-loop-only today.

- **DAC-frame, filter-aware excitation design is not implemented.** The whole crest/peak story is
  plant-referred; the whitening/de-whitening and actuation chain between the digital request and
  the DAC limit is not modeled (`why-optimal-excitation.qmd` §2.1, §9; `excitation.py:92-105`
  docstring). The binding actuator limit is at the DAC, and the design does not see it.

- **The per-bin noise model is the period-variance estimator, not the LPM** the docs name.
  `_estimate_tf_periodic` computes the FRF variance from period-to-period residual scatter
  (`loop.py:438-444`) — the Pintelon period-variance estimator, which needs many full periods
  (`P_eff`). The docs and the DARM note repeatedly call it the **Local Polynomial Method**
  (`notes/darm-calibration-via-pns.md` §3 item 3; the examples' HTML). The LPM is a *different*
  technique (local polynomial fit of FRF + noise transient across neighbouring bins, designed for
  the few-period / single-realisation case). This is a real mislabel that a P&S-literate reviewer
  will catch immediately, and it matters: the DARM harness already had to fight `P_eff` (the
  comments at `darm.py:233-239`, `253-262` exist precisely because 2 periods underflow to a
  fabricated σ floor of `1e-9`, `loop.py:460`).

- **Coherence is fabricated-floored, not measured, on a clean twin.** `coh` is derived from the
  same variance (`loop.py:462-464`) and floored to `1e-6` on unexcited/clean bins. On a noiseless
  twin it is not an independent diagnostic.

### Claimed vs verified — watch-items

- "**Same API — twin or hardware … single config line, no code changes**" (landing page) is false
  while `CDSBackend` is a stub; already flagged C4 in `notes/docs-audit-2026-06-20.md`. README is
  honest; the site is not.
- "**LPM / local polynomial**" noise model — claimed in docs, **not** what the code does (above).
- DARM novelty ("**no one has applied P&S to GW calibration**") and the O4 numbers are flagged
  *unverified / targets-not-results* by the authors' own note (`darm-calibration-via-pns.md` §6).
  Good — but it means the DARM page is a *design argument on a representative twin*, not evidence.
- "Validated" appears throughout; almost always it means *against the twin or the sysIDlib oracle*,
  never against hardware. The `cds.py` docstring says this plainly; the marketing should too.

---

## 2. The questions/tests a real LIGO controls engineer will raise

### A. Measurement validity on real CDS (timing, clocking, decimation)

**A1. Does the leakage-free guarantee survive real AWG↔NDS timing?**
*Why it matters:* the entire pitch rests on integer-period synchronous capture. On CDS the
excitation is generated by `awg`/`cdsutils` and read back through `nds2`/online frames; there is no
guarantee the first analysed sample sits on the injected period boundary, and the front-end and
your analysis clock are not the same oscillator. A constant integer-sample offset cancels in the
ratio-of-averages (the code relies on this), but a fractional-sample drift over a multi-minute
record re-introduces leakage — exactly the bias the method claims to kill. *Current state:* the
twin assumes perfect co-sampling; `cds.py:18-28` lists this as the #1 untested gap. There is **no
detector** in the pipeline for lost period alignment. *Test that answers it:* on the compiled
RTSfreerun model or in a control-room test, inject a known multisine, deliberately offset the
analysis window by a fractional sample (resample by 1+ε), and plot recovered-peak bias and the
period-variance error bar vs ε. Demonstrate a *monitor* (e.g. per-period phase walk of the drive
lines, or a watchdog on `arg(X_p)` drift) that flags decohered capture before the FRF is trusted.

**A2. GPS-timestamp → reshape-grid alignment.** The reshape into periods (`loop.py:411`) assumes
`len(x)` is an integer multiple of `nperseg` and that sample 0 is a period boundary. On real frames
you get GPS-timestamped data starting anywhere. *Test:* a `CDSBackend.read` that aligns the first
returned sample to a known GPS period boundary, plus a test that a 1-sample misalignment is
detected, not silently absorbed.

**A3. Decimation / anti-alias phase.** The front-end runs at 16384 Hz; `measurement.fs` is lower
and the adapter decimates (`.llm/roadmap.md` A1: "clean ×64 decimation"). On hardware the AA/decim
filters add phase the rational fit will absorb as plant phase or delay. *Test:* show the recovered
delay/phase is the decimation filter's, characterise it, and divide it out (or fit it) — don't let
it masquerade as `f_cc` or a suspension pole.

### B. The DAC frame (the limit that actually binds)

**B1. What does a DAC-referred, filter-aware excitation design look like?**
*Why it matters:* operators do not care about plant-referred RMS; they care that the DAC does not
rail and the ESD/coil driver stays inside its analog range. The actuator limit lives *after* the
whitening/de-whitening and actuation filters. The current design minimises crest in plant units,
which the whitening chain scrambles (`why-optimal-excitation.qmd` §2.1, §9 are explicit). So the
package optimises a quantity that does not bind and cannot guarantee the one that does.
*Current state:* not modeled; flagged as future work. *Test/demo that answers it:* add an
actuation-chain `Filter` between the designed PSD and the injected counts (a Foton-exported
whitening + coil-driver model), push the Schroeder/crest optimisation *through* it, and minimise
the peak **at the DAC node**. Demonstrate on the HSTS coil driver: show plant-referred-optimal vs
DAC-optimal drives and the peak-DAC-counts each produces against `COIL_DRIVER_LIMIT` (30000, the
number the roadmap already tunes against, A2). Until this exists, the honest claim is "we optimise
information per second and per in-band-RMS," not "per actuator headroom."

**B2. Counts↔physical units across the chain.** See §H.

### C. Comparison to the tools the control room already uses

**C1. Why this instead of DTT/diaggui swept sine + awggui?** *Why it matters:* every LIGO controls
engineer already measures FRFs with `diaggui` (swept sine / broadband) driving through `awggui`,
fits with `vectfit`/Foton, and trusts the workflow. A new pipeline must beat the incumbent on a
real measurement, not on a twin. *Current state:* the comparison exists only against *idealised*
swept-sine/random *simulations* the package itself generates (`why-optimal-excitation.qmd` §3–§5,
`darm.py` swept harness). That is circular to a skeptic — the baseline is a strawman the author
controls. *Test that convinces:* take one real `diaggui` HSTS or DARM measurement (an existing
`.xml` from a maintenance period), re-fit the *same data* with `GMLEstimator`, and show
same-or-better parameter σ; then run a P&S multisine of *equal wall-clock and equal in-band RMS*
through `awggui` on the same suspension and compare recovered f0/Q and the measured (not modeled)
σ. Head-to-head, same instrument state, same drive budget — that is the only thing that moves a
controls engineer.

**C2. Foton interop.** The fit comes out as `num/den` (`TFModel`). Foton/CDS speak ZPK/SOS. A prior
commit removed Foton export (git log: "refactor: remove Foton export — out of scope"). For
adoption that is backwards: an engineer wants the fit *as a Foton filter string* to drop into a
damping bank. *Test:* round-trip a fitted `TFModel` → Foton ZPK → back, and show it matches; ship
it as an export.

### D. Safety / operations

**D1. Does the watchdog cover what an operator actually needs?** *Current state:* `safety.Watchdog`
checks two things — per-channel drive peak vs `actuator_sat`, and per-DoF output RMS vs a ceiling —
and runs a ramp-down + snapshot/restore on breach (`safety.py:97-142`). That is a reasonable
*minimum*, but it is **post-hoc and per-segment**: it inspects a segment *after* it was injected
and read (`loop.py:215`), so a saturating drive has already gone to the coil before the breach is
seen. There is no pre-injection check that the *designed* drive's worst-case peak (through the
actuation chain — see B1) is within limit. *Test:* a pre-flight assertion that
`max|drive_at_DAC| < actuator_sat` before `inject`, plus a unit test that a design exceeding the
limit is rejected, not ramped-down after the fact.

**D2. ESD limits and charge.** The TST stage is an ESD; it has bias-voltage and charge constraints
the watchdog knows nothing about (only a generic `actuator_sat`). *Test:* an ESD-specific limit
(per-stage voltage ceiling) and a demonstration that the DARM TST-stage drive respects it.

**D3. Lock-loss risk and Guardian integration.** *Why it matters:* injecting a multisine comb into
a locked interferometer risks breaking lock; an operator needs the run to be Guardian-aware
(request a state, abort on lock-loss, not fight the IMC/ASC). *Current state:* none. The "operator
STOP" is a dashboard websocket button (`loop.py:198`), not a Guardian node. *Test/demo:* a Guardian
node (or a documented integration) that gates injection on lock state and aborts on a lock-loss
flag, with the existing ramp-down as the safe exit.

**D4. Is ramp-down enough, fast enough?** `ramp_down_secs` default 2 s, half-cosine taper
(`twin.py:207-220`). On a real high-Q suspension is 2 s a safe de-excitation, and does
snapshot/restore actually restore *filter module state* (history), not just the drive array? On the
twin, `snapshot_state` only saves `_drives` (`twin.py:223`). On hardware "restore pre-run filter
state" is a much bigger promise. *Test:* define and test what state CDS restore must capture
(SDF/filter-module settings, `fm_clear_history` — the roadmap already notes history carryover
biases successive measurements, A-results "one model per process").

### E. Closed loop on a *real* loop

**E1. Recovering the open-loop plant through a real digital servo.** The twin's controller is a
rational `C(s)` (`twin.py:_build_closed_loop`); the real damping bank is a Foton SOS cascade with
delays, decimation, and saturations. The reference-based FRF cancels the controller *if* the drive
monitor `X` is truly the after-controller plant input. On the composite this required getting the
`"+-"` feedback sum sign right (`feedback_coeff=-1`) — a one-sign error that "dominates at the
damped modes" (roadmap A3 root-cause). *Why it matters:* on hardware you must point `channels.drive`
at the genuine plant-input test point, and a wrong/missing one silently biases toward the
closed-loop response `T` (the loop already warns about this, `loop.py:_warn_open_drive_monitor`).
*Test:* demonstrate, on the compiled `x1hsts6dof` with the L1-MC2 loops, that recovery degrades
gracefully (and is *flagged*) when the drive monitor is wrong by a sign or a one-cycle delay —
i.e. that the failure mode the roadmap hit by accident is now *detected*, not just fixed.

**E2. Stability margin during injection.** Adding a comb inside a loop near UGF can eat phase
margin. *Current state:* unaddressed. *Test:* show the in-band injected power does not push the OLG
past a margin threshold (a pre-flight check using the controller + plant prior).

### F. Statistical rigor

**F1. Is the per-bin noise model validated against real data?** *Current state:* it is the
period-variance estimator on twin white/seismic noise; never confronted with real CDS noise (lines,
glitches, non-Gaussian tails, 1/f). And it is **mislabeled LPM** (see §1). *Tests:* (a) on real
quiet-time data, check residual whiteness (Anderson–Darling / Ljung–Box on the post-fit residuals),
Gaussianity, and that the *predicted* per-bin σ matches the *realised* scatter over independent
records (a σ-calibration plot — predicted vs empirical). (b) Decide whether to actually implement
the LPM for the few-period regime, or rename everything to "period-variance estimator." Right now
the package claims a method it does not run.

**F2. Coherence thresholds.** Coherence is plotted everywhere but never defined or thresholded
(docs audit I8); on a clean twin it is floored, not measured. *Test:* define γ²(f) as computed
here, set a trust threshold, and show a low-coherence bin correctly down-weighting the fit on real
noisy data.

**F3. Model-order selection.** The one piece of hard-won operational wisdom — "L carries five modes;
pitch and yaw carry fewer, and over-modelling them is what used to break the optimal-excitation
design" (roadmap A-findings; `examples/07`) — is buried, and the package has **no order-selection
procedure**. `safe_inverse`'s pinv fallback (`fisher.py:128`) papers over the rank deficiency that
over-modelling causes, rather than diagnosing it. *Test/tool:* an order-selection diagnostic (AIC/
MDL or a whiteness-gated order sweep) and a demonstration that it picks 5 modes for L and fewer for
P/Y *without* the operator hand-tuning the prior — turning the lesson into code.

**F4. Error budget / systematics separation.** Calibration wants systematic (bias) separated from
statistical (CRB) error. The pipeline reports CRB only; leakage/decimation/aliasing biases are not
budgeted. *Test:* a worked error budget on one recovery (CRB + leakage residual + decimation phase
+ prior-misallocation), in the form the cal group ships a 68% envelope.

### G. Generalization / MIMO

**G1. Real suspensions are coupled — what is the joint-identification story?** *Current state:*
Track B is **unbuilt** (§1). The package fits diagonal SISO transfer functions and recovers the
off-diagonal tensor only by hand in tests. A quad's L↔P and R↔Y coupling, and the shared normal-
mode poles across elements, are not jointly estimated; the SISO fits don't even share poles.
*Why it matters:* an engineer identifying a HSTS/quad wants the *matrix* with common modes, not 36
independent fits that disagree on the same resonance. *Test that demonstrates it:* the Track B
deliverable — a config-declarable coupled twin, a joint estimator with shared modal denominator and
per-element residue numerators, MIMO Fisher/CRB — recovering both diagonal anti-resonance notches
and notch-free off-diagonals, with the shared poles coming back common. The twin already emits
correct coupled data (`plant.coupled_suspension`, `tests/test_twin_mimo.py`), so the oracle exists;
this is buildable now with no hardware.

**G2. MIMO + closed loop simultaneously.** The twin *forbids* coupling with controllers
(`twin.py:113-117`). Real suspensions are coupled *and* damped at once. That combination — the
actual operating condition — is currently unrepresentable even in simulation. *Test:* lift the
guard (or model it in the RTSfreerun composite, where A3+A4 already coexist) and show joint MIMO
recovery through closed loops.

### H. Units & calibration / traceability

**H1. Counts vs physical units.** Everything internal is counts (twin) or model units (`g_c` in
ct/m, stages in m/ct, `darm.py`). Transferring a fit to the real actuation chain needs the
calibration from counts to Newtons/metres at each stage, with traceability. *Current state:* no
units layer; `TFModel` is unitless num/den. *Test:* a worked counts→physical conversion on one HSTS
stage with the chain gains, and a statement of how the absolute scale enters (for DARM it's Pcal,
unchanged — the note is clear; for suspensions it's the coil/ESD calibration, undocumented).

**H2. Provenance.** A fit must carry: the config, the prior, the seed, the data GPS span, the model
order, the recovered σ, and the oracle/comparison if any. *Current state:* `LoopResult` carries
models + history, no provenance metadata. *Test:* a result manifest (config hash, GPS span, env)
written alongside every run.

### I. Reproducibility & trust

**I1. Can a third party reproduce a result on their own interferometer state?** *Current state:*
the twin demos are reproducible (`--twin --yes`); the *hardware* result does not exist yet. The
onboarding last mile is also missing (docs audit I7: no quickstart, no full run-config reference,
no "bring your own subsystem" guide), so even reproducing the *twin* result on a new plant is
under-served. *Test/doc:* the Configuration page the docs audit asks for, plus a single command that
reproduces an `experiments/rtsfreerun` recovery from a checked-in config + oracle, end to end, in CI
where the model is available.

---

## 3. Prioritized recommendations (highest leverage first)

1. **Implement and exercise `CDSBackend` against the compiled RTSfreerun model with a deliberate
   timing-perturbation test (A1–A3, F1).** This is the single biggest credibility gap: every
   "validated" claim today stops at the twin. Build `awg`/`nds2` (or a faithful frame-replay) read
   path, align to GPS period boundaries, and ship a test that injects a fractional-sample drift and
   shows (a) the leakage bias vs ε and (b) a *monitor that flags it*. **Artifact:** a green
   hardware-path test + a "decoherence monitor" panel, turning "leakage-free" from an assumption
   into a guarded guarantee.

2. **A real head-to-head against `diaggui`/`awggui` on one shared measurement (C1).** Re-fit an
   existing real `diaggui` HSTS or DARM `.xml` with `GMLEstimator`, then run an equal-wall-clock,
   equal-in-band-RMS P&S multisine on the same suspension and compare recovered f0/Q/σ. **Artifact:**
   a one-page result with the incumbent tool as the baseline (not a self-generated strawman) — the
   only evidence that converts a skeptic.

3. **DAC-frame, filter-aware excitation + a pre-injection saturation check (B1, D1).** Put a
   Foton-exported whitening/coil-driver model between the designed PSD and the injected counts,
   minimise the peak *at the DAC*, and assert `max|drive_at_DAC| < limit` before injecting.
   **Artifact:** an HSTS demo showing plant-optimal vs DAC-optimal drives and peak-counts vs
   `COIL_DRIVER_LIMIT`, plus a watchdog that rejects an over-budget design pre-flight.

4. **Build Track B: joint MIMO identification with shared modal poles + MIMO CRB (G1, G2).** The
   twin oracle already exists; this is pure software and removes the most obvious "but real
   suspensions are coupled" objection. **Artifact:** a config-declarable coupled twin recovered as a
   matrix with common poles, off-diagonal notches matched, reported per-element σ.

5. **Fix the statistical story: rename/implement the noise model correctly and add residual
   diagnostics + order selection (F1, F3).** Either implement the actual LPM or stop calling the
   period-variance estimator "LPM" everywhere; add a predicted-vs-empirical σ calibration plot,
   residual whiteness/Gaussianity tests, and an order-selection diagnostic that reproduces the
   "5 modes on L, fewer on P/Y" lesson *automatically*. **Artifact:** a "trust" panel (σ-calibration
   + whiteness + chosen order) on a real-noise recovery, and a code/docs that no longer claim a
   method they don't run.

**Honorable mentions** (cheap, high trust-per-effort): Foton ZPK export of the fit (C2); Guardian/
lock-aware injection gating (D3); a result provenance manifest (H2); and the docs last-mile
(quickstart + full run-config reference, docs audit I7). The docs-audit trust-breakers (dead stat
pill C1, stale API reference C2, "CDS works" overstatement C4) should be fixed regardless — a
measurement tool cannot ship a home page that contradicts its own source.

---

### One-paragraph bottom line

The P&S core is solid, the twins are unusually faithful, and the RTSfreerun A1–A4 demonstration on
compiled CDS numerics is genuinely impressive — it even caught a real sign bug. But every adoption-
blocking question a controls engineer asks lands in the unbuilt half: there is no hardware backend,
so nothing is validated against real CDS timing/noise; the excitation is optimised in a frame
(plant-referred) that is not the one that binds (the DAC); the comparison to DTT/awggui is against a
self-generated strawman, not a real measurement; MIMO joint ID is absent; and the per-bin noise
model is mislabeled and never confronted with real data. The work to earn cross-subsystem trust is
not more twin polish — it is closing the loop on real hardware timing, a DAC-frame design, and one
honest head-to-head against the tool the control room already trusts.
