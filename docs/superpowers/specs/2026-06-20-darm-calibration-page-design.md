# DARM calibration via P&S — executable twin + docs page (Example 08)

**Status:** approved design, pre-implementation
**Date:** 2026-06-20
**Companion note:** `notes/darm-calibration-via-pns.md` (architecture + swept-sine comparison, prose)

## 1. Goal

Build an **executable** DARM-calibration demonstration that runs the existing
Pintelon–Schoukens (P&S) pipeline — periodic multisine → leakage-free reference FRF →
maximum-likelihood fit → Cramér–Rao refinement — against a faithful closed-loop DARM twin, and
present it as docs Example 08, in the same "show me, don't trust me" style as Example 07
(rtsfreerun). The page must let a LIGO calibration reader see *what went in and what came out*:
raw time series, leakage-free FRFs with per-line coherence, recovered parameters with CRB
uncertainty, and a fair head-to-head against the present-day swept sine.

This is **one P&S pipeline** (the project's standing constraint) applied to a new plant — no new
estimation method.

## 2. What is modeled (the DARM twin)

A representative aLIGO-like single-DARM-loop twin. All numbers are **representative, not a
specific interferometer state**, and are labeled as such in the page.

Loop algebra (test-mass displacement referred):

```
d_err = C · (x_free + x_pc + Σ_i κ_i N_i c_i) / (1 + G),     G = C · A · D
A(f)  = κ_U N_UIM(f) + κ_PU N_PUM(f) + κ_T N_TST(f)
R(f)  = (1 + G) / C            # the calibration deliverable (counts → displacement)
```

- **Sensing** `C(f) = g_C / (1 + i f/f_cc) · e^{−i 2π f τ}`, `f_cc ≈ 360 Hz`, `τ ≈ 77 µs`,
  optical gain `g_C` (ct/m).
- **Actuation** `A(f)` — three quad stages `N_UIM, N_PUM, N_TST` built from the existing
  suspension TF primitives, scaled so UIM dominates low-f and TST high-f with realistic
  crossovers. `κ_U, κ_PU, κ_T` are the TDCF-like strengths the cal team fits.
- **Servo** `D(f)` — a representative DARM digital control filter, UGF ~50 Hz.
- **Two noise sources, both real and distinct** (the correction that drove this design):
  - **Process disturbance** `x_free` — length/seismic noise entering the loop, so it appears at
    `d_err` shaped by the closed loop (`C/(1+G)`-colored), *not* white.
  - **Sensing noise** `n` — additive readout noise on `d_err` (flat-ish).
  These mirror `TwinBackend.disturbance_asd` vs `sensor_asd`. The P&S period-to-period variance
  gives the **per-bin noise covariance from this real mixture** — it is not handed an assumed
  white floor. That is the methodological point of the page.

## 3. Measurement campaigns and recovery

Excitations are P&S periodic multisines, injected through the backend's 3 s Tukey on/off ramp
(actuator-safe; ramp lives outside the leakage-free measured periods, per the package default).

1. **Pcal campaign** — inject multisine `x_pc` at Pcal (the absolute displacement reference).
   - Leakage-free FRF `d_err / x_pc = C / (1 + G)`.
   - Multiply by the model `(1 + G)` to expose `C`; ML-fit `g_C, f_cc, τ`.
   - Form the deliverable `R(f) = (1 + G)/C` with its CRB envelope.

2. **Per-stage actuation campaign** — inject multisine counts `c_i` at each stage UIM/PUM/TST.
   - FRF `d_err / c_i = C κ_i N_i / (1 + G)`.
   - Divide by the Pcal FRF → `κ_i N_i`, **referenced to the Pcal meter** (Pcal is the absolute
     ruler). With the known stage shape `N_i`, recover each `κ_i`.
   - A Pcal-only measurement cannot separate `κ_U/κ_PU/κ_T` (degenerate inside `G`); the separate
     stage drives are what break the degeneracy. This is the real DARM-cal procedure.

All per-bin uncertainties come from the period-to-period variance on the real disturbance+sensing
mixture; parameter CRBs come from the existing Fisher machinery.

## 4. Head-to-head vs the swept sine

Same twin, same disturbance + sensing noise, same total wall-clock `T`:

- **Multisine (P&S):** all bins measured at once → CRB `σ(R(f))` over the comb.
- **Swept sine (baseline):** one frequency per dwell; `σ` at each point from the single-bin SNR
  over its dwell; `M` points so `M · dwell = T`.

Plot both `σ(R(f))`-per-hour envelopes for equal wall-clock, plus the figure-of-merit table from
the companion note (§4). The efficiency claim is **shown on the same twin, not asserted** (note §6).
No crest-factor / Schroeder argument — the win is simultaneity + leakage-free estimation + CRB
allocation, which is what the multisine buys for DARM.

## 5. Code layout

- `src/system_ident/darm.py` — `DARMLoop`: builds `C`, the 3-stage `A`, the servo `D`; exposes
  `G(f)`, `R(f)`, and the closed-loop FRFs per injection point; and a time-series generator that
  drives a multisine through the loop under process disturbance + sensing noise. Tested package
  code (not docs glue).
- `src/system_ident/backends/darm_adapter.py` — `DARMBackend` implementing the `ChannelBackend`
  API with the shared 3 s Tukey ramp (`ramp_s = 3.0` default), channels
  `{PCAL_EXC, UIM_EXC, PUM_EXC, TST_EXC, DARM_ERR}`. The existing P&S loop runs unchanged against
  it; `from_config` reads `measurement.t_ramp` like the other backends.
- `docs/darm_demo.py` — presentation glue + house-style figure wrappers (`docs/sysid_plots.py`),
  mirroring `docs/rtsfreerun_demo.py`. **Every figure is SVG and tracked in Git LFS** (hard rule).
  Y-limits are computed from the actual data so traces are never clipped (hard rule).
- `tests/test_darm.py` — see §7.
- `docs/examples/08-darm-calibration.qmd` — the page, `freeze: true` (heavy multi-campaign compute,
  consistent with Example 07), with a committed `docs/_freeze/` entry.
- `docs/_quarto.yml` already globs `examples/*.qmd`; add an `08` thumbnail
  (`docs/examples/thumbnails/08.svg`, LFS).

## 6. Page structure (`08-darm-calibration.qmd`)

1. The DARM loop and what calibration must deliver (`h = R·d_err/L`, `R=(1+G)/C`, `G=A·D·C`).
2. The twin's truth bodes: `C`, the three actuation stages `A`, servo `D`, open-loop `G`, `R`.
3. One raw Pcal multisine measurement — ramped drive in, `d_err` out under real disturbance +
   sensing noise. "What you'd watch in the control room."
4. Leakage-free `d_err/x_pc = C/(1+G)` with per-line coherence; recovered sensing `C`
   (`g_C, f_cc, τ`) vs truth; normalized residuals (~N(0,1) if the noise model is right).
5. Per-stage actuation: recover `κ_U/κ_PU/κ_T` (table) + the stage-crossover bode, Pcal as ruler.
6. The deliverable `R(f)` with its CRB envelope.
7. Swept-sine head-to-head: `σ(R(f))`-per-hour envelopes (P&S vs sweep) + FoM table.
8. Honest gaps and what must be validated before claiming anything (note §6).

## 7. Testing

`tests/test_darm.py`:

- **Loop self-consistency:** `R(f)` equals `(1+G)/C` from the twin's own `C, A, D`; the closed-loop
  FRF identities (`d_err/x_pc = C/(1+G)`, `d_err/c_i = Cκ_iN_i/(1+G)`) hold to numerical tol.
- **Disturbance vs sensing coloring:** disturbance-only readout PSD is loop-shaped (high
  peak/median); sensing-only is approximately flat — mirroring `test_disturbance_asd_colors_quiet_readout`.
- **Recovery under noise:** Pcal campaign recovers `g_C, f_cc, τ` and `R(f)`; per-stage campaign
  recovers `κ_U/κ_PU/κ_T`, each within a stated tolerance, with CRB shrinking pass over pass.
- **Comparison harness:** both swept-sine and multisine `σ(R(f))` are computed on the same twin for
  equal wall-clock (no tolerance asserted on which "wins" — the point is that both are produced
  honestly).

Tolerances are set from real recovery runs, never loosened to pass. No skipif-hidden regressions.

## 8. Out of scope (YAGNI)

- No optical spring / SRC detuning in `C` (single cavity pole + delay is enough to make the point).
- No multi-line TDCF tracking lines, no time-dependent κ(t) — the κ's are static per measurement.
- No MIMO / multi-loop (single DARM loop only).
- No Foton/real-CDS DARM filter import — a representative servo is sufficient and labeled as such.
- No claim that P&S beats the sweep in wall-clock; the page *shows* both envelopes and lets the
  reader judge (note §6 honesty).

## 9. Hard rules honored

- One P&S pipeline; no method divergence ([[stay-on-pintelon-schoukens]]).
- Every plot SVG, all graphics in Git LFS ([[graphics-svg-lfs-only]]).
- Y-limits data-driven and verified against real data before render (no clipped traces).
- Trunk-based: commit + push straight to `main` ([[trunk-based-push-to-main]]).
- No silent reversal of any prior user-directed change ([[never-silently-reverse-user-commands]]).
