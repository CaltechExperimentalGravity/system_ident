# DARM calibration — two fidelity upgrades: implementation plan (July 2026)

> **UPDATE 2026-07-29 (review + Upgrade 1 implemented).** After a code-verified review:
> - **Upgrade 1 is DONE** — `DARMLoop.default_reduced()` + `ReducedStageShape` in `darm.py`,
>   tests in `tests/test_darm_reduced.py`, green; full suite green. Stages are the real
>   reduced-quad columns; `G==A·D·C` holds; quad modes (0.43/0.52/0.99/1.98 Hz) are embedded.
>   **Finding:** the modes are embedded at any `fmin`; lowering the *campaign* band to 0.3 Hz
>   biases κ recovery (Q≈50 modes, ~0.01 Hz linewidth, unresolvable at the 1 Hz bin spacing), so
>   κ is measured in the smooth 10–1500 Hz region. Per-stage counts→force gains anchored at
>   100 Hz; absolute scale is irrelevant (loop invariant).
> - **Upgrade 2 is DONE — but NOT with the broken form below.** The agent's optical-spring factor
>   `ωs²(δ)/(s²+bs+ωs²(δ))` with `ωs²=K·sin2δ` was wrong (`C(δ=0)≡0`; in-band shape corrupted at
>   every δ). Replaced with the **Cahillane form** (Rana's choice; PRD 96, 102001):
>   `C = [single cavity pole] · f²/(f² + f_s² − i·f·f_s/Q_s)`, with **signed** `f_s²=spring_K·sin2δ`.
>   At δ=0 the spring factor is identically 1 ⇒ C == the single-pole BRSE model **exactly**
>   (verified atol=0). Detuning lifts a complex spring resonance in band (restoring for δ>0,
>   anti-restoring for δ<0), spanning the split at δ=0 across the ±7° drift. `darm.py`:
>   `optical_spring_factor`, `sensing_model_detuned`, `DARMLoop.{delta,spring_K,spring_Q,fs2,
>   spring_pole}`, `C()` routed through the detuned model. δ recovered from the Pcal FRF
>   (`darm_tv.snapshot_delta`/`track_delta`) within CRB, both signs; reuses `fit_tv`/`resolvability`.
>   Tests `tests/test_darm_detuned.py` (8). Note: δ is well-identified even near tuning (the √f_s²
>   damping term is steep there) — it is NOT unidentifiable at BRSE.

> **UPDATE 2026-07-29 (hierarchical actuation + calibration lines).** DARM now drives the quad
> hierarchically across **M0 / PUM / TST** (`_REDUCED_MAP`; `default_reduced()` stage set changed
> from UIM→M0 per Rana). Added an optional per-stage **`distribution` filter** slot so the
> actuation is `A_i = κ_i·D_i(f)·N_i(f)` — the inter-stage crossovers live entirely in these
> filters (the mechanical columns don't cross; TST always dominates), to be populated from the
> twin actuation design. Added `darm_tv.cal_line_response(loop, freqs)`: injects calibration
> lines on each stage and returns the ruler-calibrated per-stage actuation `A_i = H_stage/H_pcal`
> at the lines (for measuring where each stage crosses the one below). Tests
> `tests/test_darm_callines.py`; full suite green. (The placeholder `default()` loop keeps
> its UIM/PUM/TST toy stages.)

> **UPDATE 2026-07-29 (twin offload filters wired in).** `default_reduced(hierarchical=True)`
> populates `DARMLoop.distribution` with the nested-offload filters reproduced from
> `digital_twin/twin/experiments/cavity_arm_lsc_hierarchical/lib.py::offload_filters`
> (`src/system_ident/darm_actuation.py`, FRF-identical — verified 0.0 diff): `D_TST=1`,
> `D_PUM=O_A`, `D_M0=O_A·O_B`. `snapshot_kappa` made distribution-aware (rules by the full
> `D_i·N_i` shape). Tests `tests/test_darm_hierarchical.py`.

> **UPDATE 2026-07-30 (damp-first + force-unit offload — crossovers now land on target).** Rana:
> "in twin, the reduced quad model is DAMPED and then the hierarchical loop is made; the damping
> has ~no effect on the crossover frequencies." Two corrections to the 07-29 wiring:
> (1) **Damp the quad first, with the twin's REAL ETMX damping.** The twin damps the reduced quad
> with a 6-DOF M0 velocity-damping loop BEFORE the offload design, so the 0.4–3 Hz suspension-mode
> forest (Q~1e3) doesn't mask the smooth 1/f² compliance tails that set the crossovers. The damping
> is reproduced verbatim (FRF-identical, 0.0 diff — verified) from
> `digital_twin/aligo-suspension-models/docs/source/_doc_helpers.py` (`SUS_CONFIG["ETMX"]` +
> `damping_filter`): per-DOF velocity damper `k_d·s/(1+s/2π·8)` × per-DOF LP (L: hand-placed
> zeros/poles; T/V/R/P/Y: cheby1), production gains (L −1000, T/V −3000, R −10, P −3, Y −100),
> aLIGO connectMatrix sign (positive feedback, negative gains). `darm_actuation` now carries
> `velocity_damper`/`damping_filter`/`etmx_m0_damping_filters` and `DampedQuadCompliance` closes the
> loop exactly in the frequency domain; `hierarchical_stage_shapes` returns the three damped
> compliances. (2) **Drop `STAGE_GAINS` from the loop.** The twin's offload runs in FORCE units and
> explicitly does NOT use the drive-referred `STAGE_GAINS` (lib.py:90); the compliances already
> carry the relative stage strengths, so folding them in double-counted and was the real cause of
> the wrong crossover. With κ_i=1: **PUM/TST crosses at 10.34 Hz (≈F_EP=10)** as a single clean
> crossing, and **M0/PUM clusters at ~0.5–0.63 Hz (≈F_PT=0.5)** — a small residual wiggle because
> the real L-DOF M0 damping is loose (the M0 sensor barely feels the slow L mode; the twin flags
> this on its ETMX page), faithful to the plant. The earlier "~4 Hz caveat" was wrong — it was the
> STAGE_GAINS double-count, not the closed loop. Tests `tests/test_darm_hierarchical.py`
> (`test_crossovers_land_at_design_targets`, `test_etmx_damping_filters_match_twin_design`).
> Plot: `docs/darm_demo.py::hierarchical_actuation` → `docs/assets/darm_hierarchical_actuation.svg`,
> wired into `docs/examples/08-darm-calibration.qmd` (§Hierarchical actuation).
>
> **FUTURE UPGRADE (WiP).** Replace the reproduced L1 ETMX M0-only velocity damping with the new
> **damped quad plants vendored by the `ssprescale` package** — a properly-scaled, multi-loop
> (M0 + L1 + L2) damped suspension model. Then the in-band compliances and the exact crossover
> frequencies come from the production damping design rather than the single-loop M0 stand-in used
> here. Swap point: `darm_actuation.DampedQuadCompliance` / `etmx_m0_damping_filters`.

**Status:** planning only (no code changed by this note). Phase-1 twin.
**Scope:** two roadmap items from `notes/darm-cal-progress-2026-07.md` §Next —
(1) fold the committed reduced-order QUAD suspension plant into the DARM loop, and
(2) replace the single real "DARM pole" with an SRC-detuning optical spring whose pole
migrates real → complex under one physical knob δ, tracked by the existing time-varying
machinery.

---

## Context (what exists today)

- **DARM loop** — `src/system_ident/darm.py`, class `DARMLoop`. Closed-loop twin
  `d_err = C·(x_free + Σ κ_i N_i c_i)/(1+G)`, `G = C·A·D`, synthesised in the frequency
  domain (`DARMLoop.simulate`, `frf_pcal`, `frf_stage`, `R = (1+G)/C`).
  - **Sensing** `C(f) = g_c/(1+i f/f_cc)·e^{-i2πfτ}` — `sensing_model()`, a **single real
    cavity pole** `f_cc≈360 Hz`. No optical spring.
  - **Actuation** `A(f) = Σ stage(name,f)`, each stage a **placeholder single pendulum
    resonance** `_pendulum_stage(f_pend,q,gain) = TFModel.from_resonances([(f_pend,q)],gain)`;
    `default()` uses UIM(0.43 Hz)/PUM(1.0 Hz)/TST(3.4 Hz), strengths κ = 1.0/0.40/0.08.
  - **Servo** `D = G/(A·C)` is **derived** from a *designed* open-loop shape `_ol_shape`
    (integrator to UGF≈50 Hz + rolloff pole + delay). **Key invariant:** because `D` absorbs
    `A` and `C`, the *closed-loop* FRFs (`frf_pcal`, `frf_stage`, `R`) are invariant to the
    **absolute scale** of `A` and depend only on the **shapes** of `N_i(f)` and `C(f)` and on
    the designed `G`. This makes both upgrades low-risk drop-ins for loop stability.
  - `with_params(**overrides)` clones the loop at a drifted operating point (scalar fields +
    `kappa_<STAGE>`); this is the hook the TV tracker drives.
- **Reduced QUAD plant** — `src/system_ident/reduced_plant.py::ReducedStateSpacePlant`,
  committed as `models/quad_reduced_50hz.{npz,json}` (59-state modal truncation of the aLIGO
  quad, cutoff 50 Hz). `.eval(freq) -> (F,n_out,n_in)`, `.subplant(sensors,actuators)`,
  `.modes() -> [(f0,Q)]` (the oracle). Inputs/outputs are labelled
  `{M0,L1,L2,L3,toposem}.drive.{L,T,V,Y,P,R}` / `.disp.…`.
  - **L3 = test mass**; **longitudinal (`.L`) is the DARM DOF.** Verified read-only: the L-chain
    `L3.disp.L / Li.drive.L` carries longitudinal pendulum resonances at **0.428, 0.527, 0.999,
    1.997 Hz** (all below the 10–1500 Hz DARM band) with the correct `~1/f²` in-band tail;
    `L3.disp.L/L3.drive.L` DC ≈ 2.65e-3, `L3.disp.L/M0.drive.L` DC ≈ 3.53e-4 (twin units, m/N).
- **DARM backend** — `src/system_ident/backends/darm_adapter.py::DARMBackend` (unchanged by
  either upgrade; it only injects/reads through `DARMLoop.simulate`).
- **TV tracker** — `src/system_ident/darm_tv.py`: `snapshot_kappa` → `track_kappa` →
  `fit_tv` (Legendre/Fourier weighted-LS basis, `TVFit.predict` gives θ(t), θ̇(t) with CRB) →
  `resolvability`. `recover_actuation` uses Pcal as the ruler.
- **Fit/CRB utilities** — `fit_sensing` (weighted complex LS of C with `(JᵀJ)⁻¹` CRB) in
  `darm.py`; `resonator.py::ResonatorModel` (a `(f0,Q,gain)` biquad with `.eval/.jacobian/
  .with_params`), `resonator_design.py::fisher_information` (gauge-free Fisher/CRB for any
  model exposing that 4-method protocol), `fisher.py` (TFModel-gauge Fisher + dispersion for
  optimal excitation). `mimo_fit.py` / `mimo_modal.py` are the MIMO modal fitters used by the
  reduced-quad demos (`docs/reduced_quad_demo.py`, `…_closed_demo.py`).
- Demos to mirror: `docs/darm_demo.py` (`pcal_audit`, `actuation_campaign`, `comparison`),
  `docs/darm_tv_demo.py` (`campaign`, `drift_fig`). Tests: `tests/test_darm.py` (≈15),
  `tests/test_darm_tv.py` (7).

---

## Upgrade 1 — Reduced QUAD plant as the DARM mechanical response

### Goal
Replace each stage's placeholder single-resonance `N_i(f)` with the **real reduced-quad
longitudinal column** `N_i(f) = [L3.disp.L / L_i.drive.L]`, so the test-mass mechanical response
in DARM is the committed reduced-order quad chain, not a lumped pendulum.

### DOF / channel mapping (the physics)
- **DARM displacement** ↔ `L3.disp.L` (test-mass longitudinal).
- **Actuation stages** (drive counts → test-mass displacement) — one reduced-quad column each:
  - `UIM` → `L1.drive.L → L3.disp.L`
  - `PUM` → `L2.drive.L → L3.disp.L`
  - `TST`/ESD → `L3.drive.L → L3.disp.L`
- **Pcal / free length** enter as displacement at the test mass (`L3.disp.L`), unchanged — Pcal
  radiation pressure and `x_free` both act at L3 longitudinal. (No reduced-plant change needed
  for the Pcal path; it stays the reference ruler.)

### Design (files/functions)
1. **New shape provider** (add to `darm.py`, or a small new `darm_plant.py` imported by it):
   a duck-typed wrapper exposing `.eval(freq)` so it drops into the existing `stage()` /
   `recover_actuation` paths without touching their signatures:
   ```python
   class ReducedStageShape:            # 1-in/1-out complex shape from the reduced quad
       def __init__(self, sub, out_idx, in_idx): ...
       def eval(self, freq): return self._sub.eval(freq)[:, self._oi, self._ii]
   ```
   Build once from `ReducedStateSpacePlant.load("quad").subplant(
   sensors=["L3.disp.L"], actuators=["L1.drive.L","L2.drive.L","L3.drive.L"])`; cache the
   `subplant` (its `.eval` loops a 59×59 solve per bin — fine for a few 2049-bin grids, but
   memoise per `freq` array id to keep snapshots cheap).
2. **New constructor** `DARMLoop.default_reduced()` (sibling of `default()`): populate `stages`
   with `(ReducedStageShape(...), κ_i)` instead of `_pendulum_stage(...)`. Keep the **same
   dict/tuple contract** `name -> (shape, kappa)` so `stage()`, `A()`, `with_params(kappa_*)`,
   `snapshot_kappa`, `recover_actuation` are all **unchanged** — they only call `.eval(freq)`
   and read the scalar κ.
3. **Normalisation / coil gain.** Fold a per-stage constant `g_coil,i` (counts→force) into the
   shape or κ so nominal κ_i stay O(1) and `|A(f)|` keeps its representative in-band scale.
   Because `D = G/(A·C)` is derived, the **absolute** choice does not affect closed-loop FRFs
   or stability — pick `g_coil,i` to match today's `|stage_i|` at a mid-band reference (e.g.
   100 Hz) so plots and κ semantics carry over. Document the choice in the constructor.
4. **`R`, `G`, `D` untouched** — they compose `A` and `C` exactly as now; only the `N_i` shapes
   feeding `A` and `frf_stage` change.

### Why it's safe
`frf_stage = C·κ_i·N_i/(1+G)` with `G` held at its designed shape; swapping `N_i` changes the
stage numerator shape (adds the true multi-resonance phase and the cross-stage structure) but
not loop stability. In-band (10–1500 Hz) the quad modes sit below the band, so the change is
mainly correct phase / `1/f²` tail and correct relative stage shapes; extending `fmin` later
exposes the real modes.

### Tests / demos
- `tests/test_darm.py`: add a `default_reduced()` variant of `test_stage_frf_identity`,
  `test_G_equals_A_D_C_by_construction`, `test_loop_is_stable_with_margin`; assert the L-chain
  modes (0.428/0.527/0.999/1.997 Hz) appear in `N_i(f)` below band.
- `tests/test_darm_tv.py`: re-run `snapshot_kappa`/`track_kappa` on `default_reduced()` — the
  ruler cancellation (`H_stage/H_pcal = κ_i N_i`) still recovers κ within its CRB.
- Optionally add a `docs/darm_demo.py` panel overlaying placeholder vs reduced-quad `|N_i(f)|`.

---

## Upgrade 2 — DARM pole splitting: SRC-detuning optical spring (one knob δ)

### Goal
Replace the single real sensing pole with a **detuned-SRC optical-spring biquad** parameterised
by one physical knob δ (SRC detuning phase) such that, as δ grows, a **real pole pair splits
smoothly into a complex-conjugate pair** (the optical-spring resonance lifting off the real
axis). Suitable for `darm_tv.py` to track δ(t).

### Functional form (a quadratic in s whose discriminant changes sign)
Multiply the existing sensing by an optical-spring factor (Cahillane-style sensing model — a
biquad on the cavity-pole response):

```
C(f;δ) = g_c · ωs²(δ) / ( s² + b·s + ωs²(δ) ) · 1/(1 + s/ω_cc) · e^{-sτ},   s = i2πf
   ωs²(δ) = K · sin(2δ)          # radiation-pressure optical-spring stiffness ∝ sin(2·detuning)
   b       = 2π f_cc  (or a free optical-damping term)   # optical/cavity damping
```

- The optical-spring **denominator** `s² + b s + ωs²(δ)` has discriminant `b² − 4ωs²(δ)`.
- **δ small** → `ωs²` small → discriminant > 0 → **two real poles** (one near DC, one near the
  cavity pole) → in-band `C` ≈ today's single-pole response (smooth δ→0 limit; keep the
  explicit `1/(1+s/ω_cc)` cavity factor so δ=0 reproduces `sensing_model` exactly).
- **δ large** → `ωs²` large → discriminant < 0 → **complex pair** at
  `f_s = √ωs²/2π`, `Q_s = √ωs²/b` → optical-spring resonance in band.
- **Split** at `ωs(δ_c) = b/2`. Numerically verified (read-only) with `f_cc=360`, `b=2πf_cc`,
  `K` scaled to `f_s≈300 Hz @ δ=20°`:

  | δ | poles (Hz) | regime |
  |---|---|---|
  | 0.5° | −353, −6.9 | real |
  | 5° | −270, −90 | real |
  | 10° | −180 ± 124j | complex, f_s=219, Q=0.61 |
  | 20° | −180 ± 240j | complex, f_s=300, Q=0.83 |
  | 45° | −180 ± 328j | complex, f_s=374, Q=1.04 |

  → a clean, continuous real → collision (≈7–8°) → complex locus under the single knob δ.

### Connection to real detuned-SRC physics
`ωs² ∝ sin(2δ)` is the textbook detuned-cavity radiation-pressure spring constant (stiffness
vanishes on resonance δ=0, grows with detuning); `b` is the optical damping set by the
coupled-cavity linewidth (`~f_cc`). This reproduces the RSE picture in the handoff note
(`~395 Hz` real pole → complex; ± detuning phase). The unstable-spring / anti-damping regime
(`b<0`, poles into the RHP) is **out of scope** for now (notes: "sign/unstable-spring regime not
critical yet") — keep `b>0`, `ωs²≥0`, poles in the LHP.

### Design (files/functions)
1. **`darm.py`:** add `optical_spring_factor(freq, f_cc, delta, *, K, b)` returning the biquad,
   and `sensing_model_detuned(freq, g_c, f_cc, tau, delta, K, b)` = existing `sensing_model`
   × factor. Add fields `delta` (and spring constants `K`, `b` or a normalised `f_s0`) to
   `DARMLoop`; route `C()` through the detuned model when `delta` is set (δ=0 → identical to
   today, preserving all current `test_darm.py` assertions).
2. **`with_params`:** allow `delta` (already handled as a scalar field — just add it to the
   dataclass), so the TV tracker can drift δ like any scalar. Add a `δ → (f_s,Q_s)` helper and
   its inverse for reporting the pole locus.
3. **Snapshot estimator** (`darm.py`, extend the sensing path): δ is a **sensing** parameter, so
   — unlike κ — it does **not** cancel in `H_stage/H_pcal`. Identify it from the **Pcal** FRF
   shape (as `darm_demo.pcal_audit` already forms `C_meas = H·(1+G)`):
   - **Option A (extend `fit_sensing`):** add `(f_s, Q_s)` (or δ directly) to the weighted
     complex-LS parameter vector; CRB from `(JᵀJ)⁻¹`. Smallest change, reuses the existing fit.
   - **Option B (`ResonatorModel` + `resonator_design.fisher_information`):** fit the biquad
     pole pair as a `(f0,Q,gain)` resonance on `C_meas`, then map `(f_s,Q_s)→δ`. Better
     conditioned near the split and gives a gauge-free CRB; reuses `resonator.py`. **Preferred.**
   - Place excited multisine lines near `f_s` using `fisher.py::dispersion` (optimal excitation)
     so δ is maximally informative — the CLAUDE.md feasibility discipline.
4. **`darm_tv.py`:** add `snapshot_delta(base_loop, delta_value, …)` (mirror of `snapshot_kappa`
   but returning `(delta_hat, sigma_delta)` from the Pcal-FRF biquad fit) and reuse
   `track_kappa`'s loop / `fit_tv` / `resolvability` verbatim for δ(t) — the basis fit and CRB
   are parameter-agnostic. `drift_profile` already supplies a known δ(t) truth.

### Tests / demos
- `tests/test_darm.py`: `test_optical_spring_split_locus` — sweep δ, assert discriminant sign
  flips at δ_c and poles go real→complex; assert `delta=0` reproduces `sensing_model` to
  machine precision (protects existing tests).
- `tests/test_darm_tv.py`: `test_tracks_injected_detuning_within_crb` — inject δ(t), snapshot,
  `fit_tv`, assert pull ≈ O(1) and `resolvability.resolve_ratio > 1` (compute the bound, per
  CLAUDE.md — do not call it a limit without the number).
- New demo panel (mirror `docs/darm_tv_demo.py`): pole-locus-vs-δ (root-locus in the s-plane)
  and `|C(f;δ)|` fan; δ(t) tracked ± CRB.

---

## How the two upgrades interact (both live in the DARM loop)
`d_err = C(f;δ)·(x_free + Σ κ_i N_i^{quad}(f) c_i)/(1+G)`, `G = C(f;δ)·A^{quad}(f)·D`.
- Upgrade 1 sets the **mechanical** side (`A`, the stage FRF numerators `N_i`).
- Upgrade 2 sets the **optical** side (`C`, the δ-dependent DARM pole).
- They are **separable in identification:** κ_i are recovered ruler-style (`H_stage/H_pcal`,
  where `C` — hence δ — cancels), so the mechanical fit is immune to δ drift; δ is recovered from
  the Pcal FRF shape, where the reduced-quad `A` is absorbed by the derived `D` and does not bias
  the sensing biquad. Because `D = G/(A·C)` re-derives against whatever `A`, `C` are installed,
  the loop stays consistent and stable for any (reduced-quad, δ) combination.
- Joint drift (κ_i and δ together) is roadmap item 3 in the progress note; this plan leaves the
  two snapshots independent (κ via the ruler, δ via Pcal), which already supports it.

## Verification approach (end-to-end)
1. **Upgrade 1:** `G = A·D·C` identity and loop margin hold on `default_reduced()`; L-chain
   modes (0.428/0.527/0.999/1.997 Hz) present in `N_i`; `snapshot_kappa` recovers κ within CRB
   through the reduced plant. Physical sanity: relative UIM/PUM/TST in-band magnitudes and the
   `1/f²` slopes match the quad columns.
2. **Upgrade 2:** pole-locus-vs-δ is continuous and crosses real→complex at the predicted δ_c;
   `f_s(δ)`, `Q_s(δ)` monotone as tabulated; `delta=0` byte-matches the old sensing; δ(t)
   recovered within the fit CRB with `resolve_ratio > 1` and `local_stationarity_err` reported.
3. **Combined:** run `pcal_audit` + `actuation_campaign` on the reduced-quad, detuned loop;
   confirm `R = (1+G)/C` still recovered within its measured CRB envelope and the swept-sine
   head-to-head (`comparison`) still runs.
4. **Gate (CI down — GitHub billing lock, see progress note):**
   `conda run -n sysid pytest` and the quarto render locally.

## Open questions for the user
1. **Stage↔mass map:** confirm UIM↔L1, PUM↔L2, TST/ESD↔L3 for the reduced-quad columns (vs any
   M0/R0 upper-stage convention).
2. **Coil-gain normalisation:** anchor `|A(f)|` to today's scale at a reference frequency, or
   adopt physical counts→force per stage (do we have representative ESD/coil gains)?
3. **Optical-spring constants:** target `f_s` and `Q_s` range for the nominal detuning (the note
   mentions ~395 Hz RSE) — set `K`, `b` to hit these; is the operating point far below or near
   the split δ_c?
4. **δ definition:** track δ as detuning **phase** (used here, `ωs²∝sin2δ`) or map to an SRC
   length/frequency offset for the CDS-facing story?
5. **δ estimator choice:** Option B (`ResonatorModel` pole-pair fit) preferred — OK, or keep
   everything inside `fit_sensing` (Option A)?
6. **Band:** keep `fmin=10 Hz` (quad modes stay out of band, upgrade-1 gain is mostly phase), or
   lower it to exercise the real reduced-quad resonances?

> **UPDATE 2026-07-31 (coupled SRC sensing + Fisher-optimal cal-line design).** Rana: "when SRC
> is detuned the f_cc splits into a complex pair" — the factorized `optical_spring_factor` was
> wrong (separate low-f spring, f_cc fixed). Replaced with the coupled detuned-cavity response
> `C = g_c/(1 + i f/f_cc − A·sin(2δ)·(f/f_cc)²)·e^{-2πifτ}`: δ=0 byte-matches the single pole; the
> pole splits real→collide→complex at δ_c≈7° (A=1.033); anti-spring side (δ<0) puts a pole in the
> RHP. `optical_spring_factor→coupled_cavity_factor`, `+coupled_cavity_poles`, `fs2/spring_pole/
> spring_K/spring_Q → alpha()/cavity_poles()/detune_coupling`; `snapshot_delta` fits it;
> `test_darm_detuned` rewritten to check the denominator ROOTS split. (commit d9f2ea4)
>
> New `src/system_ident/darm_callines.py`: Fisher/CRB engine for the 7 TDCFs (κ_C, f_cc, δ, τ,
> κ_M0/PUM/ESD) measured by Pcal + one actuator line per hierarchical stage. `Γ=Σ 2 SNR² Re[∂lnH*∂lnH]`
> in log params (every observable ∝ C ⇒ ∂lnH/∂lnθ=∂lnC/∂lnθ for sensing, =1 for the line's own κ;
> θ-independent pieces precomputed, no plant re-solve). `size_lines_for_target` jointly optimises
> line freq+amp (differential evolution) to minimise the worst-param fractional σ; reports per-param
> T_req(0.1%) and feasibility (<5 min). `O3_LINES/O4_LINES`+`reference_scheme` for the head-to-head.
> **Result (representative twin, equal total drive): P&S-optimal reaches 0.1% on all 7 in ~90 s;
> the O3/O4 line placements need ~4.5×/5.5× longer, the gap concentrated in δ,τ (which LIGO
> monitors but does not correct to 0.1%); κ's are comparable.** Absolute time scales with drive
> (representative; paper amplitude/full-O4-list match deferred) — the ratio is scale-invariant.
> Tests `tests/test_darm_fisher.py` — **since deleted**; see the 2026-07-31 update below, which
> removed that engine and its tests in full. Docs: `docs/darm_demo.py` (cal_sizing/convergence_to_target_fig/
> scheme_bars_fig/sizing_table) + example 08 §"Sizing the lines" / §"Head-to-head with LIGO O3/O4".
> Sources: Sun 2020 (O3, dcc P1900245); Cahillane 2017 (PRD 96 102001); O4 arXiv:2508.08423.

> **UPDATE 2026-07-31 (response-error budget vs O3/O4).** Rana: "compare against the O3/O4
> systematic error budgets." Propagate the 7-TDCF CRB into the detector-response uncertainty
> δR/R(f): `darm_callines.response_log_jacobian` computes ∂lnR/∂lnθ with the digital servo D held
> FIXED (so a κ_i error moves A→G→R, matching the real pipeline — unlike the twin's derived-D
> invariant under which R is κ-insensitive); `response_budget` propagates the full covariance
> Σ=Γ⁻¹ → (σ_mag %, σ_phase °). Reference: Sun 2020 O3A budget (`O3_BUDGET`): total error+uncertainty
> <7%/<4° (68%, 20-2000 Hz), systematic-error floor <2%/<2°; O4 comparable (arXiv:2508.08423).
> **Result: at the 16 s design point (all TDCFs at 0.1%) δR/R ≈ 0.1%/0.1° — ~20-70× inside the O3
> budget; 5 min lower still. The cal-line statistics are NOT the limiting factor — the budget is
> systematics-dominated (Pcal 0.54% absolute, unmodelled low-f response, actuator-model error),
> none of which the twin includes.** `docs/darm_demo.py::response_budget_fig` + example 08
> §"Against the O3/O4 systematic-error budget"; test `test_response_budget_propagation`. Also fixed
> a `size_lines_for_target` bug (returned `lineset` was in pre-sort order vs the frequency-sorted
> `lines` → mismatched amps downstream); now rebuilt consistently.

> **UPDATE 2026-08-01 (reframe: random-error measurement-design study + amplitude↔time Pareto).**
> Rana clarified the thesis: the point is to reach the O3/O4 RANDOM (statistical) response-error
> levels with much LESS injected amplitude and time (gentler/faster), and to expose an amplitude↔
> time Pareto slider (down to a 0.1% stretch) for designing the cal strategy. Interview decisions:
> derive O3 random level from the papers (√(7²−2²)≈6.7% / √(4²−2²)≈3.5°; O4-class provisional =
> O3 systematic floor 2%/2°, final O4 budget forthcoming per 2508.08423); Pareto plane = time (x)
> vs amplitude/floor (y), iso-precision contours A(T)=√(K/T); amplitude referenced to the noise
> floor; baselines = O3/O4 fixed-line AND naive broadband.
> Key correction: `size_lines_for_target` minimises worst-PARAMETER σ, which is NOT response-
> optimal (naive/fixed beat it on δR/R). Added `size_lines_for_response` (minimise band-max
> combined response error ρ=√((σ_|R|/|R|)²+σ_φ²)) — the scheme for this study. Cost metric
> K=A²·T (`pareto_cost`, scheme-characteristic since δR/R∝1/√(A²T)); `naive_broadband`,
> `TARGET_LEVELS`, `rho_of_target`, `band_response_rho`.
> **Result: response-optimal P&S reaches the O3 random level with ~7× less injected energy than the
> O3/O4 fixed-line placement (≈2.6× less amplitude OR ~7× less time) and ~4× less than naive; the
> factor is target-independent.** Abstract rewritten to this thesis; budget section reframed
> ("Reaching the O3/O4 random-error levels — gently and fast") with the Pareto plot + savings table;
> `docs/darm_demo.py::pareto_campaign/pareto_fig/pareto_table`. Test
> `test_response_optimal_is_gentler_than_baselines`.

> **UPDATE 2026-08-01 (drift example on the new plant).** Example 13 (drifting DARM) moved off the
> old lumped `DARMLoop.default()` onto the **new coupled plant** `default_reduced(fmin=10,
> hierarchical=True)` (M0-damped reduced-quad M0/PUM/ESD + coupled detuned-cavity sensing).
> κ_TST(t) tracking now runs through the M0-damped compliance + nested-offload ruler (5% drift,
> ~0.05% per-snapshot σ). Added a δ(t) (SRC-detuning) drift-tracking section — the new physics the
> single-pole twin couldn't represent (the split cavity pole wandering): δ recovered from the Pcal
> FRF shape (independent of the κ drift; the two snapshots compose), ~0.004° per snapshot around a
> 5° operating point. `docs/darm_tv_demo.py::campaign_delta/delta_drift_fig/delta_resolvability_table`;
> example 13 abstract/gaps updated (optical-spring drift now DONE; remaining: stochastic/GP wander,
> joint multi-param fit, one-shot LP). Docs-only change; suite unchanged (298).

> **UPDATE 2026-08-02 (drift round 2: joint multi-param + stochastic wander).** Closed the two
> drift gaps. `darm_tv.stochastic_drift` (Ornstein–Uhlenbeck random wander, stationary std
> amp_frac·base, correlation time tau_s — realistic meandering vs the smooth `drift_profile`).
> `darm_tv.joint_snapshot` recovers SEVERAL drifting params at once (subset of g_c/f_cc/delta/tau
> + kappa_<stage>) by weighted complex LS of the measured Pcal + per-stage FRFs to the model
> (C/(1+G), C·κ_i·D_iN_i/(1+G)); returns θ̂, σ, and the correlation matrix. `track_joint` runs it
> over time. Fit auto-scaled (trf, x_scale='jac') for the g_c~1e6 / δ~0.09 / κ~1 disparity; DN
> (κ=1 stage shape) reused. **Finding: κ_C (g_c) and κ_ESD are anti-correlated ~−0.99 (both scale
> the ESD line; only the Pcal absolute reference separates them), while δ is clean (~0.1) — the
> joint fit surfaces the sensing/actuation degeneracy honestly.** All three random wanders recovered
> within CRB (pulls ~0.6–0.8). Example 13: added "Round 2: everything drifts at once, and at random"
> (joint_drift_fig + joint_corr_fig heatmap); reframed scope/gaps (joint+random now done; remaining
> = GP wander w/ measured spectrum, all-7 together, one-shot LP). Tests in test_darm_tv.py
> (stochastic OU std, joint untangling, track_joint shapes).

> **UPDATE 2026-08-02 (real DARM noise floor).** Replaced the two-scalar placeholder floor with a
> physical aLIGO displacement floor: `darm.darm_design_asd(freq)` = the aLIGO design strain
> sensitivity bucket (15-pt log-log table, 10–5000 Hz) × 4 km arm → m/√Hz. `DARMLoop.noise_asd`
> hook + `displacement_noise_asd`; `simulate` grows a colored-noise branch when it is set;
> `default_cal_loop`/`floor_asd` wire it into the Fisher engine; drives rescaled to a realistic
> Pcal displacement (`PX_REAL=A_TOT_REAL=5e-17` m) → per-record σ(R)/R≈0.8%.
> Cascade of fixes it forced: (a) hardened `fisher.safe_inverse` (nan_to_num + ridge fallback on
> pinv SVD non-convergence); (b) new `darm_callines._crb_cov` (eigenvalue-floored inverse) so an
> under-constrained param returns a large σ, not a spurious zero variance — used in `sigma`,
> `band_response_rho`, `response_budget`; (c) both sizers guard objectives with try/except→1e30.
> **Placement re-opt (Rana's call): optimize ALL line frequencies, unconstrained** — the real floor
> makes frequency matter, so lines may land in-band. Re-enabled `pcal=list(range(n))` in both
> `size_lines_for_target` and `size_lines_for_response`.
> **Results with the real floor (supersede the ~7×/2.6×/~4× above):** response-optimal P&S reaches
> every random-error target with **18.1× less injected energy than the O3/O4 fixed-line placement**
> (≈4.3× less amplitude OR ~18× less time) and **3681× less than naive broadband** — still
> target-independent. Cal-line sizing to 0.1% on all 7: **P&S 90 s (δ binds), O3 2543 s / O4 3919 s
> (κ_M0 binds, 28×/44× slower)**. The O3/O4 gap is concentrated in κ_M0/κ_PUM: their ~15–35 Hz
> actuator lines poorly measure the twin's top/penultimate stages (genuine authority-rolloff SNR
> effect + a stage-set caveat, M0/PUM/ESD twin vs UIM/PUM/TST papers) — stated honestly in ex-08.
> Prose in example 08 updated throughout (abstract, sizing, head-to-head, Pareto) to these numbers.
> 7 fisher tests pass.

> **UPDATE 2026-08-02 (real O4 noise floor, replacing the hand-typed design curve).** Rana asked
> for a real O4 floor. Vendored `aligo_O4high.txt` verbatim from LIGO-T2000012-v2 (the ~190 Mpc
> "O4 high" representative sensitivity, 2736 pts, 10–5000 Hz) at `src/system_ident/data/`; added
> `darm.darm_o4_asd(freq)` = strain × 4 km, log-log interp, lru-cached loader (`_o4_strain_table`).
> Repointed default_cal_loop + both demos' `_twin` to it; kept `darm_design_asd` (the optimistic
> design curve) for reference. package-data glob `data/*.txt` added to pyproject.
> Real O4 floor: best ≈1.2e-20 m/√Hz near 330 Hz, but a STEEP seismic wall below ~20 Hz
> (2.67e-17 at 10 Hz, ~1e-18 at 15–17 Hz) — much steeper than the design curve there.
> **Effect (supersedes the ~18× design-floor numbers):** because O3/O4 place their upper-stage
> actuator lines at ~15–17 Hz — right in the O4 wall — pinning κ_M0/κ_PUM there is slow; the
> unconstrained optimiser moves ALL lines (sensing AND actuator) into the ~110–340 Hz sensitive
> bucket. Pareto: **~400× less energy vs fixed-line** (20× amplitude / 400× time), ~10⁶× vs naive.
> Sizing to 0.1%: **P&S 90 s (δ binds) vs O3 5.3 h / O4 15 h (κ_PUM binds, 213×/588×)**.
> **HONESTY (Rana chose "unconstrained + caveat"):** the ~110–340 Hz bucket is exactly where
> operational cal avoids strong lines (protect the astrophysical band), so the 400×/200–590× is a
> pure-CRB CEILING, not an operational speed-up; a band-protecting constraint would give a smaller,
> fairer factor. Example 08 rewritten throughout with this framing + a callout-warning explaining
> the ceiling; the response Pareto is the robust, constraint-free thesis. My earlier "actuator lines
> follow the hierarchy (M0 low, PUM mid)" prose was WRONG under the real floor (optimum is all
> mid-band) — corrected. New test `test_darm_o4_asd_matches_the_vendored_curve`. Possible next phase:
> a band-constrained sizer to report the fair operational factor alongside the ceiling.

> **UPDATE 2026-08-02 (actuator-range model; band-protection framing RETRACTED).** Rana: "there is
> no reason to constrain the lines [out of the sensitive band]... the main issue is actuator range
> and parameter estimation." Correct — cal lines live in-band routinely (the 331.9 Hz sensing line).
> My "ceiling / keep the astrophysical band clean" caveat was WRONG and reflected a MODEL bug:
> `fisher()` used SNR = amp·√T/floor with `amp` the DARM displacement, treating any amplitude as
> free to obtain at any frequency — so the optimiser parked M0/PUM lines mid-band where no real
> actuator has the range. FIX: fold **actuator authority** into the achievable amplitude. LineSet
> now carries `authority` (per line); `_stage_authority(loop)` = |κ_i·D_i·N_i(f)|/max (real
> reduced-quad response, rolls off steeply for upper masses), cached on the loop; Pcal authority=1
> (ruler). SNR = amp·authority·√T/floor, so placement is governed by authority↔floor. Also: search
> fmin 0.3→10 Hz (floor data range; kills the vestigial sub-Hz lines); made
> `size_lines_for_response` accept optimize_freq=False and added `reference_scheme_response` so the
> Pareto fixed-line baseline is response-optimal in ALLOCATION at fixed O3/O4 freqs (fair — isolates
> placement). Removed the callout-warning + all band-protection prose from example 08.
> **New (correct) numbers, real O4 floor + actuator range:** Pareto **~9× less energy vs fixed-line**
> (3× amplitude / 9× time), ~3500× vs naive. Sizing to 0.1%: **P&S 90 s vs O3 37 min / O4 43 min
> (~25×/29×), binding param κ_M0 for everyone** — the top mass is the hard one (authority only at low
> f, where the O4 wall is worst). The P&S optimum now places actuator lines LOW (M0 ~10 Hz, PUM
> ~26 Hz, TST ~42 Hz) — close to where LIGO actually puts them — vindicating "no need to constrain."
> Advantage is modest, real, and a genuine actuator-range/CRB limit. New test
> `test_darm_o4_asd_matches_the_vendored_curve` already covers the floor; authority path exercised by
> the existing sizing/CRB tests. Honest gap: Pcal free-mass (1/f²) range not yet modelled
> (authority=1); absolute per-stage actuator ranges are issue #3 territory.

> **UPDATE 2026-08-02 (Pcal free-mass range folded in).** Rana: fold in the Pcal free-mass range
> too — "~200 mW peak-to-peak range in terms of power modulation." Made authority ABSOLUTE (meters
> of DARM displacement at full range) for ALL lines. `pcal_range_disp(f)` = radiation-pressure force
> F_rms=(P_pp/c)/√2 from ±200 mW on the 40 kg free test mass → x=F/(M(2πf)²) ∝ 1/f² (≈3e-17 m at
> 100 Hz) — REAL, no free parameter. Stage authority = |κ_i D_i N_i(f)| shape × representative range
> (_STAGE_RANGE_M anchored to pcal_range_disp(20 Hz); stage absolute ranges still issue #3). `amps`
> are now DRIVE FRACTIONS ∈[0,1] of range; `‖drive‖₂ = A_tot = 1` ⇒ no actuator exceeds its range
> (verified: max drive 0.93–0.97). Pareto y-axis switched from amplitude/floor → drive-fraction with
> a hard "actuator-range limit" line at 1. pareto_table + captions reworded amplitude→drive.
> **Result (Pcal range now genuinely binding):** Pareto **~2× less drive-energy vs fixed-line** (1.5×
> drive / 2× time), ~4× vs naive. Sizing to 0.1%: P&S 90 s vs **O3 7.6 min (5.1×) / O4 4.8 min
> (3.2×)**; binding TDCFs τ (needs high-f Pcal, where the ±200 mW 1/f² range rolls off) and κ_M0
> (top-mass range vs seismic wall). **Big-picture: with real floor + real ranges, LIGO's line
> placement is already near-optimal; P&S gives a modest, real ~2× and, more usefully, identifies the
> true actuator-range limits.** Example 08 rewritten throughout (abstract/what-you'll-learn/sizing/
> head-to-head/Pareto) to this honest, modest framing. Tests: fixed test_single_stage_line CRB to
> include authority; added test_pcal_free_mass_range + test_lineset_authority_is_absolute. Honest gap
> remaining: Pcal beam angle (cosθ≈1 assumed), absolute stage ranges (issue #3).

> **CORRECTION 2026-08-02 (fabricated cal-line design results REMOVED).** The primary author found
> that the per-stage suspension actuator ranges underpinning the cal-line design/sizing narrative
> above were **invented numbers**, not real hardware values (`darm_callines._STAGE_RANGE_M` and the
> absolute stage-"authority" machinery it fed). Without real per-stage ranges the actuation-side
> sizing and every cross-scheme comparison cannot be computed, so they were removed rather than
> caveated. Removed as fabricated (and every conclusion built on them):
> - the invented `_STAGE_RANGE_M` and `_stage_authority` machinery, and the whole Fisher-optimal
>   *sizing/design* engine that depended on it (`build_lineset`/`LineSet.authority`, `fisher`,
>   `sigma`, `size_lines_for_target`, `size_lines_for_response`, `reference_scheme[_response]`,
>   `naive_broadband`, `pareto_cost`, `band_response_rho`, `response_budget[_log_jacobian]`,
>   `TARGET_LEVELS`, `O3_BUDGET`, seeding/allocation helpers) in `src/system_ident/darm_callines.py`;
> - the representative drive budget `A_TOT_REAL` in `docs/darm_demo.py` and the demo functions that
>   produced the fabricated absolute/comparative results (`cal_sizing`, `convergence_to_target_fig`,
>   `sized_lines_fig`, `scheme_bars_fig`, `sizing_table`, `response_budget_fig`, `pareto_campaign`,
>   `pareto_fig`, `pareto_table`, and the `_sigma_curves` helper);
> - the example-08 sections built on them — "Sizing the lines…", "Head-to-head with LIGO O3/O4"
>   (incl. the "help us make this exact / issue #3" callout), and "Reaching the O3/O4
>   random-error levels" (the Pareto) — plus every "~N× less energy/drive/time", "reaches 0.1% in
>   ~90 s / ~7.6 min / ~4.8 min", and "LIGO's placement is (near-)optimal" claim in the abstract,
>   description, and "What you'll learn";
> - the tests that only exercised the removed engine (`tests/test_darm_fisher.py` in full, and
>   `test_lineset_authority_is_absolute…` in `tests/test_darm_callines.py`).
> **KEPT (genuine provenance):** the real O4 noise floor (`darm_o4_asd`, LIGO-T2000012) and its test;
> the Pcal ±200 mW free-mass range (`pcal_range_disp`, `PCAL_POWER_PP_W`/`TEST_MASS_KG`) and
> `test_pcal_free_mass_range`; the published `O3_LINES`/`O4_LINES` frequencies; and the
> method-demonstration sections of example 08 (recovery of C / κ / R(f) with CRB, the `1/√T`
> convergence, the hierarchical crossovers, the cal-line SNR spectrum, and the swept-sine
> head-to-head), whose shapes do not depend on the (representative) drive magnitude. The earlier
> UPDATE entries above are left intact as the historical log; the numbers they report are the
> fabricated ones this correction retracts.

---

## UPDATE 2026-08-03 — joint P&S-optimal drift tracking, grounded (example 13)

Rebuilt the tracking work on the corrected framing (one plant = opto-mech DARM sensing `C` + the
quad suspension; one joint θ; a **few** optimally-placed cal lines, NOT broadband; design from
drift-variance priors to minimise the joint estimator variance on the drifts).

New engine (`src/system_ident/darm_callines.py`): `joint_fisher` (coupled TDCF Fisher/CRB — every
line informs every param), `design_lines` (Bayesian A-optimal placement, `tr((Γ'+I)⁻¹)`, under
force caps), `stage_force_caps` (DERIVED, ruler-matched), `line_displacement`, `pcal_budget_crosscheck`.
Readout (`src/system_ident/darm_tv.py`): `joint_snapshot_lines`/`track_joint_lines` inject the
DESIGNED lines leakage-free and fit θ jointly. Tests in `tests/test_darm_callines.py`.

Grounding (all via `provenance.record`): κ_C drift 1–2 % (Sun 2020 §4.2, PAPER); Pcal ±200 mW
(`pcal_range_disp`); O4 Pcal line freqs 17.1/33.43/53.67/77.73/102.13/284.01/410.3/1083.7 Hz (Wade
2025 Fig. 4); 17.1 Hz line ≈4e-19 strain/√Hz (Wade Fig. 2, cross-checks the 200 mW budget to a
factor <2). The papers give NO per-stage line amplitudes or per-param drift for δ/f_cc/τ/κ_i → those
are placeholders kept in the docs demo layer (src provenance gate stays green).

Honest scope: the twin's **M0 is the top mass, offloaded <0.5 Hz** (unlike LIGO's UIM), so κ_M0 is
not identifiable with ≥10 Hz lines; f_cc/τ need a wideband high-f Pcal spread. Identifiable joint set
(O4 floor, ≥10 Hz) = **κ_C, δ, κ_PUM, κ_ESD** — recovered within CRB from the designed lines; the
designed drive beats a same-budget broadband multisine ~1000× on the A-cost. The rest are
feasibility-gated (the CRB states what each needs). Example 08's fabricated pareto/response-budget
section was already fully reverted from source (confirmed; `_site` is stale local build only).

### Addendum 2026-08-03 — the 10 Hz band was arbitrary; the real limiter is the seismic wall

Rana asked why lines were capped at 10 Hz (M0 could go lower). Lifted the design band to **1 Hz**.
Investigating honestly: the O4 curve stops at 10.2 Hz, and a flat clamp below it is wrong — the
seismic wall keeps rising (measured local slope ≈ **f^−7.36**). Added `darm.darm_o4_asd_seismic`
(extrapolates that wall below 10 Hz, held flat below 1 Hz for finiteness) as the **design/placement
floor**, while `darm_o4_asd` stays **clamped** as the **simulation** floor (a huge sub-10 Hz noise
injection leaks broadband within a 1 s analysis period; the two floors are identical ≥10 Hz where
every designed line lands, so the sim is exact at each line). Objective changed from the Bayesian
posterior to the **data-A-optimal** `Σ 1/margin²` (a ridge, not the prior, keeps it finite) so a
data-degenerate parameter blows the cost up instead of being propped up by the prior; added per-line
frequency bounds (each stage line confined to its authority window).

Result on κ_M0: with the honest wall, the top mass is **squeezed out both sides** — the f^−7 seismic
wall buries its line below ~10 Hz, and its own hierarchical offload rolls its DARM authority off as
≈ f^−8 above ~10 Hz. The analytic Fisher is optimistic for this offloaded stage, but a leakage-free
simulation cannot recover κ_M0 anywhere in band, so it is **not** claimed (documented; it would need
a sub-Hz line where the detector is deaf). The identifiable tracked set stays **κ_C, δ, κ_PUM,
κ_ESD** (recovered within CRB). Lesson reinforced: the simulation is the arbiter; don't ship an
analytic CRB the sim won't back up.
