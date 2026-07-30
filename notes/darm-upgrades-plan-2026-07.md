# DARM calibration — two fidelity upgrades: implementation plan (July 2026)

> **UPDATE 2026-07-29 (review + Upgrade 1 implemented).** After a code-verified review:
> - **Upgrade 1 is DONE** — `DARMLoop.default_reduced()` + `ReducedStageShape` in `darm.py`,
>   tests in `tests/test_darm_reduced.py` (6, green; full suite 270 pass). Stages are the real
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
> `tests/test_darm_callines.py` (4); full suite 282 pass. (The placeholder `default()` loop keeps
> its UIM/PUM/TST toy stages.)

> **UPDATE 2026-07-29 (twin offload filters wired in).** `default_reduced(hierarchical=True)` now
> populates `DARMLoop.distribution` with the nested-offload filters reproduced from
> `digital_twin/twin/experiments/cavity_arm_lsc_hierarchical/lib.py::offload_filters`
> (`src/system_ident/darm_actuation.py`, FRF-identical — verified 0.0 diff): `D_TST=1`,
> `D_PUM=O_A`, `D_M0=O_A·O_B`, with κ = the twin's per-stage authorities `STAGE_GAINS`
> (M0/TOP 334.3, PUM 1.0, TST/ESD 0.001697). The strong, slow M0 dominates DARM actuation at low
> f and hands off up the chain, so adjacent stages cross over (measurable with cal lines).
> `snapshot_kappa` made distribution-aware (rules by the full `D_i·N_i` shape). Caveat: the offload
> *filters* are exact-from-twin, but the labeled F_EP/F_PT=10/0.5 Hz crossovers belong to the
> twin's full nested-offload *closed loop* — in this simplified derived-servo loop the M0/PUM
> actuation crossover lands ~4 Hz. Tests `tests/test_darm_hierarchical.py` (6); suite 288 pass.

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
