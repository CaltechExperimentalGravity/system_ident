# Joint MIMO parametric fit — common-denominator SML + Cramér–Rao bound

**Status:** approved design, pre-implementation
**Date:** 2026-06-22
**Arc:** Step 2 of "joint MIMO identification of closed-loop systems" — the headline
SEI/ISC commissioning need (see `.llm/roadmap.md`).
- **Step 1 (done, on `main`):** generic coupled+closed-loop twin + nonparametric per-bin
  matrix-inverse recovery `G(f) = Y_mat·X_mat⁻¹`.
- **Step 2 (this spec):** the joint *parametric* fit — one common-denominator MIMO model
  with **shared modal poles** across all elements, fit by Pintelon–Schoukens **sample-ML
  (SML)**, with the **Cramér–Rao bound** on the modal parameters and the FRF.
- **Step 3 (named follow-on):** demonstrate the same fit on the RTSfreerun compiled twin
  (`x1hsts6dof`). The fit is backend-agnostic so step 3 is a demonstration, not a rewrite.

Phase 1 (RTSfreerun) only — real hardware is Phase 2, explicitly out of scope
([[two-phase-cds-plan]]).

**This spec implements P&S's *literal* method.** The procedure, equations, and page
citations are scraped into **`.llm/pintelon-schoukens-mimo-fit.md`** (from
`docs/SysID-Pintelon.pdf`, 2nd ed.) — read it alongside this spec; it is the authoritative
derivation and this spec cites it throughout. Earlier (pre-book) framings of step 2 (a
"recover G then fit G" two-stage, modal/vector-fitting parameterization, variable
projection) are **superseded** — see §9. [[stay-on-pintelon-schoukens]].

## 1. Goal

Fit a **single common-denominator MIMO transfer-function model** `G(Ω,θ) = B(Ω,θ)/A(Ω,θ)`
— a **scalar denominator `A`** carrying the **shared modal poles** and a **polynomial-matrix
numerator `B`** with **free per-element numerators** — **directly to the measured
input/output DFT spectra** (`X_mat`, `Y_mat`) and their period-to-period sample covariances,
by minimizing the P&S **sample-ML equation-error cost** (12-15). Report the **Cramér–Rao
bound**: the parameter covariance `(2 Re(JᴴJ))⁻¹` with the SML finite-sample inflation,
propagated to **per-mode `f0`/`Q`** (roots of `A`) and to a **per-bin FRF uncertainty band**
on each `G_ij(f)`.

The shared denominator is constrained by **all excited bins across all elements jointly**,
so the parametric model **predicts through the resonances** — where step-1's per-bin matrix
inverse is ill-conditioned. This is the regularization step 1 deferred. It serves SUS, SEI,
ASC, LSC: it turns the nonparametric coupled FRF into a modal model with calibrated
uncertainty — `f0`, `Q`, mode shapes, and confidence bands — which SISO-per-DoF tooling
cannot give a commissioner.

## 2. The model — common-denominator MIMO (wiki §1; P&S §6.6, eq. 6-53)

```
G(Ω,θ) = B(Ω,θ) / A(Ω,θ) = ( Σ_{r=0}^{n_b} B_r Ω^r ) / ( Σ_{r=0}^{n_a} a_r Ω^r )
```

- **`A`** — scalar common-denominator polynomial, real coeffs `a_r` (the shared poles).
- **`B_r ∈ ℝ^{n_sens × n_act}`** — polynomial-matrix numerator (one free numerator per I/O
  element).
- **`Ω = s`** (continuous-time, the suspension is a lumped CT mechanical plant). A discrete
  `Ω = z⁻¹` variant is allowed but not required for v1 (§7 notes the tustin twin).
- **Parameter vector:** `θ = [vecᵀ(A_0)…vecᵀ(A_{n_a})  vecᵀ(B_0)…vecᵀ(B_{n_b})]ᵀ`, `A_r`
  scalar.
- **Identifiability constraint:** `‖θ‖₂ = 1` (one-dimensional, sufficient for the
  common-denominator model; P&S §12.3.5).

**Why common-denominator and not rank-1 / modal-residue (P&S Remark (iii) p.194 + App
6.N):** partial-fraction/matrix-fraction/state-space residue matrices are rank-1 **only**
for proportionally-damped reciprocal structures; the common-denominator model carries
**full-rank** residues. Our plant has deliberate cross-coupling + frequency-dependent
`M_out`, so rank-1 is not guaranteed — P&S's rule is then to **use the common-denominator
model**. We also do **not** use the modal partial-fraction parameterization (6-22): P&S note
it has poor starting values and "gets stuck in local minima."

**Model order** `(n_a, n_b)`: `n_a = 2·M_modes` (a conjugate pole pair per mode);
`n_b ≤ n_a` per element. The order is a user input; §6 validates it (under/over-modeling
tests). For the twin, `M_modes` is known (it is built from a known modal expansion).

## 3. The data — periodic multisine, sample means + covariances (wiki §2; P&S §12.3.1)

Reuses step-1's measurement campaign verbatim — **no new excitation or estimation method**
([[stay-on-pintelon-schoukens]]):

- **Robust method, `n_exp = n_act`:** drive **each actuator separately** with the periodic
  multisine (step-1 already does exactly this — sequential per-actuator campaigns).
- Per experiment `l` (driven actuator), over `M` steady-state periods at each excited bin
  `k ∈ 𝕂`: the **sample-mean** input spectrum `Û^[l](k)` (the `n_act` drive monitors =
  one column of `X_mat`), the **sample-mean** output spectrum `Ŷ^[l](k)` (the `n_sens`
  sensors = one column of `Y_mat`), and the **stacked `(n_sens+n_act)` sample covariance**
  `Ĉ_Z^[l](k)` of `Z=[Y;U]` — computed from period-to-period scatter (the existing
  period-variance machinery; **not** floored).
- **Closed loop is valid because the reference is available** (the injected multisine is
  the reference): periodic averaging puts the noise-free reference-driven response in the
  sample **mean** and the noise in the sample **covariance**, so the EIV fit is unbiased in
  closed loop (P&S Remark (iv) p.474, indirect method §7.2.7).
- **`dof ≥ n_sens + 8`** periods/blocks are required for a trustworthy CRB (P&S Thm 12.7);
  6-DoF ⇒ `dof ≥ 14`. The campaign must collect enough periods; the fit asserts it and the
  CRB applies the finite-sample inflation either way.

## 4. The estimator — SML equation-error fit (wiki §3–§5; P&S §12.3)

**Cost (12-15):** `V_SML(θ) = Σ_{l} Σ_{k∈𝕂} eᴴ(Ω_k,θ) · Ĉ_ε^[l](Ω_k,θ)⁻¹ · e(Ω_k,θ)`

**Equation error (12-16), `n_sens × 1`:** `e = Ŷ^[l](k) − G(Ω_k,θ)·Û^[l](k) =
[I_{n_sens}, −G(Ω_k,θ)]·Ẑ^[l](k)` — fit to the **measured** spectra (no matrix inverse).

**Residual covariance (12-17):** `Ĉ_ε^[l] = [I,−G]·Ĉ_Z^[l](k)·[I,−G]ᴴ` — the input/output
sample covariance (with cross-covariance) projected through `[I,−G]`; `θ`-dependent through
`G`, hence iterative. This EIV weighting is what makes the cost valid in closed loop and
**handles the resonances without any inverse blow-up** — the regularization mechanism.

**Starting values (12-31/32/33):** multiply (12-16) by `A` → an error **linear** in the
coefficients, `A·Ŷ − B·Û ≈ 0`; stack `J_LS·θ ≈ 0`; solve by **iterative weighted linear LS
(Sanathanan–Koerken)** under `‖θ‖₂=1`. Generalize the existing
`estimators/gml.py:_sanathanan_koerner` to the common-denominator MIMO case.

**Iteration (12-24/25/27):** Gauss–Newton on the **pseudo-Jacobian** (12-25),
`J₊ = (Ĉ_ε)^{-1/2}(∂e/∂θ_r − ½(∂Ĉ_ε/∂θ_r)Ĉ_ε⁻¹ e)` (keep the `½` weighting-derivative
term); update `J₊re·Δθ = −ε_re` solved by **SVD** on the real-stacked system; **over-
parameterize then constrain** (all coeffs free → pseudo-inverse for the rank-deficient
solve → impose `‖θ‖₂=1` on `θ+Δθ`); **normalize angular frequencies by their median** and
**scale Jacobian columns by their 2-norm** for conditioning.

## 5. Uncertainty & CRB (wiki §6; P&S §12.3.4, §11.2, §16.12)

- **Parameter covariance / CRB (12-29, 12-30):** `Cov(θ̂) ≈ (2 Re(JᴴJ))⁻¹` (inverse Fisher),
  with the SML finite-sample inflation `λ(dof)`; the same analytic `∂G/∂θ` Jacobian from §4
  builds it (one Jacobian, two consumers).
- **Per-mode `f0`/`Q` (P&S §11.2.3):** propagate `C_θ` to the **roots of `A(Ω,θ)`** by
  linearization; map roots → `f0 = |λ|/2π`, `Q = |λ|/(−2 Re λ)`. **Caveat (P&S):** the
  linearized pole ellipses can under-cover at high SNR / sharp resonances — flag it, and use
  the **App 11.D improved bounds** when the linearization is inadequate.
- **Per-bin FRF band (11-2):** `var(G(Ω,θ̂)) ≈ (∂G/∂θ)·C_θ·(∂G/∂θ)ᴴ` → an uncertainty band
  on each `G_ij(f)`.

## 6. Validation (wiki §7; P&S §12.3.6)

Mirror P&S's own model-validation suite (tolerances from real runs, never loosened to pass):

- **Recovers truth:** on the twin, the fitted `f0`/`Q` match the known modal poles within
  the CRB; the fitted `G(Ω,θ̂)` matches the analytic oracle across the band **including the
  resonances** (the headline result step 1 could not deliver off-resonance-only).
- **Beats the nonparametric inverse at resonances:** quantitatively, `|G(θ̂) − G_oracle|` at
  the resonance bins is far below step-1's per-bin `Y·X⁻¹` error there.
- **Per-entry FRF comparison (12-34):** `|G_{ij}(θ̂) − Ĝ_{ij}(Ω_k)| ≤ √(F_p(2,2dof))·σ̂_Ĝ`
  holds for ~`p%` of bins; multivariate form (12-35).
- **Cost sanity:** the SML cost minimum ≈ its expected value `dof/(dof−n_sens)·(…)`
  (P&S 12-19) — a too-large minimum flags under-modeling.
- **Whiteness** of the FRM residuals `Ĝ(Ω_k) − G(Ω_k,θ̂)`.
- **Honest uncertainty:** sample covariances genuinely estimated (period-to-period, not
  floored); `dof` sufficiency asserted; the negative-residual-variance caveat (P&S 11-5)
  documented for sharp resonances.
- **Scale:** proven at **2-DoF**, run at **6-DoF** (L/P/Y/R/V/T) — a marked/slower test, the
  same convention as step 1.

## 7. Components (each independently testable)

1. **`MIMOCommonDenomModel`** — `G(Ω,θ)=B/A` (scalar `A`, polynomial-matrix `B`);
   `eval(freq) → (n_sens, n_act, F)` complex tensor (same shape as the step-1 oracle);
   analytic `∂G/∂θ` (feeds the Jacobian **and** the CRB); order `(n_a,n_b)` and the
   `‖θ‖=1` constraint. Continuous-time (`Ω=s`).
2. **`starting_values`** — SK/Levy linear-LS init (12-31/32/33), generalizing
   `_sanathanan_koerner`.
3. **`MIMOSampleMLEstimator`** — assembles `Ẑ^[l]`, `Ĉ_Z^[l]` from the per-actuator
   campaigns; Gauss–Newton SML fit (12-15/16/17, 12-24/25/27); returns `θ̂`, the fitted
   model, the cost, and the per-`(l,k)` residuals.
4. **`parameter_covariance` / CRB** — `(2 Re(JᴴJ))⁻¹` + SML inflation (12-29/30); reuse the
   §3 Jacobian. Generalize/extend `fisher.py` patterns; **leave the SISO `fisher.py`
   functions intact** (new code path, same as step-1's boundary discipline).
5. **`modal_uncertainty`** — roots of `A` → `f0`/`Q` with propagated covariance (§11.2.3 /
   App 11.D); FRF band (11-2).
6. **`validation`** — the §6 suite.

**Boundaries:** new module(s) (`mimo_fit.py` + a `MIMOCommonDenomModel`), consuming the
step-1 `MIMOTwinBackend`/campaign outputs. The SISO `model.py:TFModel`,
`estimators/gml.py:GMLEstimator`, and `fisher.py` are reused/extended, **not disturbed**.
Backend-agnostic: the fit consumes `(Ẑ, Ĉ_Z)` from any `ChannelBackend` campaign, so step 3
(RTSfreerun) feeds it unchanged.

## 8. Dependencies

- **numpy / scipy** — already present (SVD, polynomial roots, least squares).
- **python-control** — present (step-1 plant/loop construction, used by the twin that
  generates the data and the oracle to score against).
- **No new third-party dependency.** The estimator is numpy/scipy (the §4 Gauss–Newton +
  SVD + the SK init). slycot stays a step-1 dependency (loop construction), not needed by
  the fit itself.

## 9. Out of scope (this spec)

- The **RTSfreerun demonstration** on `x1hsts6dof` — step 3, its own follow-on; this spec
  only keeps the fit backend-agnostic so it transfers.
- The **discrete-time (`Ω=z⁻¹`) variant** and tustin pre-warping — v1 fits continuous-time
  at the physical bin frequencies; warping is negligible in-band (modes ≲2 Hz ≪ fs) and is
  documented, revisited only if a 6-DoF residual shows it (P&S supports both domains).
- **Simultaneous uncorrelated multisines** across actuators (the "fast"/single-experiment
  method, `n_exp=1`) — v1 uses the robust per-actuator method; deferred.
- The **damping-paradigm study** (Euler vs eigenmode optimum) — the twin supports the knob;
  running the study is later work.
- **Real hardware / CDS** (pyepics/pyawg/cdsutils) — Phase 2 ([[two-phase-cds-plan]]).

**Superseded pre-book assumptions** (recorded so the plan does not reintroduce them):
- ❌ "recover `G = Y·X⁻¹` then fit the model to `G(f)` weighted by a delta-method
  covariance" → ✅ fit the **raw `X_mat`/`Y_mat` spectra** via the EIV equation-error SML
  (12-16/17); step-1's `G(f)` is a **validation overlay + starting-value aid** only. (The
  fit-Ĝ route is P&S's sanctioned *fallback*, Remark (iv), not the primary path.)
- ❌ modal / vector-fitting pole-residue parameterization, rank-1 residues → ✅
  **common-denominator `B/A`, free per-element numerators**.
- ❌ variable projection → ✅ **full Gauss–Newton** over all coefficients, over-parameterize
  then constrain.
- ❌ "down-weight the ill-conditioned `G(f)` at resonances" → ✅ resonances handled
  intrinsically by the **equation-error weighting** (no inverse).

## 10. Hard rules honored

- One P&S pipeline; reuse the periodic multisine + sample-mean/covariance machinery; **no
  new estimation method** — the fit is P&S's literal SML ([[stay-on-pintelon-schoukens]]).
- `conda run -n sysid` for all execution ([[use-conda-run-sysid-env]]).
- Any plot SVG + Git LFS; data-driven y-limits ([[graphics-svg-lfs-only]]).
- Trunk-based, push to main ([[trunk-based-push-to-main]]); don't silently reverse user
  changes ([[never-silently-reverse-user-commands]]); this spec is book-grounded, not
  guessed ([[dont-guess-ask]]). Phase 1 only ([[two-phase-cds-plan]]).
