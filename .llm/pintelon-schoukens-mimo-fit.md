# P&S literal procedure — joint MIMO parametric fit (step 2 reference)

**Source:** Pintelon & Schoukens, *System Identification: A Frequency Domain Approach*,
2nd ed. (`docs/SysID-Pintelon.pdf`). Scraped 2026-06-22, reading the equation pages
visually (text extraction garbles all math). Citations are **printed page (PDF index =
printed + 35)** and equation numbers.

**Why this file exists:** step 2 (joint MIMO fit + CRB) must implement P&S's *literal*
method, not an equivalent reparameterization. I had been proposing modal/vector-fitting
forms + variable projection + a "recover G then fit G" two-stage; reading the book shows
the actual P&S procedure is different and simpler. This is the authoritative reference
for the step-2 spec. See [[stay-on-pintelon-schoukens]] memory.

---

## 0. The one-paragraph summary

P&S fit a **common-denominator MIMO transfer-function model** `G = B/A` (scalar shared
denominator `A`, full-rank polynomial-matrix numerator `B`) **directly to the measured
input/output DFT spectra** of a periodic multisine, by minimizing a **sample-ML (SML)
equation-error cost** weighted by the input/output sample covariance. There is **no
intermediate "invert to get G(f)"** step. Closed loop is handled automatically because the
*measured* plant input is used with its noise covariance (errors-in-variables), provided
the **reference is available** (it is). Starting values come from a **linear LS** on the
denominator-multiplied equation error (Sanathanan–Koerken / Levy). The iteration is
**Gauss–Newton on a pseudo-Jacobian**. Parameter covariance / CRB is `(2 Re(JᴴJ))⁻¹`;
`f0/Q` and FRF uncertainties are **propagated** from it.

---

## 1. The model — common-denominator MIMO (§6.6, pp. 193–194)

For an `n_y × n_u` system, when the transfer functions **share the same denominator**
(modal analysis, multi-port mechanical/electrical systems — **our case**, shared modal
poles), use the **common-denominator model (6-53):**

```
G(Ω,θ) = B(Ω,θ) / A(Ω,θ) = ( Σ_r B_r Ω^r ) / ( Σ_r a_r Ω^r )
```

- `A(Ω,θ)` = **scalar** common-denominator polynomial, coeffs `a_r ∈ ℝ` (the shared poles).
- `B_r ∈ ℝ^{n_y × n_u}` = polynomial-matrix numerator coefficients (one numerator
  polynomial per I/O element — **free per element**).
- `Ω = s` (continuous-time lumped), `z⁻¹` (discrete), `√s` (diffusion), `tanh(τ_R s)`.
- Param vector (starting-value form, 12-33):
  `θ = [vecᵀ(A_0) … vecᵀ(A_{n_a})  vecᵀ(B_0) … vecᵀ(B_{n_b})]ᵀ`, with `A_r` scalar
  (`n_p = 1`) for the common-denominator model.

**Residue-rank settles "free per-element vs rank-1" — per P&S, Remark (iii) p. 194 + App
6.N:** the partial-fraction/matrix-fraction/state-space residue matrices are **rank-1**
(`L_r = v_r w_rᵀ`, modal vectors) — but **only for proportionally-damped reciprocal
structures**. The **common-denominator model has full-rank residues**. P&S's rule
verbatim: *"the common denominator model should be used in all applications where the rank
of the residue matrices"* is not known to be one. Our coupled plant has deliberate
cross-coupling + frequency-dependent `M_out` ⇒ rank-1 is **not** guaranteed ⇒
**common-denominator (full-rank residue) is the P&S-correct choice.** (Rank-1 matrix
fraction / state space is the fewer-parameter option *only if* rank-1 is known a priori —
not us.)

**Do NOT lead with the modal partial-fraction form (6-22)/(6-23).** P&S note (p. 183) it
has *worse* starting values and "gets stuck in local minima"; the rational `B/A` (6-20/53)
is the default precisely because it gives good starting values.

**Identifiability constraint** (remove scale ambiguity of `B/A`): common-denominator →
one-dimensional constraint `‖θ‖₂ = 1` suffices (or monic `a_{n_a}=1`). (Left matrix
fraction would need `A_0 = I_{n_u}`, an `n_u²` constraint — not our model.)

**Time delay** can be appended: `G = e^{−τs} B/A` (6-29); `θ` then includes `τ`.

---

## 2. The data — periodic multisine, sample means + covariances (§12.3.1, p. 470)

- Excite with a **periodic multisine**; observe `M` steady-state periods; DFT each period.
- `𝕂` = the set of `F = O(N)` **excited** DFT lines in the band of interest.
- **Sample mean** spectra over periods: input `Û^[l](k)`, output `Ŷ^[l](k)`; and the
  **stacked sample covariance** `Ĉ_Z^[l](k)` of `Z = [Y; U]`, an `(n_y+n_u)×(n_y+n_u)`
  Hermitian matrix per bin (the period-to-period scatter — what `system_ident` already
  computes as period-variance).
- **Experiments `l = 1…n_exp`.** The **"robust" method uses `n_exp = n_u`** —
  **drive each input separately** with the multisine (uncorrelated experiments). *This is
  exactly our sequential per-actuator campaign.* ("arb"/"fast" methods use `n_exp = 1`
  with one rich excitation; not our v1.)

**Noise model (10-9):** `C_Nz(k) = E{N_Z N_Zᴴ}` is the input/output noise covariance;
for SISO it is `[[σ_Y², σ_YU²],[σ̄_YU², σ_U²]]` — note the **cross-covariance `σ_YU²`**.

**Closed loop (Remark (iv) p. 474, §10.10):** *"The asymptotic properties of the SML
estimator remain valid for noisy input-output observations of systems operating in closed
loop, provided the reference signal `r(t)` is available."* Mechanism: with periodic
excitation the **sample mean over periods is the noise-free reference-driven response**
(noise averages out) and the **sample covariance is the noise** — so the EIV fit on
(mean, covariance) is unbiased in closed loop. Use the **indirect method §7.2.7** (project
on the reference) to get unbiased sample means/covariances. We have the reference, so this
is satisfied. (Only the robust method of §7.3.3 avoids needing `r(t)`.)

---

## 3. The estimator — SML equation-error cost (§12.3, p. 471; SISO §10.3.1, p. 387)

**MIMO SML cost (12-15):**

```
V_SML(θ,Z) = Σ_{l=1}^{n_exp} Σ_{k∈𝕂}  eᴴ(Ω_k,θ,Ẑ^[l](k)) · ( Ĉ_ε^[l](Ω_k,θ) )⁻¹ · e(Ω_k,θ,Ẑ^[l](k))
```

**Equation error (12-16)** — `n_y × 1` per experiment, fit to the **measured** spectra:

```
e(Ω_k,θ,Ẑ^[l](k)) = Ŷ^[l](k) − G(Ω_k,θ)·Û^[l](k) = [ I_{n_y} , −G(Ω_k,θ) ] · Ẑ^[l](k)
```

**Residual covariance (12-17)** — project the input/output covariance through `[I,−G]`:

```
Ĉ_ε^[l](Ω_k,θ) = [ I_{n_y} , −G(Ω_k,θ) ] · Ĉ_Z^[l](k) · [ I_{n_y} , −G(Ω_k,θ) ]ᴴ
```

This `Ĉ_ε` is **θ-dependent** (through `G`) — hence the iteration. It is the EIV weighting:
it carries the noise on **both** `Y` and `U` *and* their cross-covariance, which is what
makes the cost valid in closed loop.

**SISO intuition (10-10/10-11), equivalent via `G=B/A`:** multiply the residual by `A`:
```
ê = A(Ω_k,θ)·Ŷ(k) − B(Ω_k,θ)·Û(k)
σ̂²_e = σ̂_Y²|A|² + σ̂_U²|B|² − 2 Re( σ̂_YU² · A · B̄ )      ;   σ̂²_ê = σ̂²_e / M
V_SML = Σ_k |ê|² / σ̂²_ê        (= Σ_k |ε|²,  ε = ê/σ̂_ê normalized equation error)
```

**Finite-sample (sample-covariance) facts — `dof` = degrees of freedom of the sample
covariance (≈ number of periods/blocks):**
- Need **`dof ≥ n_y + 8`** for asymptotic normality + parameter uncertainty (Thm 12.7);
  `≥ n_y+2` to converge, `≥ n_y+7` for the rate. (SISO single-output: `dof ≥ 9`; the
  classic SISO rule is M ≥ 7 periods, §10.3.) Regularizing `Ĉ_ε⁻¹` relaxes this to
  `dof ≥ n_y+2` (Remark (iii) p. 474).
- **Cost inflation (12-19):** `V_SML/F = dof/(dof−n_y) · V_ML/F` (SISO 10-12: `(M−1)/(M−2)`).
- **Covariance inflation (12-23/12-30):** `Cov_SML = λ·Cov_ML` with
  `λ₁(dof) = dof(dof−n_y)/[(dof−n_y+1)(dof−n_y−1)]` (SISO 10-16: `(M−2)/(M−3)`).

**Remark (iv) p. 472 — the FRM alternative (my earlier "fit recovered G" idea):** P&S
*do* sanction fitting the **nonparametric FRM** `Ĝ(Ω_k)` directly — set `Ŷ=vec(Ĝ)`,
`Û=1`, weight by `Ĉ_vecĜ(Ω_k)`, replace `G→vec(G(θ))`. **But this needs forming
`Ĝ = Y·X⁻¹` and its covariance** (ill-conditioned at resonances). The **primary route
(12-16/17) avoids the inverse entirely** — that is what we implement; the FRM fit is a
fallback only.

---

## 4. Computation — Gauss–Newton on the pseudo-Jacobian (§12.3.3, p. 474)

Analytic differentiation of `(Ĉ_ε)^{-1/2}` is impractical, so P&S use a **pseudo-Jacobian**:

**(12-25)** column `r` of the per-`(l,k)` Jacobian block:
```
J₊^{[l,k]}_{[:,r]} = (Ĉ_ε^[l])^{-1/2} · ( ∂e/∂θ_r  −  ½ (∂Ĉ_ε^[l]/∂θ_r)(Ĉ_ε^[l])⁻¹ e )
```
The `½(∂Ĉ_ε/∂θ)…` term accounts for the **θ-dependence of the weighting** — do not drop it.

**(12-24)** stack `J₊` over experiments `l` and bins `k`. **(12-28)** `ε^{[l,k]} =
(Ĉ_ε^[l])^{-1/2} e`. **Update (12-27):** solve `J₊re · Δθ = −ε_re` by **SVD**, where
`(·)_re` stacks real & imaginary parts (so the problem is real).

**Practical numerics:**
- **Over-parameterize then constrain:** keep *all* coefficients free during the iteration
  (avoids switching parameterizations mid-solve). The pseudo-Jacobian is then rank
  deficient → solve via **pseudo-inverse** (the number of zero singular values is known a
  priori). After the update, impose the constraint on `θ+Δθ` (`‖θ‖₂=1` / monic for
  common-denominator).
- **Normalize angular frequencies by their median** (CT models, `Ω=s,√s`) for conditioning.
- **Scale each Jacobian column by its 2-norm** before the SVD solve.

---

## 5. Starting values — linear LS / SK / Levy (§12.3.5, p. 476)

Multiply the equation error (12-16) by the **denominator** to get an error **linear in the
coefficients (12-31):**
```
A(Ω_k,θ)·Ŷ^[l](k) − B(Ω_k,θ)·Û^[l](k) ≈ 0
```
Stack into **(12-32)** `J_LS(Z)·θ ≈ 0` (`J_LS` is `n_y·n_exp·F × n_θ`), solve for `θ`
(12-33) under `‖θ‖₂=1`. Because (12-32) matches the scalar case, build the init with
**iterative weighted linear LS (Sanathanan–Koerken)** / generalized TLS / bootstrapped TLS
— the repo already has `_sanathanan_koerner` in `estimators/gml.py` to generalize to the
common-denominator MIMO case.

---

## 6. Uncertainty & CRB (§12.3.4 p. 475; §11.2 p. 433; §16.12 p. 588)

**Parameter covariance (12-29)** — the practical CRB (inverse Fisher, Gauss–Newton approx):
```
Cov(θ̂_ML) ≈ ( 2 Re( J_ML+ᴴ J_ML+ ) )⁻¹
```
evaluated at `θ̂`. **SML form (12-30):** `Cov(θ̂_SML) ≈ 0.5·λ₂(dof)·(V_J Σ_J⁺)(V_J Σ_J⁺)ᵀ`
from the pseudo-Jacobian SVD `U_J Σ_J V_Jᵀ`, with the finite-sample inflation
`λ₂(dof) = dof² / [(dof−n_y+1)(dof−n_y−1)]`. The **CRB** (§16.12) is `Fi(θ_0)⁻¹` with the
Fisher matrix `Fi = 2 Re(JᴴJ)`; ML reaches it asymptotically.

**FRF uncertainty band (11-2):** propagate `C_θ = Cov(θ̂)` through the model Jacobian:
```
var(G(Ω,θ̂)) ≈ (∂G/∂θ|_θ̂) · C_θ · (∂G/∂θ|_θ̂)ᴴ
```
→ a per-bin uncertainty band on each `G_ij(f)`.

**Pole / `f0`,`Q` uncertainty (§11.2.3, p. 435):** propagate `C_θ` (covariance on the
polynomial coefficients) to the **roots of `A(Ω,θ)`** by linearization. **Caveat (P&S):**
for high SNR / sharp resonances the linearized pole/zero ellipses "may not cover the true
uncertainty regions" — use the **improved bounds of Appendix 11.D** (p. 455) when the
linearization is inadequate. (The roots → `f0/Q` map via `f0 = |λ|/2π`,
`Q = |λ|/(−2 Re λ)` for a pole `λ`.)

---

## 7. Model selection & validation (§12.3.6, p. 477)

Three tests (mirror these in step-2 testing):
1. **Per-entry FRF comparison (12-34):** `|G_{[r,s]}(θ̂) − Ĝ_{[r,s]}(Ω_k)| ≤
   √(F_p(2,2dof))·σ̂_Ĝ` should hold for ~`p%` of frequencies (Ĝ = nonparametric FRM with
   its variance). Multivariate version **(12-35):**
   `vecᴴ(G_θ−Ĝ)·Ĉ_vecĜ⁻¹·vec(G_θ−Ĝ) ≤ (n_1 dof/n_2)·F_p(n_1,n_2)`,
   `n_1=2 n_y n_u`, `n_2 = 2 dof − 2 n_y n_u + 2`.
2. **SML cost vs expectation:** at the true model `E{V_SML} ≈ dof/(dof−n_y)·F·(n_y…)` (the
   cost is ~χ²); a too-large minimum ⇒ modeling errors / undermodeling.
3. **Whiteness of the FRM residuals** `Ĝ(Ω_k) − G(Ω_k,θ̂)`.
Overmodeling: penalize model complexity (add penalty to the cost), Chapter 11 tools.

**Validation caveat (11-5, p. 434):** the residual-variance compensation
`var(G_meas − G_model) = σ²_G(k) − σ²_G(θ̂)` can go **negative** at sharp resonances where
the model is flexible and rides only 1–2 data points — there, residual significance can't
be judged. Relevant exactly at our suspension resonances.

---

## 8. Mapping onto `system_ident` step-2 (what to build)

| P&S | our twin |
|---|---|
| `U^[l]` (measured plant input) | **drive monitors** `X_mat` column (`n_act = n_u`) |
| `Y^[l]` (output) | **sensor response** `Y_mat` column (`n_sens = n_y`) |
| robust method, `n_exp = n_u` | **sequential per-actuator** multisine campaign (step-1 already does this) |
| sample mean + `Ĉ_Z` over `M` periods | reuse the existing **period-variance** machinery |
| reference available ⇒ closed-loop OK | the injected multisine **is** the reference |
| common-denominator `G=B/A`, full-rank `B` | shared scalar denom (modal poles) + free per-element numerators |

**The build (step-2 components):**
1. `MIMOCommonDenomModel` — `G(Ω,θ)=B/A`, scalar `A` (shared poles), poly-matrix `B`;
   `eval(freq) → (n_sens,n_act,F)`; analytic `∂G/∂θ` (feeds Jacobian **and** CRB).
2. SK/Levy **starting values** (12-31/32/33) — generalize `_sanathanan_koerner`.
3. **SML fit** (12-15/16/17): Gauss–Newton pseudo-Jacobian (12-24/25), SVD update (12-27),
   over-parameterize + `‖θ‖=1` constraint, freq-median normalize, column scaling.
4. **Covariance/CRB** (12-29/30) + **FRF band** (11-2) + **`f0/Q` propagation** to roots of
   `A` (App 11.D when needed).
5. **Validation** (12-34/35 + whiteness + cost-vs-expected).

**Corrections to my earlier (pre-book) design assumptions — the spec must use these:**
- The fit target is **the raw `X_mat`/`Y_mat` spectra + sample covariances**, *not* the
  recovered `G = Y·X⁻¹`. The step-1 matrix-inverse `G(f)` becomes a **nonparametric
  overlay for validation (12-34) and a starting-value aid**, not the data the fit consumes.
  (P&S Remark (iv) keeps "fit Ĝ + Ĉ_vecĜ" as a sanctioned *fallback*.)
- The model is **common-denominator `B/A`** (free per-element numerators, shared scalar
  denominator) — **not** a modal pole-residue / vector-fitting parameterization, and
  **not** rank-1 residues.
- The solver is **full Gauss–Newton over all coefficients** with the over-parameterize-
  then-constrain trick — **not** variable projection.
- **Resonances are handled by the EIV equation-error weighting (no inverse), not by
  down-weighting an ill-conditioned `G(f)`.** This is the real reason the parametric fit
  beats step-1's per-bin inverse at the peaks.
- `dof ≥ n_y + 8` sets the **minimum period/block count** for trustworthy CRB
  (6-DoF ⇒ `dof ≥ 14`).
