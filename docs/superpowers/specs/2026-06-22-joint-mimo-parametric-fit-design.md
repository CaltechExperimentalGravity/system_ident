# Joint MIMO parametric fit — rank-1 modal SML + Cramér–Rao bound

**Status:** approved design (v2 — rank-1 modal), pre-implementation
**Date:** 2026-06-22
**Arc:** Step 2 of "joint MIMO identification of closed-loop systems" — the headline
SEI/ISC commissioning need (see `.llm/roadmap.md`).
- **Step 1 (done, on `main`):** generic coupled+closed-loop twin + nonparametric per-bin
  matrix-inverse recovery `G(f) = Y_mat·X_mat⁻¹`.
- **Step 2 (this spec):** the joint *parametric* fit — one **rank-1 modal** MIMO model with
  **shared modal poles**, fit by Pintelon–Schoukens **sample-ML (SML)**, with the
  **Cramér–Rao bound** on the modal parameters and the FRF.
- **Step 3 (named follow-on):** demonstrate the same fit on the RTSfreerun compiled twin
  (`x1hsts6dof`). The fit is backend-agnostic so step 3 is a demonstration, not a rewrite.

Phase 1 (RTSfreerun) only — real hardware is Phase 2, out of scope ([[two-phase-cds-plan]]).

## 0. Why rank-1 modal (the design evolution — read this first)

This spec **started** as P&S's literal common-denominator model `G=B/A` (free per-element
numerators), per [[stay-on-pintelon-schoukens]] ("use their method first; modify if it
fails or is slow"). **It failed — and the failure was measured, not guessed:**

- On the **real `mimo_suspension` plant** at 6-DoF, the common-denominator SML **does not
  identify the poles.** Even starting *exactly* at the right poles (or 10% off), the fit
  drifts to wrong poles; `cost@fit < cost@true`.
- **Root cause:** 36 elements × degree-11 free numerators = **432 numerator coefficients**
  vs the shared **12-coefficient** denominator — a **33:1 ratio**. The numerators absorb any
  pole error, so the poles are *weakly identifiable*. (At 2-DoF the ratio is 3:1 → it works;
  it's the MIMO size that breaks it.) Confirmed noise-independent (identical at 0.3% and 1%
  noise) ⇒ structural, not statistical.

**The fix is P&S's own Remark (iii) (§6.6, p.194):** "if the residue matrices have rank one
(e.g. in modal analysis), the … state-space / matrix-fraction descriptions are preferred
[fewer parameters]." A reciprocal, proportionally-damped suspension has **rank-1 residues per
mode** (mode shape ⊗ mode shape). The twin is **exactly rank-1 by construction**
(`mimo_plant.py:34`: `term = gain·φ[i,k]·ψ[j,k]`). So rank-1 is **zero-undermodeling here**,
physically correct, and **P&S-sanctioned** — not a divergence.

**Verified before writing this spec** (on the real plant, with the rank-1 model + the
data-driven peak-pick init of §4): poles recovered **exactly** (`maxerr 0.0000`) from priors
10%, 50%, and effectively arbitrary; ~1 s per fit (84 params, not 445). The full derivation +
the scraped P&S text live in `.llm/pintelon-schoukens-mimo-fit.md` and
`.llm/ps-book/` (full-text + marker).

## 1. Goal

Fit a **single rank-1 modal MIMO model** to the step-1 coupled-loop campaign — shared modal
poles `(a_k,b_k)` with per-mode **rank-1 residues** `R_k = φ_k ψ_kᵀ` (sensor shape `φ_k`,
actuator shape `ψ_k`) — by the P&S **sample-ML equation-error cost**, initialized from a
**data-driven peak-pick** so it is robust to arbitrary prior error. Report the **Cramér–Rao
bound**: parameter covariance `(2 Re(JᴴJ))⁻¹` with the SML finite-sample inflation, propagated
to **per-mode `f0`/`Q`**, the **mode shapes**, and a **per-bin FRF band** on each `G_ij(f)`.

The shared modal poles, constrained jointly by all elements through the rank-1 shapes, make
the model **predict through the resonances** where step-1's per-bin inverse is
ill-conditioned. It serves SUS/SEI/ASC/LSC: it turns the nonparametric coupled FRF into a
**modal model with calibrated uncertainty** — `f0`, `Q`, mode shapes, confidence bands —
which SISO-per-DoF tooling cannot give a commissioner.

## 2. The model — rank-1 modal (wiki §1; P&S §6.6 Remark (iii))

```
                M
G_ij(s)  =     Σ    φ_k,i · ψ_k,j  /  ( sₙ² + b_k·sₙ + a_k ) ,     sₙ = s / s_ref
              k=1   └── rank-1 ──┘     └──── shared pole ────┘
                     residue R_k          (mode k)
```

- **Shared poles** `(a_k, b_k)`, 2 per mode (`a_k = ωₖ²/s_ref²`, `b_k = (ωₖ/Qₖ)/s_ref`).
- **Per-mode shapes** `φ_k ∈ ℝ^{n_sens}`, `ψ_k ∈ ℝ^{n_act}` — the residue `R_k = φ_k ψ_kᵀ`
  is real rank-1.
- **Parameters:** `θ = {(a_k, b_k, φ_k, ψ_k)}`, total `M·(2 + n_sens + n_act)` (6-DoF: 84).
- **Frequency normalization (P&S §12.3.3) — required:** `s_ref = 2π·median(freq)`; coeffs in
  normalized `sₙ`; `poles()` un-normalizes roots. Without it the high-order evaluation
  overflows.
- **Gauge:** per-mode scale `φ_k→cφ_k, ψ_k→ψ_k/c` (M flat directions) — handled by the LM
  damping / pseudo-inverse in §5; CRB reports gauge-invariant quantities (`f0`, `Q`, `R_k`).

**Coupling is fully captured** — it lives in the *shapes* `φ_k`, `ψ_k` (non-trivial,
cross-coupled), not in the residue rank. Rank-1 ≠ uncoupled. **Dimensions** are generic
(`n_sens`, `n_act` independent — square for SUS/SEI, rectangular for ISC/LSC).
**Rank-r extension** (sum of r outer products per mode) is the documented path if a real
system's residues are not rank-1; v1 is rank-1 (exact for reciprocal suspensions).

## 3. The data — periodic multisine, sample means + covariances (wiki §2; P&S §12.3.1)

Reuses step-1's campaign verbatim — no new excitation or estimation method:

- **Robust method, `n_exp = n_act`:** drive each actuator separately with the periodic
  multisine (step-1 already does this). Per experiment `l`, over `M` periods at each excited
  bin: sample-mean input spectrum `Û^[l]` (drive monitors = `X_mat` column), sample-mean
  output `Ŷ^[l]` (sensors = `Y_mat` column), and the stacked `(n_sens+n_act)` sample
  covariance `Ĉ_Z^[l]` (period-to-period scatter; not floored).
- **Closed loop is valid** because the reference (the injected multisine) is available:
  periodic averaging puts the noise-free response in the mean, noise in the covariance
  (P&S Remark (iv) p.474).
- **`dof ≥ n_sens + 8`** periods for a trustworthy CRB (P&S Thm 12.7); 6-DoF ⇒ `dof ≥ 14`.

## 4. Initialization — data-driven peak-pick (the robustness key)

The init makes the fit robust to **arbitrary** prior error by reading the modes off the data,
not the prior:

1. **Peak-pick the resonances** from the **step-1 nonparametric recovered `G(f)=Y·X⁻¹`** —
   the *open-loop* plant FRF, where the resonances appear as full peaks (the closed-loop
   sensor spectrum has them suppressed by the damping loops, so peak-pick uses the recovered
   `G`, not raw `Y`). Sum `|G_ij(f)|²` over elements → `find_peaks` → the `M` strongest →
   initial `f0_k`; a default `Q` (e.g. 20) seeds `b_k`.
2. **Residues by linear LS:** with the poles fixed, `G_ij = Σ_k R_k,ij/D_k(sₙ)` is **linear**
   in the full residue matrices `R_k` — solve a weighted linear LS from `Ŷ = G·Û`.
3. **Rank-1 projection:** SVD each `R_k`; take the leading singular triple →
   `φ_k = √σ₁·u₁`, `ψ_k = √σ₁·v₁`.
4. **Prior as fallback only** — used to seed a mode that peak-pick cannot detect (too weak /
   overlapping), and to order/label modes. **Verified:** peak-pick alone recovers all 6 poles
   exactly regardless of a 10%/50% prior error. Multi-start was tried and is *not* reliable
   (lands in mode-collision basins); peak-pick is the chosen strategy.

## 5. The estimator — IQML / iteratively-reweighted SML (wiki §3; P&S §9.12.2, §12.3)

**Cost (SML, 12-15):** `V = Σ_l Σ_k eᴴ Ĉ_ε^[l](θ)⁻¹ e`, with **equation error (12-16)**
`e = Ŷ^[l] − G(θ)·Û^[l]` and **residual covariance (12-17)**
`Ĉ_ε = [I,−G]·Ĉ_Z·[I,−G]ᴴ`. Fit to the **raw `X_mat`/`Y_mat` spectra** — never `Y·X⁻¹`.

**Solver:** **Levenberg–Marquardt with cost-based step acceptance** on the analytic
Jacobian, **freezing the weighting `Ĉ_ε` within each iteration** (P&S's Iterative Quadratic
ML, §9.12.2) — this drops P&S's per-parameter `−½ ∂Ĉ_ε/∂θ` term (12-25), making each step
~10× faster (no per-parameter loop) and, with LM damping, robust at high order. Verified: ~1 s
per 6-DoF fit vs ~200 s for the full-Jacobian fixed-step GN, and it converges where the
fixed-step GN diverged. The frozen-weighting choice is a P&S-sanctioned approximate ML, taken
under the "modify if slow" rule; the full SML weighting can be reinstated as a final polishing
iteration if a residual demands it.

## 6. Uncertainty & CRB (wiki §6; P&S §12.3.4, §11.2, §16.12)

- **Parameter covariance / CRB (12-29/30):** `Cov(θ̂) ≈ (2 Re(JᴴJ))⁻¹` (pseudo-inverse; drops
  the per-mode gauge directions) × the SML inflation `λ₂(dof) = dof²/((dof−n_sens+1)
  (dof−n_sens−1))`. Same analytic Jacobian as §5.
- **Per-mode `f0`/`Q`** by propagating the pole-block covariance to the roots of
  `sₙ²+b_k sₙ+a_k` (§11.2.3; App 11.D bounds if the linearization is inadequate at high SNR).
- **Mode shapes** `φ_k`, `ψ_k` std (gauge-fixed, e.g. `‖φ_k‖=1`).
- **Per-bin FRF band (11-2):** `var(G) ≈ (∂G/∂θ)·C_θ·(∂G/∂θ)ᴴ`.

## 7. Validation (wiki §7; P&S §12.3.6)

Tolerances from real runs, never loosened to pass:

- **Recovers truth incl. resonances:** fitted `f0`/`Q` match the known modes within CRB;
  fitted `G(θ̂)` matches the analytic oracle across the band **including the peaks**.
- **Beats the nonparametric inverse at resonances:** `|G(θ̂) − G_oracle|` at resonance bins
  far below step-1's per-bin `Y·X⁻¹` error there.
- **Prior-robustness:** exact pole recovery from peak-pick under 10% **and** 50% prior error
  (the headline robustness result).
- **Off-resonance agreement** with the nonparametric inverse + **SML cost ≈ expected**
  (P&S 12-19); whiteness of the FRM residuals. (The rigorous per-bin 12-34 F-test needs the
  delta-method `Ĉ_vecĜ` — documented follow-on, not faked.)
- **Honest uncertainty:** sample covariances genuinely estimated (period-to-period, not
  floored); `dof` sufficiency asserted.
- **Scale:** proven at **2-DoF**, run at **6-DoF** (L/P/Y/R/V/T) — marked/slower test.

## 8. Components (each independently testable)

1. **`Rank1ModalModel(n_sens, n_act, n_modes)`** — `eval`, analytic `jacobian` (FD-verified),
   `poles → f0/Q`, `pack/unpack`, `set_reference`. (`src/system_ident/mimo_modal.py`.)
2. **`peak_pick_init` + `init_residues`** — peak-pick on recovered `G`; linear residue LS +
   rank-1 SVD; prior fallback. (`src/system_ident/mimo_fit.py`.)
3. **`MIMOModalEstimator`** — IQML-LM SML fit; returns `θ̂`, Jacobian, cost.
4. **`parameter_covariance` / `modal_uncertainty` / `frf_band`** — CRB + propagation.
5. **`mimo_campaign.assemble_campaign`** — `(Ẑ^[l], Ĉ_Z^[l])` from any `ChannelBackend`.
6. **`validate_fit`** — the §7 suite.

**Boundaries:** new modules; the SISO `model.py`/`estimators/gml.py`/`fisher.py` and the
common-denominator code are **not** used (the latter is abandoned — see §10). Backend-agnostic
so step 3 (RTSfreerun) feeds it unchanged.

## 9. Dependencies

numpy / scipy (SVD, roots, `find_peaks`, lstsq, eigh). python-control only via the step-1 twin
that generates data + the oracle. **No new third-party dependency.**

## 10. Out of scope / superseded

- **Common-denominator `B/A` free-numerator model** — **abandoned**: empirically unidentifiable
  at 6-DoF MIMO (§0). Replaced by rank-1 modal.
- **Multi-start init** — tried, unreliable (§4); peak-pick is the chosen strategy.
- **Rank-r residues** (non-reciprocal systems), the **rigorous 12-34 F-test** (needs
  `Ĉ_vecĜ`), the **RTSfreerun demo** (step 3), and **simultaneous multisines** — all
  documented follow-ons.
- **Real hardware / CDS** (pyepics/pyawg/cdsutils) — Phase 2 ([[two-phase-cds-plan]]).

## 11. Hard rules honored

- One P&S pipeline; reuse the periodic multisine + sample-mean/covariance machinery; the fit
  is P&S SML with the rank-1 modal model that P&S Remark (iii) prescribes for rank-1 residues
  — **modify-if-it-fails was earned by measurement, not preference** ([[stay-on-pintelon-schoukens]]).
- `conda run -n sysid` for all execution ([[use-conda-run-sysid-env]]).
- Any plot SVG + Git LFS; data-driven y-limits ([[graphics-svg-lfs-only]]).
- Trunk-based, push to main ([[trunk-based-push-to-main]]); don't silently reverse user
  changes ([[never-silently-reverse-user-commands]]); design is evidence-driven, not guessed
  ([[dont-guess-ask]]). Phase 1 only ([[two-phase-cds-plan]]).
