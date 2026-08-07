---
name: compute-the-bound-before-claiming-a-limit
description: "HARD RULE — compute the CRB/SNR/headroom/resolution bound (with numbers) BEFORE calling any result a fundamental limit, documenting a limitation, or asking how to proceed"
metadata: 
  node_type: memory
  type: feedback
---

Recurring failure the user has caught ~10×: when an identification/measurement result is
worse than hoped, I conclude "fundamental/physical limit" and pivot to *documenting the
limitation* or asking *"how do you want to proceed?"* — when a back-of-envelope
(college-level) calculation would have shown the limit isn't real and named the knob to fix
it. The user (a LIGO expert) finds wading through math-refutable limit-claims a waste of his
paid time; a confident wrong "it's a limit" is worse than no answer — it steers the work
wrong.

**HARD RULE — before (a) calling a result a fundamental/physical limit, (b) pivoting to
"document the limitation," or (c) asking the user how to proceed on a stuck/underwhelming
result, FIRST compute the relevant bound with real numbers and post it.** If the data carry
the information / the headroom exists, the failure is **my implementation** (init,
conditioning, model order, parameterization, under-driving, a sign error) — not a limit.

The bound to compute (pick what applies):
- **Estimation/recovery "limit"** → the **CRB / Fisher information** for the quantity. CRB ≪
  the observed error ⇒ implementation bug, not a limit. (This project: common-denominator
  pole "non-identifiability" = parameterization+conditioning; 6-DoF "Q-limit" = undermodeling
  at n_modes=8; 0.67 Hz "unresolvable doublet" = fitting 13 modes, collapsing the pair by
  construction.)
- **"Can't resolve X"** → distinguish the **non-parametric** limit (Rayleigh / peak-pick / 1
  DFT bin / 1 linewidth) from the **parametric ML** limit. ML super-resolves: two modes Δf
  apart of linewidth Γ=f0/Q are resolvable once **SNR·N ≳ (Γ/Δf)⁴** — finite, SNR-beatable.
  NEVER apply the non-parametric limit to a parametric fit.
- **"Noise-limited"** → compute the **SNR AND the actuator/dynamic-range HEADROOM**. Am I
  using the available drive? (I drove 0.05 counts against a 30000-count coil limit and called
  it seismic-limited.) You bury a fixed noise floor under drive; you don't tune the drive
  *down* to "fight" it. Fisher info ∝ SNR ∝ drive².
- **"Resolution-limited"** → df vs feature width / mode spacing. Longer record (T=1/df) and
  **Fisher-optimal line placement** (cluster excited lines at the informative frequencies)
  are KNOBS, not constraints.
- **Conditioning** → check the condition number / overflow (e.g. high-order Vandermonde) and
  fix it (normalization, reparameterization) before blaming the method.

The measurement-design knobs — **drive amplitude (to the actuator limit), df / record length,
excited-line placement, model order, # periods** — are things I CONTROL and set from the math
(Fisher-optimal), not fixed limits. Only present "document the limit / give up" AFTER the
bound says it is real, and SHOW the calculation + the cost to beat it. Full pre-conclusion
checklist: `.llm/engineering-practices.md`. Ties to [[dont-guess-ask]],
[[stay-on-pintelon-schoukens]].
