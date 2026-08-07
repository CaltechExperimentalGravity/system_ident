---
name: rank1-modal-mimo-fit
description: "Step-2 joint MIMO fit: common-denominator B/A fails at 6-DoF; use rank-1 modal + data-driven peak-pick init"
metadata: 
  node_type: memory
  type: project
---

For the step-2 joint MIMO parametric fit, the P&S **common-denominator `G=B/A`
free-numerator** model **does not identify the poles at 6-DoF MIMO** — measured on the real
`mimo_suspension` plant: even starting exactly at the true poles the fit drifts away;
`cost@fit < cost@true`. Root cause: 36 elements × degree-11 numerators = 432 numerator
coeffs vs the 12-coeff shared denominator (**33:1**), so the numerators absorb any pole
error → poles weakly identifiable. Noise-independent ⇒ structural, not statistical. (At 2-DoF
the ratio is 3:1 and it works fine — it's the MIMO size that breaks it.)

**Fix (verified, exact recovery):** the **rank-1 modal** model
`G_ij(s) = Σ_k φ_k,i ψ_k,j / (sₙ² + b_k sₙ + a_k)` — shared poles `(a_k,b_k)`, per-mode
rank-1 residues `R_k = φ_k ψ_kᵀ`. 84 params not 445. This is **P&S's own Remark (iii)**
(§6.6) for rank-1 residues, and the twin's residues **are** rank-1 by construction
(`mimo_plant.py:34`). Coupling is captured in the *shapes*, not the rank.

**Init that makes it robust:** **data-driven peak-pick** — find resonances in the step-1
nonparametric recovered `G(f)=Y·X⁻¹` (the open-loop plant; closed-loop sensor peaks are
suppressed by the damping loops), seed poles, residues by linear LS + rank-1 SVD. Recovers
all 6 poles **exactly** from 10%, 50%, and arbitrary prior error. **Multi-start does NOT
work** (lands in mode-collision basins) — peak-pick is the strategy.

**Solver:** IQML / iteratively-reweighted Levenberg–Marquardt (freeze `Ĉ_ε` per iteration,
drop P&S's −½∂Ĉ_ε/∂θ term, cost-based step acceptance) — ~1s vs ~200s for full-Jacobian
fixed-step GN, and robust where fixed-step GN diverged. P&S §9.12.2 sanctions it. Also
**frequency-normalize** `sₙ=s/s_ref`, `s_ref=2π·median(freq)` (P&S §12.3.3) — without it the
high-order Vandermonde overflows (loses a pole at 6-DoF).

This was earned by measurement under [[stay-on-pintelon-schoukens]]'s "modify if it fails"
rule — not a preference for diverging from P&S. Spec:
`docs/superpowers/specs/2026-06-22-joint-mimo-parametric-fit-design.md`. Method reference:
`.llm/pintelon-schoukens-mimo-fit.md`; chapter map in `.llm/ps-book/README.md` (the book itself
is copyrighted and is not in the repo — Ch 19–20 cover over-parameterized models and invariants).
