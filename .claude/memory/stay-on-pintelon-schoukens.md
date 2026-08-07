---
name: stay-on-pintelon-schoukens
description: system_ident is a single Pintelon–Schoukens pipeline; do not propose non-P&S escapes when stuck
metadata: 
  node_type: memory
  type: feedback
---

The `system_ident` repo is deliberately a **single Pintelon–Schoukens pipeline**
(periodic multisine → leakage-free reference FRF → ML fit → CRB-driven optimal
excitation). When identification hits a wall, do **not** reach for non-P&S escapes —
flat excitation, deferring to a vague "Track B", or reframing the rung as "the method
needs something else." The user pushed back hard ("you are AGAIN trying to diverge from
P&S — what the fuck") when I offered options built around a self-inflicted problem.

**Why:** in the HSTS closed-loop (A3) work the "resonance peaks recover 50–110% off,
every method fails identically" wall was a **sign bug** in the plant-input
reconstruction (`COIL_DRV_SUM` is a `"+-"` junction → `drive − feedback`, I used `+`),
not a limitation of P&S. I'd already drifted to flat excitation in probes, then blamed
the method for the resulting ill-conditioning.

**How to apply:** when every approach fails the *same way*, suspect a shared
bug/reconstruction/sign error before concluding the method is inadequate. Keep
excitation P&S-optimal, not flat (see the roadmap's "do NOT regress to flat" note). If
a genuinely-new estimator (e.g. the MIMO joint fit) seems needed, prove the existing
P&S path is exhausted first. Related: [[ask-for-missing-tools-and-pause]],
[[rtsfreerun-env-strategy]].

**Implement P&S's LITERAL method first — don't propose equivalent-but-different
reparameterizations.** When designing the joint MIMO fit (step 2) I kept offering
modal/vector-fitting pole-residue forms and variable-projection solvers as "equivalent"
alternatives to P&S's common-denominator matrix-fraction model + Gauss–Newton ML. The
user pushed back: "i dont understand why you keep trying to be different from P&S. how
about we just try their method first, and if it fails or is slow, we can modify." So:
build the literal P&S procedure (common-denom matrix fraction, their SML cost, SK/Levy
init, poles as roots of A(s), CRB via their covariance-to-roots propagation). Modify
ONLY if it demonstrably fails or is too slow — and only then. If unsure what the exact
P&S procedure is, **ask for the book / source rather than reconstructing from memory and
guessing at the edges** (the user offered to provide the PDF to scrape). Ties to
[[dont-guess-ask]].
