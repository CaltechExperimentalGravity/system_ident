---
name: no-flat-noise-drives
description: HARD RULE — never a flat/broadband/noise excitation drive; use P&S optimal or prior_robust uncertainty-aware multisine
metadata: 
  node_type: memory
  type: feedback
---

HARD RULE (user + P&S): the excitation drive must NEVER be flat/broadband or a
"noise drive." A flat-amplitude multisine justified by an "off-resonance SNR" target is, in
spirit, a noise drive — forbidden. "Off-resonance SNR" is a non-quantity for a parametric ML
fit (Fisher info lives at/near the resonances).

**Why:** P&S parametric ML allocates drive power by Fisher information, concentrated at the
modes — not spread flat to illuminate the whole FRF (that is the non-parametric/ETFE
mindset). Pure nominal-optimal concentration ALSO fails as a *starting* drive (it trusts an
uncertain model). The repo roadmap rule: "Do not regress to flat excitation."

**How to apply:** initial drive = uncertainty-aware multisine — `prior_robust_excitation`
(design/pintelon.py) averages the optimal design over the prior's plausible band
`f0·[1±u]`, with a meaningful floor (`floor_frac`·peak) so every line carries usable power
for iteration. Then iterate: estimate → shrink uncertainty → concentrate (loop.py does this
for SISO). Size the budget by the actuator limit, not an SNR target. See
[[crest-factor-lives-at-the-dac]] and [[stay-on-pintelon-schoukens]].
