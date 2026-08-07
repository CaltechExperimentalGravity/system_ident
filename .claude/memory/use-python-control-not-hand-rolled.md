---
name: use-python-control-not-hand-rolled
description: "HARD RULE: use python-control for anything the controls lib covers; never reinvent state-space/c2d/FRF/feedback in pure numpy/scipy"
metadata: 
  node_type: memory
  type: feedback
---

For any control-systems operation, use the **python-control** library (`import control`). Do
**not** hand-roll in pure numpy/scipy anything the controls lib already provides: state-space
objects, `forced_response` / `step_response` / `impulse_response`, frequency response
(`frequency_response` / `freqresp` / `bode`), continuous↔discrete (`c2d` / `sample_system`,
not a bespoke `cont2discrete` wrapper), `tf` / `zpk` / `ss` construction and conversions,
`feedback` / series / parallel interconnection, `minreal`, `ctrb`/`obsv`, `place`/`lqr`, etc.

**Why:** these are tested, correct, and idiomatic. Rolling my own bilinear transform, FRF loop,
or interconnection is reinventing the wheel, adds bugs, and diverges from the codebase — the
twin's pyctl side is built on python-control and `[[rank1-modal-mimo-fit]]`'s
`MIMOTwinBackend` already uses `control.forced_response`. Called out explicitly as a project rule.

**How to apply:** reach for `control` first. Only drop to numpy/scipy when nothing in
python-control covers the need — and when you do, say so and why. This complements the
pyctl-first workflow (validate analytically in python-control before the rtsfree composite).
