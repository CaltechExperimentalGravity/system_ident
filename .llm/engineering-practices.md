# Engineering practices — quantify before concluding

A hard checklist for `system_ident` work (and any subagent driving it). It exists because the
same failure kept happening: hitting an underwhelming result and declaring a "fundamental
limit" / "document the limitation" / "how do you want to proceed?", when a one-line bound
calculation refutes it and names the knob. Subagent briefs should say: **"apply
`.llm/engineering-practices.md`."**

## The gate (non-negotiable)

Before you do ANY of these on a stuck or underwhelming result:
- call it a **fundamental / physical / identifiability limit**,
- pivot to **"document the limitation,"**
- **ask the user how to proceed**,

**STOP and compute the relevant bound with real numbers, and post it.** If the information /
headroom is there, the failure is **implementation** (init, model order, parameterization,
conditioning, under-driving, a sign/index error) — fix that, don't declare a limit.

## The bound checklist (compute what applies, with numbers)

1. **CRB / Fisher information** for the quantity you're estimating. Is the observed error
   consistent with the CRB, or ≫ it? CRB ≪ error ⇒ implementation bug. Estimate the CRB from
   `(2 Re(JᴴJ))⁻¹` or a closed form; don't hand-wave.
2. **Resolution — parametric vs non-parametric.** Peak-picking / DFT-bin / Rayleigh "one
   linewidth" is the NON-parametric limit. A model-based ML fit super-resolves: two modes Δf
   apart, linewidth Γ=f0/Q, are resolvable once **SNR·N ≳ (Γ/Δf)⁴** (finite, beatable by SNR
   and record length). Never quote the non-parametric limit for a parametric fit.
3. **SNR + dynamic-range headroom.** Compute the achieved SNR AND the actuator/ADC headroom
   you're leaving on the table (drive vs `COIL_DRIVER` limit; ADC bits). Fisher ∝ SNR ∝
   drive². A "noise limit" reached while under-driving is not a limit.
4. **Frequency resolution.** `df = fs/nperseg = 1/T` vs the feature width / mode spacing you
   need (use `design.recommend_resolution`). To resolve a Q the record must span the
   ringdown: `T ≥ ~Q/f0`.
5. **Conditioning / numerics.** Condition number, overflow (high-order Vandermonde), rank.
   Normalize / reparameterize before blaming the method.
6. **Model order.** Are you fitting the right number of modes? Collapsing a doublet to one
   pole (under-modeling) or adding spare poles (spurious modes) is a setup error, not a data
   limit.

## The knobs you CONTROL (set them from the math, don't assume them fixed)

- **Drive amplitude** — up to the actuator (`COIL_DRIVER`) limit; Fisher-optimal is as much
  power as safe.
- **Excited-line placement** — cluster lines where the parameters you care about are
  informative (Fisher-optimal excitation; P&S). Flat broadband is rarely optimal.
- **Record length T = 1/df** — buys resolution and Q.
- **Model order / which modes** — seed and fit the modes you mean to resolve (both members of
  a close pair).
- **# periods** — averaging, lowers the CRB.

## Traps from this project's history (recognize these instantly)

- Non-parametric Rayleigh limit applied to a parametric ML fit ("sub-linewidth doublet
  unresolvable" — false).
- Under-driving (0.05 counts vs a 30000-count coil) then calling it "seismic/noise-limited."
- Parameterization/conditioning/under-modeling called a "method/identifiability limit"
  (common-denominator poles; the 6-DoF Q).
- Resolution coarser than the feature, called a "Q limit."
- Treating a design knob (drive, df, lines, order, periods) as a fixed constraint.

## When it genuinely IS a limit

Then say so — and SHOW the CRB/budget that proves it, the assumptions, and **the cost to beat
it** (how much more drive / time / SNR / resolution would, and why that's impractical). A
limit claim without the number behind it is not allowed.

## Tooling — use the library, don't reinvent it

For any control-systems operation, use **python-control** (`import control`). Do **not**
hand-roll in pure numpy/scipy anything the controls lib already provides: state-space objects,
`forced_response`/`step_response`/`impulse_response`, frequency response
(`frequency_response`/`bode`), continuous↔discrete (`c2d`/`sample_system` — not a bespoke
bilinear/`cont2discrete` wrapper), `tf`/`zpk`/`ss` construction and conversions,
`feedback`/series/parallel interconnection, `minreal`, `ctrb`/`obsv`, `place`/`lqr`. These are
tested, correct, and idiomatic; rolling your own adds bugs and diverges from the codebase (the
twin's pyctl side and `MIMOTwinBackend` already use python-control). Drop to numpy/scipy only
when nothing in python-control covers the need — and when you do, say why. This complements the
pyctl-first workflow: validate analytically in python-control before the rtsfree composite.

See also: memory `compute-the-bound-before-claiming-a-limit`,
`use-python-control-not-hand-rolled`, `.llm/pintelon-schoukens-mimo-fit.md`, and
`.llm/ps-book/README.md` (chapter map into P&S — find the chapter and read it before asserting
what the method can/can't do; the book is copyrighted and is not in the repo).
