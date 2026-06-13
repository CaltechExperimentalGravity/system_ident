# Prior / convergence bake-off — findings (2026-06-12)

Truth: single resonance f0=1.0 Hz, Q=20, gain=100. Cases: priors at ±10/20/50%
(plus a "mixed" case). Question: how to converge from a ±20-50% prior?

## 1. Local prior tweaks do NOT solve cold-start
Exhaustive multiprocess sweep (`parallel_sweep.py`): 51 configs × 7 cases × 6
seeds = 2142 campaigns across 12 cores. **Every** uniform-weaker-prior (the
"weaker prior" hypothesis), per-parameter Q/gain-anchor, and log-param config
converged **0/7**. Median f0 error ~0.18, Q error ~0.76 — i.e. Q and gain
*collapse* (the optimizer broadens a misplaced resonance to fit the off-resonance
data) and a far f0 freezes.

Root cause: a flat broadband sweep gives **uniform weighting**; a sharp resonance
at the wrong frequency is penalised so hard that lowering Q/gain reduces the
misfit more than sliding the peak does. A purely **local** Gauss-Newton/MAP step
cannot relocate the resonance — no amount of prior shaping fixes this.

## 2. The existing `broadband_ls` mode already solves cold-start
The shipped default loop mode (broadband excitation + global `invfreqs` fit +
inverse-variance accumulation across passes) converges **all 7 cases** — ±10%,
±20%, ±50%, and mixed — to f0≈0.99, Q≈19.5 in ~4 passes, and is **prior-
independent** (every prior lands on the same answer). A single `invfreqs` pass
already nails ±10-20%; accumulation over a few passes cracks ±50%. It also
handles low SNR by accumulating more passes.

## 3. Conclusion: the two modes are complementary, not competing
- **`broadband_ls`** — cold-start / characterization. Handles any prior (±50%+),
  prior-independent, robust to low SNR via accumulation. Use this to *find* the model.
- **`bayesian`** (recursive MAP, conservative small steps) — low-SNR *refinement*
  of an already-good model (within ~1-2 linewidths). Its unique value is using a
  good prior to reach a target uncertainty in *fewer* measurements when
  measurement time is precious — NOT cold-start. The "weaker prior" intuition is
  a prior-strength tradeoff *within this refinement regime*, not a cold-start fix.

Recommended pipeline for the real ±20-50%-prior + low-SNR envelope: run
`broadband_ls` first to lock the model, then switch to `bayesian` for ongoing
low-SNR tracking (or fold a broadband first pass into the bayesian mode).

## Open empirical question (the meaningful next bake-off)
In the *refinement* regime (good prior, concentrated/optimal excitation, low
SNR), does the Bayesian update reach a target fractional uncertainty in fewer
passes than `broadband_ls`, and what prior strength is best? That comparison must
run in the LOOP (concentrated excitation), not on this flat-broadband harness.
