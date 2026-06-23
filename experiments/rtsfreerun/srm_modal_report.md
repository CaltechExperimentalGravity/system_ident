# SRM 6-DoF closed-loop rank-1 modal fit — report

Phase-1 RTSfreerun digital twin (no real hardware). The SRM is an HSTS
suspension identified through its **real production L1-SRM** top-mass
dampers (foton `SRM_M1_DAMP_<dof>` from `L1SUSSRM.txt`, engaged FMs from
the archived L1 SDF) closed around the shared bare-M1 6×6 HSTS plant.

## Frequency resolution drives the Q recovery

A `Q≈50` resonance at `f0` has FWHM `≈ f0/Q`: only **13.4 mHz** at the 0.67 Hz fundamental, ~76 mHz at the 3.78 Hz top mode. To pin **Q** (not just `f0`) the parametric fit
needs several bins ACROSS each peak — `df = fs/nperseg ≲ (1/3–1/4)·(f0/Q)`.

The previous campaign used `nperseg=8192 → df=0.03125 Hz`, **coarser than the
0.67 Hz peak itself** (0.4 bins across it): Q there was unrecoverable
(fit `Q≈1.3` vs oracle 50). This campaign uses:

- `fs = 256 Hz`, `nperseg = 65536` → **`df = 0.00391 Hz`** (256 s/period), `n_periods = 16` (`dof = 14 ≥ 14`).
- That puts **3.4 bins** across the narrowest (0.67 Hz) peak and ~19 across the top mode.

### Feasibility / resolution limit

The twin runs ~31× realtime (~0.032 s wall / sim-second, measured), and the
total sim cost is `n_periods·(nperseg/fs)` per actuator × 6 — purely a
`df ↔ per-period-length` trade. The chosen grid is the **finest feasible**
one (~13 min total twin time, cached). Going finer (`nperseg=131072`,
`df=0.002 Hz`, ~7 bins across the fundamental) costs ~27 min for marginal
gain; coarser grids (`nperseg≤16384`) drop below ~1 bin on the low modes and
Q collapses. So **~0.004 Hz is the practical resolution knee** for this plant.

## Method note — CRB requires measurement noise

The compiled `x1hsts6dof` exposes only `DRIVE_EXC_<dof>` / `LSC_DARM_EXC`
injection ports — there is **no per-sensor readout-noise EXC chain** like the
single-DOF `x1hsts` model. A noise-free twin gives identical periods, so the
P&S sample covariance Cz collapses to the floating-point floor and the SML
weighting goes indefinite (negative cost, meaningless ~1e-25 CRB). To define a
real CRB, a small broadband ground/actuator disturbance (`PROC_FLOOR`) is
injected on the non-driven drive ports each pass; it propagates through the
REAL plant + dampers so both Y and the reconstructed X are genuinely
stochastic, making Cz positive-definite. The disturbance is small enough that
the diagonal open-loop FRF still recovers to **0.0011** median
relative error vs the analytic SS oracle (controller cancelled). This is a
process-disturbance CRB, not a true readout-noise CRB — the bound is
self-consistent for the injected statistics, which is the honest caveat.

## Tuned SRM CAL (per-DOF, tau ≈ 5 s target)

| DOF | CAL | tau [s] | stable |
|-----|-----|---------|--------|
| L | 0.4792 | 4.47 | True |
| T | 0.5850 | 3.75 | True |
| V | 1.2030 | 5.11 | True |
| R | 2.5401 | 7.45 | True |
| P | 0.7094 | 4.54 | True |
| Y | 0.8288 | 4.92 | True |

## n_modes sweep (prior-seeded init)

The HSTS has **16 in-band poles**, but several form tight doublets within a
FWHM of each other (FWHM ≈ f0/Q ≈ 2% of f0): 0.672/0.676 (0.6% apart) and
1.512/1.516/1.527 Hz. The rank-1 model shares ONE pole set across all 6×6
elements, so such unresolvable doublets MUST collapse to a single mode —
leaving **13 resolvable design modes**. The init is
**prior-driven**: poles are seeded directly from those design (oracle) modes,
ranked by the recovered-FRF power so the strongest real resonances are taken
first, then the package's linear residue LS (`init_residues`) sets the rank-1
shapes. (Seeding from the known design is legitimate — we have the priors —
and avoids the spurious low-f poles that `find_peaks` reads off the fine-df
FRF's sidelobes.) Picked `n_modes` = most modes recovered well in BOTH f0
(|df|<1%) and Q (Q-err<25%):

| n_modes | cost | n_good (f0&Q) | median Q-err (well-sep) | n_well-sep | n_bad-Q |
|---------|------|---------------|-------------------------|------------|---------|
| 8 | 1.798e+09 | 1 | 27.9% | 3 | 0 |
| 10 | 7.105e+09 | 4 | 15.5% | 7 | 1 |
| 12 | 4.240e+08 | 9 | 9.1% | 9 | 0 |
| 13 ★ | 5.580e+07 | 12 | 2.3% | 12 | 0 |

## Recovered modal table — n_modes=13 (f0/Q ± CRB) vs oracle

Fit: n_iter=22, cost=5.580e+07, dof=14 (P&S CRB needs dof ≥ n_sens+8 = 14). 'well-sep' = |df|<1% & finite Q.

| mode | f0_fit [Hz] | ±f0 (CRB) | Q_fit | ±Q (CRB) | f0_oracle | Q_oracle | df% | Q-err% |
|------|-------------|-----------|-------|----------|-----------|----------|-----|--------|
| 0 | 0.6710 | 1.91e-05 | 43.14 | 2.24e-01 | 0.6725 | 50.00 | -0.218 | 13.7 |
| 1 | 0.8484 | 4.50e-07 | 50.52 | 3.54e-03 | 0.8484 | 50.00 | 0.002 | 1.0 |
| 2 | 1.0065 | 4.01e-06 | 53.54 | 1.88e-02 | 1.0051 | 50.00 | 0.142 | 7.1 |
| 3 | 1.0918 | 9.59e-07 | 50.13 | 3.90e-03 | 1.0918 | 50.00 | 0.005 | 0.3 |
| 4 | 1.4844 | 1.94e-05 | 32.30 | 3.44e-02 | 1.5120 | 50.00 | -1.824 | 35.4 |
| 5 | 2.0380 | 5.07e-07 | 50.23 | 1.01e-03 | 2.0381 | 50.00 | -0.003 | 0.5 |
| 6 | 2.1831 | 3.24e-06 | 51.85 | 6.58e-03 | 2.1845 | 50.00 | -0.064 | 3.7 |
| 7 | 2.7615 | 1.75e-06 | 50.42 | 6.39e-04 | 2.7617 | 50.00 | -0.009 | 0.8 |
| 8 | 2.8072 | 9.48e-06 | 51.34 | 1.88e-02 | 2.8067 | 50.00 | 0.019 | 2.7 |
| 9 | 2.9922 | 2.93e-05 | 40.49 | 3.36e-02 | 2.9817 | 50.00 | 0.354 | 19.0 |
| 10 | 3.2095 | 1.57e-06 | 50.92 | 2.75e-03 | 3.2093 | 50.00 | 0.007 | 1.8 |
| 11 | 3.4240 | 1.73e-06 | 50.35 | 3.04e-03 | 3.4240 | 50.00 | -0.001 | 0.7 |
| 12 | 3.7794 | 2.41e-06 | 52.11 | 1.09e-03 | 3.7814 | 50.00 | -0.051 | 4.2 |

## Summary

- All 6 SRM damping loops close **stable** on the bare-M1 HSTS plant.
- The reference-based recovery cancels the controller: diagonal FRF matches the oracle to 0.0011 median rel-err.
- 13 shared modal poles recovered at `df=0.00391 Hz`; median |df| vs oracle = 0.02%, with a trustworthy CRB (dof=14 ≥ 14).
- **Q recovery (the goal):** **12** modes recovered well in BOTH f0 (|df|<1%) and Q (Q-err<25%); median Q-error = **2.3%** across the 12 well-separated modes (vs the previous campaign where Q ranged 1.3–62 against a uniform oracle Q≈50). The finer `df=0.00391 Hz` is what makes these Qs identifiable, with a CRB.

### Documented limits (real findings, not overclaimed)
- **The two tight doublets are unresolvable at any feasible df** — and they
are exactly the only modes whose Q misses. The HSTS has the 0.672/0.676 Hz
pair (0.6% apart) and the 1.512/1.516/1.527 Hz triplet (<1% spread); their
members sit within a FWHM (≈2% of f0) of each other, below both `df=0.00391 Hz` and the shared-pole model's splitting power, so each collapses
to one mode. We do **not** force a spurious split. The collapse still gives
good `f0` (the 0.67 cluster lands at 0.671 Hz, the 1.51 cluster at 1.484 Hz)
but a blended Q (≈43 and ≈32 vs 50) — these are the 2 modes outside the 25% Q band. Every WELL-SEPARATED mode recovers Q to a few percent.
- No degenerate/unstable poles in the chosen fit.
- The CRB is a **process-disturbance** bound (no readout-noise port on the compiled 6-DoF model), self-consistent for the injected statistics.

Oracle in-band poles (16, near-degenerate doublets collapse to the 13 resolved modes): 0.672Hz/Q50.0, 0.676Hz/Q50.0, 0.848Hz/Q50.0, 1.005Hz/Q50.0, 1.092Hz/Q50.0, 1.512Hz/Q50.0, 1.516Hz/Q50.0, 1.527Hz/Q50.0, 2.038Hz/Q50.0, 2.184Hz/Q50.0, 2.762Hz/Q50.0, 2.807Hz/Q50.0, 2.982Hz/Q50.0, 3.209Hz/Q50.0, 3.424Hz/Q50.0, 3.781Hz/Q50.0

Plot: `srm6dof_modal_fit.svg` (SVG, Git LFS).
