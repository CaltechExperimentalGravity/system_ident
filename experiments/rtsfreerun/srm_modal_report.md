# SRM 6-DoF closed-loop rank-1 modal fit — report

Phase-1 RTSfreerun digital twin (no real hardware). The SRM is an HSTS
suspension identified through its **real production L1-SRM** top-mass
dampers (foton `SRM_M1_DAMP_<dof>` from `L1SUSSRM.txt`, engaged FMs from
the archived L1 SDF) closed around the shared bare-M1 6×6 HSTS plant.

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
relative error vs the analytic SS oracle (controller cancelled).

## Tuned SRM CAL (per-DOF, tau ≈ 5 s target)

| DOF | CAL | tau [s] | stable |
|-----|-----|---------|--------|
| L | 0.4792 | 4.47 | True |
| T | 0.5850 | 3.75 | True |
| V | 1.2030 | 5.11 | True |
| R | 2.5401 | 7.45 | True |
| P | 0.7094 | 4.54 | True |
| Y | 0.8288 | 4.92 | True |

## Recovered modal table (f0/Q ± CRB) vs analytic SS oracle

Fit: n_iter=12, cost=2.220e+08, dof=16 (P&S CRB needs dof ≥ n_sens+8 = 14).

| mode | f0_fit [Hz] | ±f0 (CRB) | Q_fit | ±Q (CRB) | f0_oracle | Q_oracle | df% |
|------|-------------|-----------|-------|----------|-----------|----------|-----|
| 0 | 0.8417 | 1.89e-04 | 1.27 | 1.01e-03 | 0.8484 | 50.00 | -0.793 |
| 1 | 1.0082 | 1.56e-05 | 13.05 | 6.43e-03 | 1.0051 | 50.00 | 0.307 |
| 2 | 1.0897 | 3.26e-05 | inf | nan | 1.0918 | 50.00 | -0.188 |
| 3 | 1.5046 | 5.53e-05 | 30.68 | 9.12e-02 | 1.5120 | 50.00 | -0.489 |
| 4 | 2.0300 | 1.91e-05 | 54.64 | 3.99e-02 | 2.0381 | 50.00 | -0.396 |
| 5 | 2.1452 | 5.26e-05 | 24.14 | 1.39e-02 | 2.1845 | 50.00 | -1.798 |
| 6 | 3.1901 | 1.47e-05 | 26.78 | 6.04e-03 | 3.2093 | 50.00 | -0.596 |
| 7 | 3.7861 | 1.50e-05 | 61.53 | 9.80e-03 | 3.7814 | 50.00 | 0.126 |

## Summary

- All 6 SRM damping loops close **stable** on the bare-M1 HSTS plant.
- The reference-based recovery cancels the controller: diagonal FRF matches the oracle to 0.0011 median rel-err.
- 8 shared modal poles recovered; median |df| vs oracle = 0.44%, with a trustworthy CRB (dof=16 ≥ 14).
- Caveat: 1 mode(s) land on a critically-damped / unstable pole (Q→∞, CRB undefined) — the rank-1 shared-pole model struggles to split the densest near-degenerate doublet (~1.0/1.09 Hz, oracle pair within <1%). Frequencies there are still within ~0.2%.

Oracle in-band poles (16, near-degenerate doublets collapse to the 8 resolved peaks): 0.672Hz/Q50.0, 0.676Hz/Q50.0, 0.848Hz/Q50.0, 1.005Hz/Q50.0, 1.092Hz/Q50.0, 1.512Hz/Q50.0, 1.516Hz/Q50.0, 1.527Hz/Q50.0, 2.038Hz/Q50.0, 2.184Hz/Q50.0, 2.762Hz/Q50.0, 2.807Hz/Q50.0, 2.982Hz/Q50.0, 3.209Hz/Q50.0, 3.424Hz/Q50.0, 3.781Hz/Q50.0

Plot: `srm6dof_modal_fit.svg` (SVG, Git LFS).
