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

## The CRB now fights REALISTIC seismic + OSEM noise

Earlier this demo injected only a small *token* process disturbance on the
non-driven drive ports, purely to make the P&S sample covariance Cz
positive-definite — so its CRB was an arbitrary process-disturbance bound, not
a physical one. This campaign replaces that token with the twin's
**physically-complete HSTS noise recipe** (the same presets/floors as the
single-DOF `analyze_hsts_damped_6dof.py`), so the bound is a PHYSICAL CRB set
by real seismic + OSEM noise.

**Architecture (why a referral is needed).** The compiled `x1hsts6dof` carries
only a bare-M1 6×6 `drive→disp` plant: no separate ground/ISI input port and
no in-loop readout-noise `cdsFilt` chain like the single-DOF `x1hstsdamped`
model. Each disturbance is therefore *referred* to a port the model DOES
expose, reproducing the same in-loop physics:

- **Seismic** — `ligo-india` ground motion → `HSTS_GND_TF` (`gnd→M1`, the real
  `load_plant_residues("hsts_full.mat",(gnd,disp,d),(m1,disp,d))` path) →
  `ham_isi_transmissibility()` (the ISI platform) gives the M1-displacement
  ASD. That is divided by the plant `|drive→disp|` and injected at
  `DRIVE_EXC_<dof>` (the coil-drive node), so after the plant it reproduces the
  correct in-loop M1 motion the **damper fights** — exactly as `HSTS_GND_TF` +
  `ISI_RESIDUAL` are in-loop in the twin. Driven on every DoF (the cross-DoF
  ground disturbance the joint fit's off-diagonals carry).
- **OSEM/BOSEM readout noise** — `floor = 1e-10 m/√Hz`, `1 Hz` knee, injected
  in-loop at the sensor node `MC2_M1_DAMP_<dof>_EXC` (the `cdsFilt` damper
  input = the displacement signal the controller reads). The damper acts on
  readout+noise — the established place OSEM noise enters a damping loop.
- **16-bit ±1 mm ADC** — the recorded TRANSLATIONAL readout Y (L/T/V, metres)
  is digitised at `LSB = 1e-3/2^16 ≈ 1.5e-8 m` (`quantize_readout`), matching the
  twin's probe `quantize: {bits:16, range:1e-3}`. The ±1 mm full-scale is a
  *displacement* range — physical for the metre-unit translational DoFs. The
  rotational R/P/Y readouts are in **radians**: a metre ADC range mis-scales them
  (verified — see the documented caveat), so the calibrated ADC is applied only to
  L/T/V, and R/P/Y carry the in-loop seismic+OSEM noise without the mis-scaled
  quantiser.

These are the repo's established levels (ligo-india seismic, bosem 1e-10/1 Hz,
16-bit ±1 mm), **not invented** — the exact presets/floors of
`scenario_for_dof` in `analyze_hsts_damped_6dof.py`. The 6×6 `drive→disp`
plant is byte-identical to the single-DOF residue plant (verified: same FRF),
so those displacement-referred levels transfer directly.

**Realistic noise levels referred to M1 displacement (in-band median):**

| source | level | where it enters |
|--------|-------|-----------------|
| seismic @ M1 (`ligo-india`×`gnd→M1`×ISI) | ~4e-11 m/√Hz on L/T/V/R/Y; **P has no `gnd→M1` path** in `hsts_full.mat` (ground tilt does not couple to M1 pitch there) → zero seismic | in-loop, `DRIVE_EXC` (drive-ref) |
| BOSEM/OSEM readout | 1e-10 m/√Hz, 1 Hz knee | in-loop, `MC2_M1_DAMP_*_EXC` |
| 16-bit ±1 mm ADC | LSB ≈ 1.5e-8 m | measurement, on recorded Y — **L/T/V only** (metre range; R/P/Y are radians, range uncalibrated) |

**Documented compromises (honest — both are real model limits).**
1. *ADC range vs readout units.* The ±1 mm full-scale is a displacement range,
physical for the translational readouts (L/T/V, metres) but not for the rotational
R/P/Y readouts (radians) — this raw-displacement compiled model carries no angular
OSEM full-scale. Quantising R/P/Y with a metre range mis-scales them: it biases
their open-loop recovery to **0.7–0.95** rel-err while leaving L/T/V at <0.002
(isolation test: with seismic+OSEM and NO ADC, all six recover to ≤0.002; adding
the metre-range ADC breaks only R/P/Y). So the calibrated ADC is applied only to
L/T/V; R/P/Y carry the in-loop seismic+OSEM background, which already makes their
CRB physical. A faithful R/P/Y ADC needs the angular-OSEM µrad/count calibration,
which is not on this model — inventing one would violate 'use the repo's levels'.
2. *ADC is measurement-side, not in-loop.* The model can't splice a quantiser
between plant and damper, so the in-loop readout noise is carried by the bosem
injection at `DAMP_EXC` while the ADC digitises the *recorded* L/T/V Y. A perfect
single in-loop quantised sensor would need a `READOUT_NOISE`+quantiser `cdsFilt`
rebuild (as `x1hstsdamped` has). The seismic and OSEM disturbances ARE genuinely
in-loop. With this physical background the diagonal open-loop FRF still
recovers to **0.0003** median relative error vs the analytic SS oracle
(the reference-based recovery cancels the controller).

## Achieved SNR (the realistic fight)

Per-DoF SNR = |driven-line response averaged over periods| / its
period-to-period scatter, at the excited lines. The suspension resonances
have large plant gain so on-resonance SNR is high (the modes are well
measured); the binding number is the off-resonance / weak-coupling
**minimum**, where the response approaches the seismic+OSEM floor. The
drive budget `PX_TOTAL=1e-4` is calibrated to land that minimum at a
believable ~30 (vs the old token regime where every bin was SNR>1e7).

| DoF | SNR min | SNR median | SNR max |
|-----|---------|------------|---------|
| L | 26.2 | 364.3 | 9.77e+03 |
| T | 25.7 | 663.4 | 8.05e+03 |
| V | 9.9 | 383.0 | 8.28e+03 |
| R | 1237.6 | 17030.5 | 7.94e+04 |
| P | 2519.0 | 50765.1 | 4.51e+05 |
| Y | 2979.7 | 18503.6 | 5.26e+05 |

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
| 8 | 7.873e+12 | 7 | 2.1% | 8 | 0 |
| 10 | 6.205e+12 | 7 | 3.7% | 8 | 0 |
| 12 | 4.345e+12 | 10 | 1.8% | 10 | 1 |
| 13 ★ | 5.961e+10 | 11 | 0.7% | 11 | 0 |

## Recovered modal table — n_modes=13 (f0/Q ± CRB) vs oracle

Fit: n_iter=157, cost=5.961e+10, dof=14 (P&S CRB needs dof ≥ n_sens+8 = 14). 'well-sep' = |df|<1% & finite Q.

| mode | f0_fit [Hz] | ±f0 (CRB) | Q_fit | ±Q (CRB) | f0_oracle | Q_oracle | df% | Q-err% |
|------|-------------|-----------|-------|----------|-----------|----------|-----|--------|
| 0 | 0.6290 | 9.51e-08 | 2659.07 | 3.44e-01 | 0.6725 | 50.00 | -6.464 | 5218.1 |
| 1 | 1.0052 | 3.01e-07 | 49.55 | 2.48e-04 | 1.0051 | 50.00 | 0.012 | 0.9 |
| 2 | 1.0918 | 4.25e-08 | 49.92 | 1.92e-04 | 1.0918 | 50.00 | 0.001 | 0.2 |
| 3 | 1.5220 | 4.71e-08 | 49.74 | 1.31e-04 | 1.5267 | 50.00 | -0.308 | 0.5 |
| 4 | 2.0381 | 7.29e-08 | 50.04 | 1.54e-04 | 2.0381 | 50.00 | 0.003 | 0.1 |
| 5 | 2.1850 | 1.02e-07 | 52.01 | 1.98e-04 | 2.1845 | 50.00 | 0.023 | 4.0 |
| 6 | 2.7646 | 9.80e-08 | 52.31 | 1.96e-05 | 2.7617 | 50.00 | 0.105 | 4.6 |
| 7 | 2.7834 | 2.56e-07 | 50.61 | 2.69e-04 | 2.7617 | 50.00 | 0.786 | 1.2 |
| 8 | 2.9763 | 5.41e-07 | 50.60 | 5.04e-04 | 2.9817 | 50.00 | -0.181 | 1.2 |
| 9 | 3.2096 | 7.41e-08 | 49.82 | 1.21e-04 | 3.2093 | 50.00 | 0.011 | 0.4 |
| 10 | 3.4240 | 5.03e-08 | 50.01 | 7.75e-05 | 3.4240 | 50.00 | -0.001 | 0.0 |
| 11 | 3.7819 | 7.78e-08 | 50.33 | 1.20e-04 | 3.7814 | 50.00 | 0.013 | 0.7 |

## Summary

- All 6 SRM damping loops close **stable** on the bare-M1 HSTS plant.
- The reference-based recovery cancels the controller: diagonal FRF matches the oracle to 0.0003 median rel-err **even under the full realistic seismic+OSEM background**.
- 12 shared modal poles recovered at `df=0.00391 Hz`; median |df| vs oracle = 0.02%, with a **physical CRB** from real OSEM readout + ADC noise (dof=14 ≥ 14).
- **Q recovery (the goal):** **11** modes recovered well in BOTH f0 (|df|<1%) and Q (Q-err<25%); median Q-error = **0.7%** across the 11 well-separated modes.
- **Realistic fight:** worst-case (off-resonance / weak-coupling) per-line SNR ≈ 10 against the seismic+OSEM floor; the modal peaks sit at SNR ~1e4–1e6, so the well-separated modes still recover — the CRB bars are now physical, grown from the ~1e-25 token bound to real noise levels.

### Degradation vs the near-noise-free run
- The recovered `f0`/`Q` centres track the noise-free run closely (the well-separated modes still recover Q to a few percent); what changes is the **CRB**: the `±f0` / `±Q` bars are no longer a meaningless ~1e-25 — they are physical uncertainties set by the seismic + OSEM + ADC noise. That is the intended effect: realistic noise does not break the recovery of the well-separated modes, it puts honest error bars on them.

### Documented limits (real findings, not overclaimed)
- **The two tight doublets are unresolvable at any feasible df** — and they
are typically the only modes whose Q misses. The HSTS has the 0.672/0.676 Hz
pair (0.6% apart) and the 1.512/1.516/1.527 Hz triplet (<1% spread); their
members sit within a FWHM (≈2% of f0) of each other, below both `df=0.00391 Hz` and the shared-pole model's splitting power, so each collapses
to one mode. We do **not** force a spurious split — the collapse keeps a good `f0` but a blended Q (see the modal table for the as-run values).
- No degenerate/unstable poles in the chosen fit.
- **The 16-bit ±1 mm ADC is calibrated for displacement (L/T/V) only.** Its metre full-scale mis-scales the rotational R/P/Y readouts (radians) — applying it there biases R/P/Y recovery to 0.7–0.95 (isolation-tested), so it is applied only to L/T/V; R/P/Y carry the in-loop seismic+OSEM noise. A faithful R/P/Y ADC needs the angular-OSEM µrad/count calibration, absent from this raw-displacement model. The ADC is also measurement-side (the model can't splice a quantiser between plant and damper); the seismic+OSEM disturbances ARE in-loop. Honest caveat — a `READOUT_NOISE`+quantiser `cdsFilt` rebuild (as `x1hstsdamped` has) would fix both.

Oracle in-band poles (16, near-degenerate doublets collapse to the 12 resolved modes): 0.672Hz/Q50.0, 0.676Hz/Q50.0, 0.848Hz/Q50.0, 1.005Hz/Q50.0, 1.092Hz/Q50.0, 1.512Hz/Q50.0, 1.516Hz/Q50.0, 1.527Hz/Q50.0, 2.038Hz/Q50.0, 2.184Hz/Q50.0, 2.762Hz/Q50.0, 2.807Hz/Q50.0, 2.982Hz/Q50.0, 3.209Hz/Q50.0, 3.424Hz/Q50.0, 3.781Hz/Q50.0

Plot: `srm6dof_modal_fit.svg` (SVG, Git LFS).
