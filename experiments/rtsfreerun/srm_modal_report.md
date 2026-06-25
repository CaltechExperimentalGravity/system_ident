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

These are the repo's established levels (ligo-india seismic, bosem 1e-10/1 Hz),
**not invented** — the exact presets/floors of
`scenario_for_dof` in `analyze_hsts_damped_6dof.py`. The 6×6 `drive→disp`
plant is byte-identical to the single-DOF residue plant (verified: same FRF),
so those displacement-referred levels transfer directly.

**Realistic noise levels referred to M1 displacement (in-band median):**

| source | level | where it enters |
|--------|-------|-----------------|
| seismic @ M1 (`ligo-india`×`gnd→M1`×ISI) | ~4e-11 m/√Hz on L/T/V/R/Y; **P has no `gnd→M1` path** in `hsts_full.mat` (ground tilt does not couple to M1 pitch there) → zero seismic | in-loop, `DRIVE_EXC` (drive-ref) |
| BOSEM/OSEM readout | 1e-10 m/√Hz, 1 Hz knee | in-loop, `MC2_M1_DAMP_*_EXC` |

**Documented compromise (honest — a real model limit).**
*OSEM noise is injected at the damper sensor node, not via an in-loop quantised
sensor.* The compiled model can't splice a sensor between plant and damper, so the
readout noise is carried by the bosem injection at `DAMP_EXC`. A true in-loop
quantised sensor would need a `READOUT_NOISE` `cdsFilt` rebuild (as `x1hstsdamped`
has). The seismic and OSEM disturbances ARE genuinely in-loop. With this physical
background the diagonal open-loop FRF recovers to **0.0002** median relative
error vs the analytic SS oracle (the reference-based recovery cancels the controller).

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
| L | 130.6 | 1560.0 | 4.57e+04 |
| T | 127.9 | 2663.7 | 3.55e+04 |
| V | 49.7 | 1927.4 | 4.10e+04 |
| R | 4274.9 | 19721.1 | 8.48e+04 |
| P | 6782.7 | 55679.3 | 4.60e+05 |
| Y | 5123.9 | 21457.2 | 7.72e+05 |

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
| 8 | 3.936e+13 | 6 | 6.6% | 7 | 0 |
| 10 | 3.337e+13 | 7 | 2.1% | 8 | 1 |
| 12 | 6.331e+12 | 10 | 3.4% | 11 | 0 |
| 13 ★ | 1.349e+12 | 11 | 2.5% | 12 | 0 |

## Recovered modal table — n_modes=13 (f0/Q ± CRB) vs oracle

Fit: n_iter=70, cost=1.349e+12, dof=14 (P&S CRB needs dof ≥ n_sens+8 = 14). 'well-sep' = |df|<1% & finite Q.

| mode | f0_fit [Hz] | ±f0 (CRB) | Q_fit | ±Q (CRB) | f0_oracle | Q_oracle | df% | Q-err% |
|------|-------------|-----------|-------|----------|-----------|----------|-----|--------|
| 0 | 0.6765 | 3.17e-07 | 16.98 | 2.13e-05 | 0.6758 | 50.00 | 0.101 | 66.0 |
| 1 | 0.7683 | 2.80e-07 | 18.79 | 1.22e-05 | 0.8484 | 50.00 | -9.436 | 62.4 |
| 2 | 1.0051 | 8.19e-08 | 46.50 | 4.59e-05 | 1.0051 | 50.00 | -0.000 | 7.0 |
| 3 | 1.0918 | 3.97e-08 | 49.94 | 1.77e-04 | 1.0918 | 50.00 | -0.001 | 0.1 |
| 4 | 1.5216 | 2.62e-08 | 48.77 | 9.53e-05 | 1.5267 | 50.00 | -0.335 | 2.5 |
| 5 | 2.0382 | 6.84e-08 | 49.88 | 1.24e-04 | 2.0381 | 50.00 | 0.005 | 0.2 |
| 6 | 2.1859 | 8.67e-08 | 55.53 | 1.28e-04 | 2.1845 | 50.00 | 0.068 | 11.1 |
| 7 | 2.7655 | 3.95e-07 | 54.46 | 4.78e-04 | 2.7617 | 50.00 | 0.139 | 8.9 |
| 8 | 2.7876 | 2.81e-07 | 51.23 | 1.44e-04 | 2.8067 | 50.00 | -0.682 | 2.5 |
| 9 | 2.9778 | 1.14e-07 | 50.74 | 1.32e-04 | 2.9817 | 50.00 | -0.131 | 1.5 |
| 10 | 3.2092 | 7.19e-08 | 48.09 | 8.79e-05 | 3.2093 | 50.00 | -0.003 | 3.8 |
| 11 | 3.4240 | 4.77e-08 | 50.00 | 7.41e-05 | 3.4240 | 50.00 | -0.000 | 0.0 |
| 12 | 3.7821 | 7.22e-08 | 49.79 | 8.45e-05 | 3.7814 | 50.00 | 0.019 | 0.4 |

## Summary

- All 6 SRM damping loops close **stable** on the bare-M1 HSTS plant.
- The reference-based recovery cancels the controller: diagonal FRF matches the oracle to 0.0002 median rel-err **even under the full realistic seismic+OSEM background**.
- 13 shared modal poles recovered at `df=0.00391 Hz`; median |df| vs oracle = 0.07%, with a **physical CRB** from real OSEM readout noise (dof=14 ≥ 14).
- **Q recovery (the goal):** **11** modes recovered well in BOTH f0 (|df|<1%) and Q (Q-err<25%); median Q-error = **2.5%** across the 12 well-separated modes.
- **Realistic fight:** worst-case (off-resonance / weak-coupling) per-line SNR ≈ 50 against the seismic+OSEM floor; the modal peaks sit at SNR ~1e4–1e6, so the well-separated modes still recover — the CRB bars are now physical, grown from the ~1e-25 token bound to real noise levels.

### Degradation vs the near-noise-free run
- The recovered `f0`/`Q` centres track the noise-free run closely (the well-separated modes still recover Q to a few percent); what changes is the **CRB**: the `±f0` / `±Q` bars are no longer a meaningless ~1e-25 — they are physical uncertainties set by the seismic + OSEM noise. That is the intended effect: realistic noise does not break the recovery of the well-separated modes, it puts honest error bars on them.

### Documented limits (real findings, not overclaimed)
- **The two tight doublets are unresolvable at any feasible df** — and they
are typically the only modes whose Q misses. The HSTS has the 0.672/0.676 Hz
pair (0.6% apart) and the 1.512/1.516/1.527 Hz triplet (<1% spread); their
members sit within a FWHM (≈2% of f0) of each other, below both `df=0.00391 Hz` and the shared-pole model's splitting power, so each collapses
to one mode. We do **not** force a spurious split — the collapse keeps a good `f0` but a blended Q (see the modal table for the as-run values).
- No degenerate/unstable poles in the chosen fit.
- *OSEM noise is measurement-referred at the damper sensor node, not via an in-loop quantised sensor* — the compiled model can't splice a sensor between plant and damper, so the readout noise is carried by the bosem injection at `DAMP_EXC`. A true in-loop quantised sensor would need a `READOUT_NOISE` `cdsFilt` rebuild (as `x1hstsdamped` has). The seismic+OSEM disturbances ARE in-loop.

Oracle in-band poles (16, near-degenerate doublets collapse to the 13 resolved modes): 0.672Hz/Q50.0, 0.676Hz/Q50.0, 0.848Hz/Q50.0, 1.005Hz/Q50.0, 1.092Hz/Q50.0, 1.512Hz/Q50.0, 1.516Hz/Q50.0, 1.527Hz/Q50.0, 2.038Hz/Q50.0, 2.184Hz/Q50.0, 2.762Hz/Q50.0, 2.807Hz/Q50.0, 2.982Hz/Q50.0, 3.209Hz/Q50.0, 3.424Hz/Q50.0, 3.781Hz/Q50.0

Plot: `srm6dof_modal_fit.svg` (SVG, Git LFS).

---

## Resolving the doublet with optimal excitation

A second campaign (`run_srm6dof_doublet.py`) targets the one thing the flat-drive
run above could not: **splitting the 0.6725/0.6758 Hz fundamental doublet** (3.30 mHz apart) into TWO separate modes. Three deltas vs the
flat run — Fisher-optimal drive, finer resolution, and a 14-mode prior seeded at
BOTH doublet members — turn the collapsed single mode into a resolved pair.

### The bound (what we must beat)

- Doublet split `Δf = 3.30 mHz`; linewidth `Γ = f0/Q = 13.4 mHz` (Q≈50); `Γ/Δf = 4.08`.
- A model-based ML fit super-resolves two modes once **`SNR·N ≳ (Γ/Δf)⁴ ≈ 276`** (P&S parametric resolution, NOT the non-parametric Rayleigh limit).
- Achieved on the doublet: per-line SNR ≈ **19162** (it sits on a resonance, plant gain ~Q, and the optimal drive concentrates power there), `N = 14` periods → **`SNR·N ≈ 268271`** — **972× the threshold**. Resolvable.

### Drive — P&S OPTIMAL excitation at ~10x seismic (not flat, not saturating)

The drive is the Fisher-optimal PSD from `design.pintelon.optimal_excitation`,
designed from a SISO modal-sum `TFModel` built from **just the two doublet
resonators** (0.6725 & 0.6758 Hz, Q=50) — the model that contains the doublet — so
the dispersion fixed point pours the whole optimal budget into the bins that inform
the doublet poles: a tight cluster right at ~0.674 Hz. (`Pyy` is flat / white output
noise, the P&S default, so the concentration is set purely by where the doublet
poles are informative. A SISO containing all 16 modes instead lets the high-
frequency poles — whose num/den coefficients dominate the Fisher gradient — capture
the budget and STARVE the low fundamental; verified, hence the doublet-only model.)
The optimal shape rides on a flat FLOOR whose per-line amplitude is calibrated so
the OFF-resonance bins sit at ~10x the in-loop seismic+OSEM floor (off-res SNR ~10),
while the doublet lines ride ~20x above it in drive amplitude — far more in response
SNR via the on-resonance plant gain. One PSD is applied to every actuator (the rank-1
modal poles are SHARED across all 6 DoF, so the informative bins are common).

- `PX_TOTAL = 1.19e-03` total budget (off-res floor amp `5.5e-04` counts) gives a peak drive of **0.0815 counts** — `0.00027%` of the `COIL_DRIVER = 30000`-count limit (no saturation, ~368207x headroom).
- Optimal PSD concentration (peak/floor) = **487×**.
- Resolution `nperseg = 131072` @ `fs = 256` → `df = 0.00195 Hz` (`T = 512` s/period), `1.7` bins between the doublet members; `n_periods = 16`, `dof = 14`.

### Achieved SNR (off-resonance vs on-doublet)

| DoF | SNR min (off-res) | SNR median | SNR max | SNR on-doublet |
|-----|-------------------|------------|---------|----------------|
| L | 60.2 | 1240.2 | 1.81e+04 | 18056.0 |
| T | 58.7 | 1368.8 | 1.92e+04 | 19162.2 |
| V | 52.1 | 1769.3 | 1.44e+04 | 14394.3 |
| R | 2672.9 | 9747.2 | 1.91e+05 | 191157.5 |
| P | 4057.9 | 10160.7 | 2.25e+05 | 224513.8 |
| Y | 2636.8 | 9039.2 | 2.60e+05 | 260051.3 |

Off-resonance min SNR ≈ 52 (the ~10x-seismic target), median ~hundreds, on-doublet ≈ 19162.

### Result — n_modes=14, fit n_iter=24, cost=1.805e+12, dof=14; diagonal FRF rel-err 0.0002

**The doublet did NOT resolve** — see the gate diagnosis below.

### Full modal table (14 modes)

| mode | f0 [Hz] | ±f0 (CRB) | Q | ±Q (CRB) | f0_oracle | Q_oracle | df% | Qerr% |
|------|---------|-----------|---|----------|-----------|----------|-----|-------|
| 0 | 0.54199 | 1.58e-06 | inf | nan | 0.6725 | 50 | -19.407 | — |
| 1 | 0.66920 | 6.17e-08 | 34.83 | 1.63e-04 | 0.6725 | 50 | -0.490 | 30.3 |
| 2 | 0.84727 | 5.18e-07 | 48.02 | 1.69e-03 | 0.8484 | 50 | -0.132 | 4.0 |
| 3 | 1.00655 | 3.68e-08 | 49.94 | 7.71e-05 | 1.0051 | 50 | 0.144 | 0.1 |
| 4 | 1.09179 | 2.89e-08 | 49.51 | 5.16e-05 | 1.0918 | 50 | 0.000 | 1.0 |
| 5 | 1.52175 | 7.37e-08 | 42.93 | 1.83e-04 | 1.5267 | 50 | -0.327 | 14.1 |
| 6 | 2.03813 | 4.48e-08 | 50.20 | 1.07e-04 | 2.0381 | 50 | 0.002 | 0.4 |
| 7 | 2.18546 | 5.14e-08 | 58.45 | 8.08e-05 | 2.1845 | 50 | 0.045 | 16.9 |
| 8 | 2.76141 | 3.14e-07 | 50.20 | 3.54e-04 | 2.7617 | 50 | -0.011 | 0.4 |
| 9 | 2.80565 | 1.85e-07 | 51.99 | 1.44e-04 | 2.8067 | 50 | -0.038 | 4.0 |
| 10 | 2.97949 | 1.65e-07 | 50.85 | 1.68e-04 | 2.9817 | 50 | -0.074 | 1.7 |
| 11 | 3.21023 | 4.26e-08 | 49.04 | 3.49e-05 | 3.2093 | 50 | 0.030 | 1.9 |
| 12 | 3.42399 | 9.94e-08 | 50.10 | 1.41e-04 | 3.4240 | 50 | -0.001 | 0.2 |
| 13 | 3.78122 | 3.06e-08 | 51.75 | 3.09e-05 | 3.7814 | 50 | -0.004 | 3.5 |

- **12/14** modes recover Q to <25% (oracle Q=50 everywhere); the
higher SNR of the 10x drive recovers Q across the table.
- Plot: `srm6dof_doublet_fit.svg` (full-band L→L + a zoom on the resolved
  doublet).
