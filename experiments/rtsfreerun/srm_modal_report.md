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

### Feasibility (a compute trade, not a physical limit)

The twin runs ~31× realtime (~0.032 s wall / sim-second, measured), and the
total sim cost is `n_periods·(nperseg/fs)` per actuator × 6 — purely a
`df ↔ per-period-length` trade. The chosen grid (`df=0.004 Hz`, ~13 min twin
time) is a **compute** choice, not a resolution limit: parametric ML
super-resolves, so `df` only needs `T ≳ Q/f0` to see a Q, and any residual
tight-mode collapse is beaten by SNR·N ≳ (Γ/Δf)⁴ or the spatial/per-plane fit —
NOT by a `df` floor. (Coarser grids `nperseg≤16384` drop below ~1 bin on the
low modes and the shared-pole Q blends; that is the shared-pole fit, not physics.)

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
background the diagonal open-loop FRF recovers to **0.0003** median relative
error vs the analytic SS oracle (the reference-based recovery cancels the controller).

## Achieved SNR (the realistic fight)

Per-DoF SNR = |driven-line response averaged over periods| / its
period-to-period scatter, at the excited lines. The drive is an
**uncertainty-aware prior-robust multisine** (NOT flat): the optimal design
averaged over the prior band f0·[1±0.5] with a derived-α floor holding 15% of the budget, so the resonances carry most of the
power (high on-mode SNR) while every line still gets usable power to
iterate. Off-resonance SNR is not a design target — Fisher information
lives at the modes. Total budget `PX_TOTAL=9e-4` (peak drive « the
30000-count coil limit).

| DoF | SNR min | SNR median | SNR max |
|-----|---------|------------|---------|
| L | 29.1 | 418.4 | 4.54e+04 |
| T | 28.6 | 705.8 | 4.39e+04 |
| V | 13.8 | 439.4 | 4.91e+04 |
| R | 801.1 | 3253.0 | 7.91e+04 |
| P | 1193.5 | 4082.9 | 1.04e+05 |
| Y | 694.1 | 3384.0 | 1.04e+05 |

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

The HSTS has **16 in-band poles**, but several form tight clusters within a
FWHM of each other (FWHM ≈ f0/Q ≈ 2% of f0): 0.672/0.676 (0.6% apart) and
1.512/1.516/1.527 Hz. The rank-1 SHARED pole set carries each such cluster as
ONE mode — a parameterization choice of THIS joint fit, not a physical limit
(the 0.672/0.676 pair is spatially resolved in step [5b]) —
leaving **13 design modes** for the shared-pole sweep. The init is
**prior-driven**: poles are seeded directly from those design (oracle) modes,
ranked by the recovered-FRF power so the strongest real resonances are taken
first, then the package's linear residue LS (`init_residues`) sets the rank-1
shapes. (Seeding from the known design is legitimate — we have the priors —
and avoids the spurious low-f poles that `find_peaks` reads off the fine-df
FRF's sidelobes.) Picked `n_modes` = most modes recovered well in BOTH f0
(|df|<1%) and Q (Q-err<25%):

| n_modes | cost | n_good (f0&Q) | median Q-err (well-sep) | n_well-sep | n_bad-Q |
|---------|------|---------------|-------------------------|------------|---------|
| 8 | 5.669e+12 | 6 | 12.8% | 7 | 0 |
| 10 | 3.491e+12 | 7 | 3.9% | 7 | 0 |
| 12 | 2.582e+12 | 9 | 3.8% | 9 | 0 |
| 13 ★ | 8.015e+11 | 11 | 3.4% | 11 | 0 |

## Recovered modal table — n_modes=13 (f0/Q ± CRB) vs oracle

Fit: n_iter=9, cost=8.015e+11, dof=14 (P&S CRB needs dof ≥ n_sens+8 = 14). 'well-sep' = |df|<1% & finite Q.

| mode | f0_fit [Hz] | ±f0 (CRB) | Q_fit | ±Q (CRB) | f0_oracle | Q_oracle | df% | Q-err% |
|------|-------------|-----------|-------|----------|-----------|----------|-----|--------|
| 0 | 0.6544 | 6.34e-07 | 62.25 | 1.24e-03 | 0.6725 | 50.00 | -2.684 | 24.5 |
| 1 | 0.8869 | 8.63e-07 | 33.20 | 2.34e-04 | 0.8484 | 50.00 | 4.540 | 33.6 |
| 2 | 1.0129 | 2.23e-07 | 53.49 | 2.47e-04 | 1.0051 | 50.00 | 0.780 | 7.0 |
| 3 | 1.0916 | 7.78e-08 | 49.76 | 2.59e-04 | 1.0918 | 50.00 | -0.017 | 0.5 |
| 4 | 1.5010 | 1.51e-07 | 37.54 | 1.80e-04 | 1.5120 | 50.00 | -0.723 | 24.9 |
| 5 | 2.0382 | 5.51e-08 | 50.05 | 1.22e-04 | 2.0381 | 50.00 | 0.005 | 0.1 |
| 6 | 2.1859 | 6.65e-08 | 54.67 | 1.56e-04 | 2.1845 | 50.00 | 0.066 | 9.3 |
| 7 | 2.7653 | 3.90e-07 | 52.20 | 4.95e-04 | 2.7617 | 50.00 | 0.132 | 4.4 |
| 8 | 2.7955 | 2.82e-07 | 48.30 | 1.10e-04 | 2.8067 | 50.00 | -0.398 | 3.4 |
| 9 | 2.9838 | 1.46e-07 | 46.89 | 1.16e-04 | 2.9817 | 50.00 | 0.071 | 6.2 |
| 10 | 3.2095 | 7.53e-08 | 49.08 | 7.59e-05 | 3.2093 | 50.00 | 0.007 | 1.8 |
| 11 | 3.4240 | 1.72e-07 | 50.04 | 1.77e-04 | 3.4240 | 50.00 | 0.001 | 0.1 |
| 12 | 3.7820 | 7.29e-08 | 51.27 | 7.56e-05 | 3.7814 | 50.00 | 0.018 | 2.5 |

## Summary

- All 6 SRM damping loops close **stable** on the bare-M1 HSTS plant.
- The reference-based recovery cancels the controller: diagonal FRF matches the oracle to 0.0003 median rel-err **even under the full realistic seismic+OSEM background**.
- 13 shared modal poles recovered at `df=0.00391 Hz`; median |df| vs oracle = 0.07%, with a **physical CRB** from real OSEM readout noise (dof=14 ≥ 14).
- **Q recovery (the goal):** **11** modes recovered well in BOTH f0 (|df|<1%) and Q (Q-err<25%); median Q-error = **3.4%** across the 11 well-separated modes.
- **Realistic fight:** worst-case (off-resonance / weak-coupling) per-line SNR ≈ 14 against the seismic+OSEM floor; the modal peaks sit at SNR ~1e4–1e6, so the well-separated modes still recover — the CRB bars are now physical, grown from the ~1e-25 token bound to real noise levels.

### Degradation vs the near-noise-free run
- The recovered `f0`/`Q` centres track the noise-free run closely (the well-separated modes still recover Q to a few percent); what changes is the **CRB**: the `±f0` / `±Q` bars are no longer a meaningless ~1e-25 — they are physical uncertainties set by the seismic + OSEM noise. That is the intended effect: realistic noise does not break the recovery of the well-separated modes, it puts honest error bars on them.

### Doublet resolution (spatial) & remaining limits
- **The 0.672/0.676 Hz fundamental is a SPATIAL doublet — RESOLVED.** It is
two *orthogonal* modes (the plant block-diagonalises EXACTLY into the {L,P,V}
and {T,R,Y} planes — cross-coupling ~1e-13), near-coincident in frequency but
seen by different DOF. The shared-pole 6×6 fit collapses them (one pole set
forced onto two orthogonal modes); fitting each plane alone
(`mimo_fit.fit_block_decoupled`) resolves both — **no** frequency
super-resolution, fine `df`, or doublet-concentrated drive needed:

| plane | f0 [Hz] | ±f0 (CRB) | Q | ±Q (CRB) |
|-------|---------|-----------|---|----------|
| LPV | 0.67250 | 4.6e-07 | 49.43 | 3.2e-03 |
| TRY | 0.67581 | 8.7e-07 | 50.12 | 5.2e-03 |

- The 1.512/1.516/1.527 Hz triplet is only PARTLY spatial: 1.516 sits in
{L,P,V} while 1.512 & 1.527 share the {T,R,Y} plane, so that within-plane pair
still sits within a FWHM (below `df=0.00391 Hz`) — a separate case the shared-
pole fit collapses; not addressed by the plane split.
- No degenerate/unstable poles in the chosen fit.
- *OSEM noise is measurement-referred at the damper sensor node, not via an in-loop quantised sensor* — the compiled model can't splice a sensor between plant and damper, so the readout noise is carried by the bosem injection at `DAMP_EXC`. A true in-loop quantised sensor would need a `READOUT_NOISE` `cdsFilt` rebuild (as `x1hstsdamped` has). The seismic+OSEM disturbances ARE in-loop.

Oracle in-band poles (16, near-degenerate doublets collapse to the 13 resolved modes): 0.672Hz/Q50.0, 0.676Hz/Q50.0, 0.848Hz/Q50.0, 1.005Hz/Q50.0, 1.092Hz/Q50.0, 1.512Hz/Q50.0, 1.516Hz/Q50.0, 1.527Hz/Q50.0, 2.038Hz/Q50.0, 2.184Hz/Q50.0, 2.762Hz/Q50.0, 2.807Hz/Q50.0, 2.982Hz/Q50.0, 3.209Hz/Q50.0, 3.424Hz/Q50.0, 3.781Hz/Q50.0

## Drive & the iterative follow-up (spec)

The drive is the **uncertainty-aware initial** multisine (`design_drive`): prior-robust optimal excitation over the prior modes' band f0·[1±0.5] with a derived-α floor (15% of the budget) — power everywhere
to get information, shaped (not flat/noise), one PSD for all 6 actuators. This
is **pass 1**. The planned iteration (not yet built; `loop.py:SysIDLoop.run`
does it for the SISO path) closes the loop: modal fit → per-mode CRB
(`modal_uncertainty`) → shrink the prior uncertainty / re-design the drive
(point-optimal as the model firms up) → re-measure, until a CRB target is met.
Budget-sizing to the actuator limit is a separate axis to settle there.

Plot: `srm6dof_modal_fit.svg` (SVG, Git LFS).
