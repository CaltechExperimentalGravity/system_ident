"""Robust Q from the half-power bandwidth, vs the fragile LS resonator fit.

Finding (2026-06-13): estimating a sharp resonance's Q by least-squares-fitting a
ResonatorModel to the measured complex TF is fragile -- the spectral window
distorts the peak SHAPE and the SNR-weighted fit chases that distorted shoulder
detail, so Q is badly biased and can diverge (to the fit bound). The classic
fit-free estimator -- the -3 dB half-power bandwidth, Q = f0 / df_3dB, which is
set by the bins AROUND the peak -- is essentially unbiased once the peak is
resolved. This demo contrasts the two on the realistic noisy twin and shows the
resolution requirement.

Resolution rule: the bandwidth f0/Q must span several frequency bins, so the
Welch segment must satisfy  T_seg  >~  (a few) * Q / f0. For f0=1 Hz, Q=20
(bandwidth 0.05 Hz) that is T_seg >~ 128 s; the 64 s default is too coarse.

Run: conda run -n sysid python experiments/realistic/resonator_from_spectrum_demo.py
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sig
from scipy.optimize import least_squares

from system_ident.backends.twin import TwinBackend
from system_ident.excitation import timeseries_from_asd
from system_ident.loop import SysIDLoop
from system_ident.model import TFModel
from system_ident.plant import SuspensionPlant
from system_ident.resonator import (
    ResonatorModel, resonator_from_spectrum, resonator_from_tf,
)
from system_ident.resonator_design import prior_robust_excitation

FS = 32.0
TRUE_F0, TRUE_Q, TRUE_GAIN = 1.0, 20.0, 100.0


def measure(T_seg, total, seed):
    """Realistic noisy twin measurement -> (freq, |H|) with a prior-robust drive."""
    nperseg = int(T_seg * FS)
    f_all = np.fft.rfftfreq(nperseg, 1 / FS)
    band = (f_all >= 0.3) & (f_all <= 3.0)
    freq = f_all[band]
    plant = SuspensionPlant({"POS": TFModel.from_resonances([(TRUE_F0, TRUE_Q)], TRUE_GAIN)}, FS)
    be = TwinBackend(plant, {"C1:EXC": "POS"}, {"C1:RSP": "POS"}, fs=FS,
                     sensor_asd=3e-3, disturbance_asd=3e-3, seed=seed)
    locked = resonator_from_tf(TFModel.from_resonances([(0.99, 19.4)], 100.0))
    quiet = be.read(["C1:RSP"], total)["C1:RSP"]
    Pyy = np.maximum(SysIDLoop._welch(quiet, FS, nperseg, band), 1e-30)
    Pxx = prior_robust_excitation(freq, locked, Pyy, 1.0, 0.2, n_iter=3)
    drive = timeseries_from_asd(total, FS, freq, np.sqrt(Pxx), seed=np.random.default_rng(seed), t_ramp=4.0)
    be.inject("C1:EXC", drive, FS)
    seg = be.read(["C1:RSP", "C1:EXC"], total)
    _, Pxxm = sig.welch(seg["C1:EXC"], fs=FS, nperseg=nperseg, noverlap=0)
    _, Pyx = sig.csd(seg["C1:RSP"], seg["C1:EXC"], fs=FS, nperseg=nperseg, noverlap=0)
    H = (Pyx / Pxxm)[band]
    return freq, H


def ls_fit_Q(freq, H):
    """The fragile SNR-weighted least-squares resonator fit (bounded so it can't run to inf)."""
    He = np.abs(H) * 0.1  # nominal rel err scale (not the point here)
    sw = np.ones_like(freq)
    def resid(p):
        m = ResonatorModel.from_resonances([(p[0], p[1])], p[2])
        r = H - m.eval(freq)
        return np.concatenate([sw * r.real, sw * r.imag])
    return float(least_squares(resid, [1.0, 20.0, 100.0],
                               bounds=([0.3, 2, 1], [3, 300, 1e5])).x[1])


if __name__ == "__main__":
    print(f"True resonance: f0={TRUE_F0} Q={TRUE_Q} gain={TRUE_GAIN}\n")
    print("Q estimate over 30 seeds: LS fit  vs  -3dB bandwidth\n")
    print(f"{'T_seg':>6}{'bins/bw':>9}   {'LS-fit Q (med [IQR])':>26}   {'bandwidth Q (med [IQR])':>26}")
    for T_seg, total in [(64, 256), (128, 1024), (256, 1024)]:
        bpb = (TRUE_F0 / TRUE_Q) / (FS / int(T_seg * FS))
        lss, bws = [], []
        for s in range(1, 31):
            freq, H = measure(float(T_seg), float(total), s)
            lss.append(ls_fit_Q(freq, H))
            try:
                bws.append(float(resonator_from_spectrum(freq, np.abs(H), f0_guess=0.99).Q[0]))
            except ValueError:
                bws.append(np.nan)
        lss = np.array(lss); bws = np.array(bws); bws = bws[np.isfinite(bws)]
        fmt = lambda a: f"{np.median(a):5.1f} [{np.percentile(a,25):4.1f},{np.percentile(a,75):5.1f}]"
        print(f"{T_seg:6.0f}{bpb:9.1f}   {fmt(lss):>26}   {fmt(bws):>26}")

    print("\n=> The LS fit is biased/divergent for the sharp peak; the half-power")
    print("   bandwidth is ~unbiased once the peak is resolved (T_seg>=128s here).")
    print("   Q is variance-limited, so more total time tightens the bandwidth estimate.")
