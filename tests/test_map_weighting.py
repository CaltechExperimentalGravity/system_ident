"""Regression: the MAP refine must not bias Q with a concentrated drive.

Root cause (debugged 2026-06-13): ``bayesian_update`` weighted the TF-fit
residuals by ``1/H_err**2 = 1/(|H|**2 * rel_err**2)`` — the ML inverse-estimate-
variance weight. For a resonance the ``1/|H|**2`` factor gives the high-amplitude
*peak* (which carries the Q information) ~1e4-1e5x LESS weight than the small-|H|
off-resonance shoulders, so the fit chases the shoulders and Q runs away upward
(structurally, not from noise: it persists with heavy averaging). A flat
exploration floor masked it by restoring broadband coverage; a concentrated
(``prior_robust`` / optimal) drive removes the mask and Q diverges by the
per-pass +20% step cap.

Fix: weight by SNR with a CONSTANT amplitude reference (``1/(rel_err*H_ref)**2``),
keeping units consistent with the prior precision while letting the resonance
peak carry its due weight. With the fix Q converges to truth and stays there.
"""

import numpy as np

from system_ident.backends.twin import TwinBackend
from system_ident.estimators.bayesian import bayesian_update, prior_precision
from system_ident.excitation import timeseries_from_asd
from system_ident.loop import SysIDLoop
from system_ident.model import TFModel
from system_ident.plant import SuspensionPlant
from system_ident.resonator import resonator_from_tf
from system_ident.resonator_design import prior_robust_excitation


def _refine_trajectory(seed, n_pass=5, fs=32.0, T=64.0, n_seg=4):
    """Lock a Q=20 plant, then run n_pass concentrated-drive MAP refines."""
    dur = T * n_seg
    nperseg = int(T * fs)
    f_all = np.fft.rfftfreq(nperseg, 1 / fs)
    band = (f_all >= 0.1) & (f_all <= 5.0)
    freq = f_all[band]

    plant = SuspensionPlant({"POS": TFModel.from_resonances([(1.0, 20.0)], 100.0)}, fs)
    be = TwinBackend(plant, {"C1:EXC": "POS"}, {"C1:RSP": "POS"}, fs=fs,
                     sensor_asd=3e-3, disturbance_asd=3e-3, seed=seed)
    quiet = be.read(["C1:RSP"], dur)["C1:RSP"]
    Pyy = SysIDLoop._welch(quiet, fs, nperseg, band)

    # good lock from the broadband_ls phase (f0=0.99, Q=19.4 ~ truth)
    model = resonator_from_tf(TFModel.from_resonances([(0.99, 19.4)], 100.0))
    Lambda = prior_precision(model, 0.15)
    rng = np.random.default_rng(seed)

    for _ in range(n_pass):
        Pxx = prior_robust_excitation(freq, model, Pyy, 1.0, 0.2, n_iter=3)
        drive = timeseries_from_asd(dur, fs, freq, np.sqrt(Pxx), seed=rng, t_ramp=4.0)
        be.inject("C1:EXC", drive, fs)
        seg = be.read(["C1:RSP", "C1:EXC"], dur)
        H, He, _ = SysIDLoop._estimate_tf(seg["C1:EXC"], seg["C1:RSP"], fs, nperseg, band)
        model, Lambda = bayesian_update(freq, model, H, He, Lambda)
    return float(model.f0[0]), float(model.Q[0])


def test_concentrated_refine_does_not_run_Q_away():
    """Several concentrated refine passes keep Q near truth (20), not diverge.

    Pre-fix this diverges: Q climbs 23->28->34->...->40 by the +20% cap each pass.
    """
    for seed in (1, 3, 7):
        f0, Q = _refine_trajectory(seed)
        assert abs(f0 - 1.0) < 0.05, f"seed {seed}: f0={f0:.3f}"
        assert abs(Q - 20.0) / 20.0 < 0.25, f"seed {seed}: Q={Q:.1f} ran away from 20"
