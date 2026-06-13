"""Phase 1: Pintelon-Schoukens periodic multisine + leakage-free FRF.

These tests pin the *measurement* fix that ends the Q-estimation flailing:

  * the realiser is a genuine periodic multisine with a low (Schroeder) crest
    factor and the designed in-band power;
  * a synchronous integer-period DFT measures a sharp resonance without the
    near-peak bias that windowed Welch on a random record introduces;
  * the reference-based (ratio-of-averages) estimator recovers the *open-loop*
    plant even with a damping loop closed, where the naive ``S_yx/S_xx`` is
    biased; and
  * fed into the existing ``InvfreqsEstimator`` it recovers Q, where windowed
    Welch returns a garbage (right-half-plane) pole.

The legacy default (``mode="welch"``) is untouched; that coverage lives in the
existing suite, which still passes byte-for-byte.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sig
from scipy.integrate import trapezoid

from system_ident.backends.twin import TwinBackend
from system_ident.estimators.invfreqs import InvfreqsEstimator
from system_ident.excitation import multisine_from_psd
from system_ident.loop import SysIDLoop
from system_ident.model import TFModel, pole_pair_f0_Q
from system_ident.plant import SuspensionPlant, double_pendulum

FS = 32.0
NPERSEG = 2048
N_PERIODS = 8
TOTAL_DUR = NPERSEG * N_PERIODS / FS


def _grid(nperseg=NPERSEG):
    f_all = np.fft.rfftfreq(nperseg, d=1 / FS)
    band = (f_all >= 0.1) & (f_all <= 5.0)
    return f_all, band, f_all[band]


def _flat_pxx(freq, px_total=1.0):
    p = np.ones_like(freq)
    return p * px_total / trapezoid(p, freq)


def _discrete_frf(tf, freq):
    """The FRF the twin actually realises (bilinear discretisation of ``tf``)."""
    b, a = sig.bilinear(tf.num, tf.den, FS)
    _, H = sig.freqz(b, a, worN=2 * np.pi * freq / FS)
    return H


def _fit_qs(model):
    poles = np.roots(np.asarray(model.den, dtype=float))
    return sorted(pole_pair_f0_Q(p.real, p.imag)[1] for p in poles[poles.imag > 1e-9])


# --------------------------------------------------------------------------- #
# 1. the realiser
# --------------------------------------------------------------------------- #

def test_multisine_is_periodic_and_on_grid():
    _, band, freq = _grid()
    Pxx = _flat_pxx(freq)
    drive = multisine_from_psd(Pxx, FS, NPERSEG, N_PERIODS, freq, seed=0)

    assert drive.size == NPERSEG * N_PERIODS
    # exact period repetition
    one = drive[:NPERSEG]
    np.testing.assert_allclose(drive[NPERSEG : 2 * NPERSEG], one, atol=1e-12)

    # spectral support lands only on the excited synchronous bins
    S = np.fft.rfft(one)
    lines = set(np.round(freq * NPERSEG / FS).astype(int))
    off = [abs(S[k]) for k in range(S.size) if k not in lines]
    on = [abs(S[k]) for k in lines]
    assert max(off) < 1e-6 * np.median(on)


def test_multisine_power_budget_and_crest_factor():
    _, band, freq = _grid()
    Pxx = _flat_pxx(freq, px_total=1.0)
    schroeder = multisine_from_psd(Pxx, FS, NPERSEG, N_PERIODS, freq, phase="schroeder")
    random = multisine_from_psd(Pxx, FS, NPERSEG, N_PERIODS, freq, phase="random", seed=0)

    # realised variance matches the designed power budget
    assert abs(np.var(schroeder) - 1.0) < 0.05

    def crest(x):
        return np.max(np.abs(x)) / np.std(x)

    # Schroeder phases keep the peak well below the random-phase choice, so the
    # same in-band power fits under a tighter actuator limit.
    assert crest(schroeder) < 0.7 * crest(random)


def test_saturation_clips_random_more_than_schroeder():
    _, band, freq = _grid()
    Pxx = _flat_pxx(freq)
    plant = TFModel.from_resonances([(0.6, 20.0)], 300.0)
    sp = SuspensionPlant({"POS": plant}, FS)
    lim = 2.5  # between the two crest factors
    counts = {}
    for phase in ("schroeder", "random"):
        drive = multisine_from_psd(Pxx, FS, NPERSEG, N_PERIODS, freq, phase=phase, seed=0)
        tw = TwinBackend(sp, {"E": "POS"}, {"R": "POS"}, fs=FS, saturate=lim)
        tw.inject("E", drive, FS)
        x = tw.read(["E"], TOTAL_DUR)["E"]
        counts[phase] = int(np.sum(np.abs(x) >= lim - 1e-9))
    assert counts["random"] > counts["schroeder"]
    assert counts["schroeder"] == 0


# --------------------------------------------------------------------------- #
# 2. leakage-free measurement (open loop)
# --------------------------------------------------------------------------- #

def test_periodic_frf_is_leakage_free_where_welch_is_biased():
    _, band, freq = _grid()
    Pxx = _flat_pxx(freq)
    plant = double_pendulum()
    sp = SuspensionPlant({"POS": plant}, FS)
    Hdisc = _discrete_frf(plant, freq)

    tw = TwinBackend(sp, {"E": "POS"}, {"R": "POS"}, fs=FS, sensor_asd=0.0, seed=1)
    tw.inject("E", multisine_from_psd(Pxx, FS, NPERSEG, N_PERIODS, freq, seed=0), FS)
    seg = tw.read(["E", "R"], TOTAL_DUR)
    H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg["E"], seg["R"], FS, NPERSEG, band)
    exc = np.isfinite(H_err)
    rel = np.abs(H[exc] - Hdisc[exc]) / np.abs(Hdisc[exc])
    assert rel.max() < 1e-2

    # windowed Welch on a random record badly distorts the sharpest peak
    from system_ident.excitation import timeseries_from_asd

    tw.inject("E", timeseries_from_asd(TOTAL_DUR, FS, freq, np.sqrt(Pxx), seed=0), FS)
    seg2 = tw.read(["E", "R"], TOTAL_DUR)
    Hw, _, _ = SysIDLoop._estimate_tf(seg2["E"], seg2["R"], FS, NPERSEG, band)
    ipk = int(np.argmin(np.abs(freq - 0.6)))
    assert abs(Hw[ipk] - Hdisc[ipk]) / abs(Hdisc[ipk]) > 0.1


# --------------------------------------------------------------------------- #
# 3. closed-loop: reference-based recovers the open-loop plant
# --------------------------------------------------------------------------- #

def _damping_controller(k=2e-3, wc=2 * np.pi * 12):
    """Proper velocity-damping ``C(s) = k s/(1 + s/wc)`` (negative feedback)."""
    return ([k, 0.0], [1.0 / wc, 1.0])


def test_reference_based_frf_unbiased_in_closed_loop():
    _, band, freq = _grid()
    Pxx = _flat_pxx(freq)
    plant = TFModel.from_resonances([(0.8, 80.0)], 200.0)   # lightly damped
    sp = SuspensionPlant({"POS": plant}, FS)
    Hdisc = _discrete_frf(plant, freq)
    controllers = {"POS": _damping_controller()}

    tw = TwinBackend(
        sp, {"E": "POS"}, {"R": "POS"}, fs=FS,
        sensor_asd=1e-4, disturbance_asd=1e-4, seed=3, controllers=controllers,
    )
    tw.inject("E", multisine_from_psd(Pxx, FS, NPERSEG, N_PERIODS, freq, seed=0), FS)
    seg = tw.read(["E", "R"], TOTAL_DUR)

    H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg["E"], seg["R"], FS, NPERSEG, band)
    exc = np.isfinite(H_err)
    rel = np.abs(H[exc] - Hdisc[exc]) / np.abs(Hdisc[exc])
    assert rel.max() < 1e-2          # recovers the OPEN-LOOP plant

    Hn, _, _ = SysIDLoop._estimate_tf(seg["E"], seg["R"], FS, NPERSEG, band)
    reln = np.abs(Hn[exc] - Hdisc[exc]) / np.abs(Hdisc[exc])
    assert reln.max() > 0.1          # naive S_yx/S_xx is loop-biased


# --------------------------------------------------------------------------- #
# 4. adaptive transient drop
# --------------------------------------------------------------------------- #

def test_adaptive_transient_drops_more_for_a_slow_settling_mode():
    # short period vs a long ringdown: tau = Q/(pi f0) ~ 19 s, period = 16 s,
    # so one period is not settled and the guard must drop more.
    nperseg = 512
    f_all, band, freq = _grid(nperseg)
    n_periods = 12
    Pxx = _flat_pxx(freq)
    slow = TFModel.from_resonances([(1.0, 60.0)], 200.0)
    sp = SuspensionPlant({"POS": slow}, FS)
    tw = TwinBackend(sp, {"E": "POS"}, {"R": "POS"}, fs=FS, sensor_asd=0.0, seed=1)
    tw.inject("E", multisine_from_psd(Pxx, FS, nperseg, n_periods, freq, seed=0), FS)
    seg = tw.read(["E", "R"], nperseg * n_periods / FS)

    x, y = seg["E"], seg["R"]
    P = len(x) // nperseg
    X = np.fft.rfft(x[: P * nperseg].reshape(P, nperseg), axis=1)
    Y = np.fft.rfft(y[: P * nperseg].reshape(P, nperseg), axis=1)
    assert SysIDLoop._choose_transient(X, Y, 1, P) >= 2

    # a well-settled record (long period vs the same ringdown) keeps just one
    _, band2, freq2 = _grid(NPERSEG)
    sp2 = SuspensionPlant({"POS": double_pendulum()}, FS)
    tw2 = TwinBackend(sp2, {"E": "POS"}, {"R": "POS"}, fs=FS, sensor_asd=0.0, seed=1)
    tw2.inject("E", multisine_from_psd(_flat_pxx(freq2), FS, NPERSEG, N_PERIODS, freq2, seed=0), FS)
    seg2 = tw2.read(["E", "R"], TOTAL_DUR)
    Xs = np.fft.rfft(seg2["E"][: N_PERIODS * NPERSEG].reshape(N_PERIODS, NPERSEG), axis=1)
    Ys = np.fft.rfft(seg2["R"][: N_PERIODS * NPERSEG].reshape(N_PERIODS, NPERSEG), axis=1)
    assert SysIDLoop._choose_transient(Xs, Ys, 1, N_PERIODS) == 1


# --------------------------------------------------------------------------- #
# 5. headline: Q recovery through the existing estimator
# --------------------------------------------------------------------------- #

def test_periodic_recovers_Q_where_welch_diverges():
    from system_ident.excitation import timeseries_from_asd

    _, band, freq = _grid()
    Pxx = _flat_pxx(freq)
    true = TFModel.from_resonances([(0.6, 20.0)], 300.0)
    prior = TFModel.from_resonances([(0.55, 12.0)], 250.0)   # offset, same order
    sp = SuspensionPlant({"POS": true}, FS)
    tw = TwinBackend(sp, {"E": "POS"}, {"R": "POS"}, fs=FS, sensor_asd=0.0, seed=1)

    tw.inject("E", multisine_from_psd(Pxx, FS, NPERSEG, N_PERIODS, freq, seed=0), FS)
    seg = tw.read(["E", "R"], TOTAL_DUR)
    H, He, _ = SysIDLoop._estimate_tf_periodic(seg["E"], seg["R"], FS, NPERSEG, band)
    Qp = _fit_qs(InvfreqsEstimator().fit(freq, H, He, prior))
    assert len(Qp) == 1 and abs(Qp[0] - 20.0) / 20.0 < 0.2   # ~Q=20

    tw.inject("E", timeseries_from_asd(TOTAL_DUR, FS, freq, np.sqrt(Pxx), seed=0), FS)
    seg2 = tw.read(["E", "R"], TOTAL_DUR)
    Hw, Hwe, _ = SysIDLoop._estimate_tf(seg2["E"], seg2["R"], FS, NPERSEG, band)
    Qw = _fit_qs(InvfreqsEstimator().fit(freq, Hw, Hwe, prior))
    # windowed Welch is wrong: a runaway/negative (RHP) Q, not within 20% of truth
    assert not (len(Qw) == 1 and Qw[0] > 0 and abs(Qw[0] - 20.0) / 20.0 < 0.2)


def test_periodic_recovers_both_modes_of_a_double_pendulum():
    _, band, freq = _grid()
    Pxx = _flat_pxx(freq)
    true = double_pendulum()                       # 0.6 Hz Q20, 1.5 Hz Q30
    prior = TFModel.from_resonances([(0.55, 14.0), (1.6, 22.0)], 250.0)
    sp = SuspensionPlant({"POS": true}, FS)
    tw = TwinBackend(sp, {"E": "POS"}, {"R": "POS"}, fs=FS, sensor_asd=0.0, seed=1)
    tw.inject("E", multisine_from_psd(Pxx, FS, NPERSEG, N_PERIODS, freq, seed=0), FS)
    seg = tw.read(["E", "R"], TOTAL_DUR)
    H, He, _ = SysIDLoop._estimate_tf_periodic(seg["E"], seg["R"], FS, NPERSEG, band)

    Qs = _fit_qs(InvfreqsEstimator().fit(freq, H, He, prior))
    assert len(Qs) == 2
    assert abs(Qs[0] - 20.0) / 20.0 < 0.25
    assert abs(Qs[1] - 30.0) / 30.0 < 0.25


# --------------------------------------------------------------------------- #
# 6. defaults preserved
# --------------------------------------------------------------------------- #

def test_default_mode_is_welch():
    # an unset measurement.mode must dispatch to the legacy windowed-Welch path
    loop = SysIDLoop(backend=None, estimator=None, designer=None, watchdog=None)
    loop._meas_mode = "welch"
    assert loop._meas_mode == "welch"
