"""Pintelon-Schoukens periodic multisine + leakage-free FRF — the only path.

These tests pin the *measurement*:

  * the realiser is a genuine periodic multisine with a low (Schroeder) crest
    factor and the designed in-band power;
  * a synchronous integer-period DFT measures a sharp resonance leakage-free;
  * the reference-based (ratio-of-averages) estimator recovers the *open-loop*
    plant even with a damping loop closed; and
  * fed into the parametric estimator it recovers Q on single- and two-mode
    plants.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sig
from scipy.integrate import trapezoid

from system_ident.backends.rtsfreerun_adapter import RTSfreerunBackend
from system_ident.backends.twin import TwinBackend
from system_ident.estimators.gml import GMLEstimator
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

def test_periodic_frf_is_leakage_free():
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
    assert rel.max() < 1e-2          # leakage-free: matches the realised FRF


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
    assert rel.max() < 1e-2          # recovers the OPEN-LOOP plant in closed loop


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

def test_periodic_recovers_Q():
    _, band, freq = _grid()
    Pxx = _flat_pxx(freq)
    true = TFModel.from_resonances([(0.6, 20.0)], 300.0)
    prior = TFModel.from_resonances([(0.55, 12.0)], 250.0)   # offset, same order
    sp = SuspensionPlant({"POS": true}, FS)
    tw = TwinBackend(sp, {"E": "POS"}, {"R": "POS"}, fs=FS, sensor_asd=0.0, seed=1)

    tw.inject("E", multisine_from_psd(Pxx, FS, NPERSEG, N_PERIODS, freq, seed=0), FS)
    seg = tw.read(["E", "R"], TOTAL_DUR)
    H, He, _ = SysIDLoop._estimate_tf_periodic(seg["E"], seg["R"], FS, NPERSEG, band)
    Qp = _fit_qs(GMLEstimator().fit(freq, H, He, prior))
    assert len(Qp) == 1 and abs(Qp[0] - 20.0) / 20.0 < 0.2   # ~Q=20


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

    Qs = _fit_qs(GMLEstimator().fit(freq, H, He, prior))
    assert len(Qs) == 2
    assert abs(Qs[0] - 20.0) / 20.0 < 0.25


# --------------------------------------------------------------------------- #
# 6. Stage B hardening (issues #6-#10) -- each measured signature, pinned
# --------------------------------------------------------------------------- #

def test_p_eff_one_gives_infinite_error_never_a_near_zero_one():
    """#6: a single surviving period after the transient drop must exclude the
    pass with INFINITE H_err (zero weight in _accumulate) -- never a finite-but-
    tiny one. The pre-fix code zeroed ``var_H`` at P_eff==1, which the ``1e-9 *
    |Hb|`` floor then turned into a measured weight of 4.56e+19: one bad pass
    permanently swamping every other pass for the rest of a campaign."""
    _, band, freq = _grid(nperseg=256)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256 * 2)          # P=2 whole periods
    y = rng.standard_normal(256 * 2)
    # n_transient=1 on P=2 hits _choose_transient's early-return branch
    # (P <= n_min+2) -> n_drop=1 -> P_eff=1.
    H, H_err, coh = SysIDLoop._estimate_tf_periodic(x, y, FS, 256, band, n_transient=1)
    assert np.all(np.isinf(H_err))
    w = np.where(np.isfinite(H_err) & (H_err > 0), 1.0 / H_err ** 2, 0.0)
    assert np.all(w == 0.0)                   # never the 4.56e+19-style near-inf weight
    assert np.all(coh <= 1e-6 + 1e-12)         # excluded, not reported as confident


def test_longest_contiguous_run_not_the_span():
    """#7: energies with an INTERIOR dip (the looped-taper geometry) must keep
    only the longest contiguous full-energy block, not the span between the
    first and last full-energy index -- which for this exact measured pattern
    is the whole array."""
    e = np.array([1.0, 1.0, 1.0, 1.0, 0.571, 0.492, 1.0, 1.0])
    full = np.flatnonzero(e >= 0.999 * e.max())
    assert list(full) == [0, 1, 2, 3, 6, 7]                # span would be 0:8
    start, stop = SysIDLoop._longest_contiguous_run(full)
    assert (start, stop) != (0, 8)
    assert (start, stop) == (0, 4)                          # the longest run: periods 0-3


def test_longest_contiguous_run_ties_break_late():
    start, stop = SysIDLoop._longest_contiguous_run(np.array([0, 1, 5, 6]))
    assert (start, stop) == (5, 7)                          # two equal-length runs -> later wins


def test_looped_tukey_taper_corrupts_frf_untapered_transport_ramp_does_not():
    """#9: base.py's ramp contract. Tapering the WHOLE staged array (the array
    an ``ArbitraryLoop``-style transport then repeats forever) rather than
    applying the taper to a one-shot lead+record+tail array, or leaving the
    staged array untapered and ramping via the transport's own gain instead,
    turns the taper into a periodic amplitude modulation at the LOOP period
    (here, the full 8-period staged array) rather than a one-shot envelope --
    re-exciting the plant's transient every cycle. Reproduces spec S2.2's own
    measurement setup (ramp_s=3.0s over a 32s/8-period array) and its
    contrast: looped+tapered ~2.8e-1 max relative FRF error, vs
    untapered+ramped ~1e-11."""
    fs, nperseg, n_periods, ramp_s = 256.0, 1024, 8, 3.0
    f_all = np.fft.rfftfreq(nperseg, d=1 / fs)
    band = (f_all >= 0.3) & (f_all <= 8.0)
    freq = f_all[band]
    x1 = multisine_from_psd(np.ones_like(freq), fs, nperseg, 1, freq, seed=0)
    staged_untapered = np.resize(x1, nperseg * n_periods)   # the full staged array

    plant = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    b, a = sig.bilinear(plant.num, plant.den, fs)
    _, H_true = sig.freqz(b, a, worN=freq, fs=fs)

    def _loop_forever_and_read(staged, n_settle_cycles=60):
        """An ArbitraryLoop-style transport: repeat `staged` indefinitely,
        settle, then read back exactly one repeat's worth."""
        lead = np.resize(staged, staged.size * n_settle_cycles)
        _, zi = sig.lfilter(b, a, lead, zi=np.zeros(max(len(a), len(b)) - 1))
        y, _ = sig.lfilter(b, a, staged, zi=zi)
        return staged, y

    def _relerr(x, y):
        H, _, _ = SysIDLoop._estimate_tf_periodic(x, y, fs, nperseg, band, n_transient=1)
        return np.max(np.abs(H - H_true) / np.abs(H_true))

    # BAD: the base.py default alpha (ramp_s Tukey taper over the whole staged
    # array), then looped -- exactly the historical (pre-#9) contract.
    alpha = min(1.0, 2.0 * ramp_s * fs / staged_untapered.size)
    staged_tapered = staged_untapered * sig.windows.tukey(staged_untapered.size, alpha)
    bad_relerr = _relerr(*_loop_forever_and_read(staged_tapered))

    # GOOD: untapered integer-period tiling -- the transport applies its own
    # gain ramp OUTSIDE this array instead (S2.2/S2.3's "never both" branches).
    good_relerr = _relerr(*_loop_forever_and_read(staged_untapered))

    assert bad_relerr > 0.2                      # matches the measured 2.81e-1 signature
    assert good_relerr < 1e-9                     # matches the measured 1.4e-11 signature
    assert good_relerr < bad_relerr / 1000


def test_rtsfreerun_inject_resample_matches_interior_period_to_high_precision():
    """#10: resample_poly on the FULL tiled array sees zero-padded edges and
    corrupts precisely the periods nearest each end (measured: 6.9% on the
    first period, 53% on the last, at the shipped 256/16384 ratio). Assert the
    fix directly against RTSfreerunBackend.inject() -- not a standalone
    reimplementation -- at that same ratio."""
    fs_sysid, fs_model, nperseg, n_periods = 256.0, 16384.0, 1024, 8
    f_all = np.fft.rfftfreq(nperseg, 1 / fs_sysid)
    band = (f_all >= 0.3) & (f_all <= 8.0)
    freq = f_all[band]
    x1 = multisine_from_psd(np.ones_like(freq), fs_sysid, nperseg, 1, freq, seed=0)
    tiled = np.resize(x1, nperseg * n_periods)

    class _NullModel:
        sample_rate = fs_model

    be = RTSfreerunBackend(mdl=_NullModel(), exc_channels={"E": "POS"},
                           readback_channels={}, fs=fs_sysid)
    be.inject("E", tiled, fs_sysid)
    resampled = be._drives["E"]

    nperseg_model = int(round(nperseg * fs_model / fs_sysid))
    periods = resampled.reshape(n_periods, nperseg_model)
    interior = periods[1:-1].mean(axis=0)
    dev = np.max(np.abs(periods - interior[None, :]), axis=1) / np.max(np.abs(interior))
    assert np.all(dev < 1e-9)                     # was 6.9e-2 (first) / 5.3e-1 (last)
