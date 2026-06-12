"""Step-2 validation: the ported model / fisher / excitation must reproduce the
legacy ``sysIDlib`` engine on the ``double_pend_demo`` setup.

The legacy engine is loaded as an oracle via the ``oracle`` fixture in
``conftest.py`` (which documents the h5py / np.int / trapz shims).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate as _integ
import scipy.signal as sig

from system_ident.excitation import timeseries_from_asd
from system_ident.fisher import fisher_matrix, parameter_covariance
from system_ident.model import pole_pair_f0_Q, resonance_pole_pair
from system_ident.plant import double_pendulum


# --- shared double_pend_demo setup ---------------------------------------------
def _demo_setup():
    model = double_pendulum()
    fs, T_perseg, T_tot = 32.0, 128.0, 256.0
    freq = np.arange(0, fs / 2 + 1 / T_perseg, 1 / T_perseg)
    freq = freq[(freq < 5) & (freq > 0.02)]

    Ayy = np.zeros(len(freq))
    Ayy[freq < 1] = 1e-7 * freq[freq < 1] ** (-1)
    band = (freq >= 1) & (freq < 10)
    Ayy[band] = 1e-7 * freq[band] ** (-6)
    Ayy[freq >= 10] = 1e-12
    Pyy = Ayy**2

    Pxx = np.ones(len(freq))
    Pxx *= 1e-16 / _integ.trapezoid(Pxx, freq)
    return model, freq, Pxx, Pyy, T_tot


# --- model construction --------------------------------------------------------
def test_resonance_pole_pair_matches_oracle(oracle):
    for f0, Q in [(0.6, 20.0), (1.5, 30.0)]:
        a, b = resonance_pole_pair(f0, Q)
        a0, b0 = oracle.get_res_g_pole_pair(f0, Q)
        assert a == pytest.approx(a0)
        assert b == pytest.approx(b0)
        # round-trip back to (f0, Q)
        f0_rt, Q_rt = pole_pair_f0_Q(a, b)
        assert f0_rt == pytest.approx(f0)
        assert Q_rt == pytest.approx(Q)


def test_from_resonances_reproduces_demo_polynomials():
    model = double_pendulum()
    ps_r1, ps_i1 = resonance_pole_pair(0.6, 20)
    ps_r2, ps_i2 = resonance_pole_pair(1.5, 30)
    ps = np.array(
        [ps_r1 + 1j * ps_i1, ps_r1 - 1j * ps_i1, ps_r2 + 1j * ps_i2, ps_r2 - 1j * ps_i2]
    )
    num, den = sig.zpk2tf(np.array([]), ps, 300)
    np.testing.assert_allclose(model.num, num)
    np.testing.assert_allclose(model.den, den)


# --- transfer function evaluation ----------------------------------------------
def test_eval_matches_oracle(oracle):
    model, freq, *_ = _demo_setup()
    H = model.eval(freq)
    H0 = oracle.par_dict_to_TF_vect(freq, model.to_dict())
    np.testing.assert_allclose(H, H0, rtol=1e-12, atol=0)


# --- Fisher information --------------------------------------------------------
def test_fisher_matrix_matches_oracle(oracle):
    model, freq, Pxx, Pyy, T_tot = _demo_setup()
    gamma = fisher_matrix(freq, model, Pxx, Pyy, T_tot)
    gamma0 = oracle.get_Fisher_from_psd(
        freq, model.to_dict(), Pxx, Pyy, T_tot=T_tot
    )
    np.testing.assert_allclose(gamma, gamma0, rtol=1e-9, atol=0)


def test_fisher_matrix_is_symmetric_and_invertible():
    model, freq, Pxx, Pyy, T_tot = _demo_setup()
    gamma = fisher_matrix(freq, model, Pxx, Pyy, T_tot)
    np.testing.assert_allclose(gamma, gamma.T, rtol=1e-12)
    cov = parameter_covariance(freq, model, Pxx, Pyy, T_tot)
    n = len(model.num) + len(model.den) - 1
    np.testing.assert_allclose(gamma @ cov, np.eye(n), atol=1e-6)


# --- excitation time series ----------------------------------------------------
def test_timeseries_from_asd_matches_oracle(oracle):
    _, freq, *_ = _demo_setup()
    asd = 1e-9 * np.ones(len(freq))
    duration, fs = 16.0, 32.0
    ts = timeseries_from_asd(duration, fs, freq, asd, seed=1234)
    ts0 = oracle.time_series_from_asd_vect(duration, fs, freq, asd, seed=1234)
    np.testing.assert_allclose(ts, ts0, rtol=1e-12, atol=0)
    assert ts.shape == (int(duration * fs),)
