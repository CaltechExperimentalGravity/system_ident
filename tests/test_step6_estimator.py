"""Step-6: the invfreqs weighted-LS estimator.

Validated against the legacy ``update_par_dict_from_data`` oracle, plus a
recovery check (fitting near-noiseless data returns the true model) and an
order-preservation check.
"""

from __future__ import annotations

import numpy as np
import pytest

from ligo_sysid.estimators.invfreqs import InvfreqsEstimator, invfreqs
from ligo_sysid.plant import double_pendulum


def _demo_freq():
    fs, T_perseg = 32.0, 128.0
    freq = np.arange(0, fs / 2 + 1 / T_perseg, 1 / T_perseg)
    return freq[(freq < 5) & (freq > 0.02)]


def test_invfreqs_matches_oracle(oracle):
    freq = _demo_freq()
    w = 2 * np.pi * freq
    H = double_pendulum().eval(freq)
    wt = np.ones(len(freq))
    nb = len(double_pendulum().den) - 1

    num, den = invfreqs(w, H, wt, nb)
    num0, den0 = oracle.invfreqs(w, H, wt, nb)
    np.testing.assert_allclose(num, num0, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(den, den0, rtol=1e-9, atol=1e-12)


def test_fit_matches_oracle(oracle):
    freq = _demo_freq()
    model = double_pendulum()
    H = model.eval(freq)
    H_err = 0.01 * np.abs(H)  # 1% per-point error -> SNR 100

    fitted = InvfreqsEstimator().fit(freq, H, H_err, model)
    par0 = oracle.update_par_dict_from_data(freq, H, H_err, model.to_dict())

    np.testing.assert_allclose(fitted.num, par0["num"], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(fitted.den, par0["den"], rtol=1e-7, atol=1e-10)


def test_fit_recovers_true_model_from_clean_data():
    freq = _demo_freq()
    true = double_pendulum()
    H = true.eval(freq)
    H_err = 1e-6 * np.abs(H)

    fitted = InvfreqsEstimator().fit(freq, H, H_err, true)
    # compare the recovered response, which is gauge-invariant
    np.testing.assert_allclose(fitted.eval(freq), H, rtol=1e-4)


def test_fit_preserves_model_order():
    freq = _demo_freq()
    model = double_pendulum()
    H = model.eval(freq)
    fitted = InvfreqsEstimator().fit(freq, H, 0.01 * np.abs(H), model)
    assert fitted.num.shape == model.num.shape
    assert fitted.den.shape == model.den.shape
