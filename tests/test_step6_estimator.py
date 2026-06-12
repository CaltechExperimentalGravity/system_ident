"""Step-6: the invfreqs weighted-LS estimator.

Validated against the legacy ``update_par_dict_from_data`` oracle, plus a
recovery check (fitting near-noiseless data returns the true model) and an
order-preservation check.
"""

from __future__ import annotations

import numpy as np
import pytest

from system_ident.estimators.invfreqs import InvfreqsEstimator, invfreqs
from system_ident.model import TFModel, resonance_pole_pair
from system_ident.plant import double_pendulum


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


def test_invfreqs_odd_nb_is_finite():
    """Regression: odd nb (e.g. a first-order system) must not return NaN.

    The original column-assembly loop only filled even denominator orders, so
    odd nb left zero singular values and produced NaN. A first-order cavity
    G(s) = wc/(s+wc) has nb=1.
    """
    wc = 2 * np.pi * 100.0
    freq = np.logspace(0, 3, 400)
    w = 2 * np.pi * freq
    H = wc / (1j * w + wc)

    num, den = invfreqs(w, H, np.ones_like(freq), 1)
    assert np.all(np.isfinite(num)) and np.all(np.isfinite(den))
    # the single pole is recovered at wc
    np.testing.assert_allclose(np.sort(np.abs(np.roots(den))), [wc], rtol=1e-3)


@pytest.mark.parametrize(
    "true, freq",
    [
        # first-order (nb=1): a cavity pole at 100 Hz, DC gain 1
        (
            TFModel.from_zpk([], [-2 * np.pi * 100.0], 2 * np.pi * 100.0),
            np.logspace(0, 3, 400),
        ),
        # third-order (nb=3): 1 Hz Q=20 resonance in series with a 20 Hz pole
        (
            TFModel.from_zpk(
                [],
                [complex(*resonance_pole_pair(1.0, 20.0)),
                 complex(resonance_pole_pair(1.0, 20.0)[0],
                         -resonance_pole_pair(1.0, 20.0)[1]),
                 -2 * np.pi * 20.0],
                1.0e4,
            ),
            np.logspace(-1, np.log10(50.0), 500),
        ),
    ],
)
def test_fit_recovers_odd_order_models(true, freq):
    """Recovery for odd-order systems (the closed-loop arm / cavity examples)."""
    H = true.eval(freq)
    H_err = 1e-6 * np.abs(H)
    fitted = InvfreqsEstimator().fit(freq, H, H_err, true)
    assert np.all(np.isfinite(fitted.den))
    np.testing.assert_allclose(fitted.eval(freq), H, rtol=1e-3)
