"""Step-3 validation: the dispersion-function excitation design must reproduce
the legacy ``sysIDlib`` engine end-to-end on the ``double_pend_demo`` setup.

Like step 2 we load the legacy module as an oracle (see
``test_step2_validation`` for the shim rationale) and compare iteration-by-
iteration. Both sides use the engine's default finite-difference / log settings
(``dpar_dict=None``, ``logflag_dict=None``): a well-defined path that exercises
the same math, rather than the notebook's literal arguments, which trip a latent
``logflag``-handling bug in the legacy ``get_Fisher_from_psd``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ligo_sysid.design.pintelon import PintelonSchoukensDesigner, optimal_excitation
from ligo_sysid.fisher import dispersion
from ligo_sysid.plant import double_pendulum


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

    Px_tot = 1e-16
    return model, freq, Pyy, Px_tot, T_tot


def test_dispersion_matches_oracle(oracle):
    model, freq, Pyy, Px_tot, _ = _demo_setup()
    # flat starting excitation at the budget
    from scipy.integrate import trapezoid

    Pxx = np.ones(len(freq))
    Pxx *= Px_tot / trapezoid(Pxx, freq)

    nu, gamma = dispersion(freq, model, Pxx, Pyy)
    nu0, gamma0 = oracle.get_dispersion(
        freq, model.to_dict(), Pxx, Pyy, return_gamma=True
    )
    np.testing.assert_allclose(nu, nu0, rtol=1e-8, atol=0)
    np.testing.assert_allclose(gamma, gamma0, rtol=1e-8, atol=0)


def test_optimal_excitation_matches_oracle(oracle):
    model, freq, Pyy, Px_tot, T_tot = _demo_setup()
    n_iter = 4

    Pxx_rec, nu_rec, gamma_rec = optimal_excitation(
        freq, model, Pyy, Px_tot, n_iter=n_iter, rec_progress=True
    )
    Pxx_rec0, nu_rec0, gamma_rec0 = oracle.get_opt_exc_Pxx(
        freq, model.to_dict(), Pyy, Px_tot, n_iter=n_iter, T_tot=T_tot,
        rec_progress=True,
    )

    np.testing.assert_allclose(Pxx_rec, Pxx_rec0, rtol=1e-7, atol=0)
    np.testing.assert_allclose(nu_rec, nu_rec0, rtol=1e-7, atol=0)
    np.testing.assert_allclose(gamma_rec, gamma_rec0, rtol=1e-7, atol=0)


def test_designer_returns_budgeted_psd():
    from scipy.integrate import trapezoid

    model, freq, Pyy, Px_tot, _ = _demo_setup()
    Pxx = PintelonSchoukensDesigner().design(freq, model, Pyy, Px_tot, n_iter=3)

    assert Pxx.shape == freq.shape
    assert np.all(Pxx > 0)
    # power budget is respected
    assert trapezoid(Pxx, freq) == pytest.approx(Px_tot, rel=1e-10)
    # optimisation concentrates drive near the resonances rather than staying flat
    assert Pxx.max() / Pxx.min() > 10
