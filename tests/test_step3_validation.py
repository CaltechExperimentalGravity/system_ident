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

from system_ident.design.pintelon import PintelonSchoukensDesigner, optimal_excitation
from system_ident.fisher import dispersion
from system_ident.plant import double_pendulum


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
    """Matches the legacy reweighting on the power-carrying bins, and floors the rest.

    We deliberately diverge from the legacy engine in one way: every drive bin is
    floored at ``_EXC_FLOOR_FRAC`` of the peak (``pintelon._EXC_FLOOR_FRAC``) so the
    dispersion function can never divide by a starved bin (which otherwise NaNs and
    crashes the design — see the pitch/yaw HSTS DoFs). That floor only moves
    *negligible* bins; on the bins that carry real power the reweighting still
    reproduces the legacy engine to <1e-6, and the Fisher matrices match.
    """
    from system_ident.design.pintelon import _EXC_FLOOR_FRAC

    model, freq, Pyy, Px_tot, T_tot = _demo_setup()
    n_iter = 4

    Pxx_rec, nu_rec, gamma_rec = optimal_excitation(
        freq, model, Pyy, Px_tot, n_iter=n_iter, rec_progress=True
    )
    Pxx_rec0, nu_rec0, gamma_rec0 = oracle.get_opt_exc_Pxx(
        freq, model.to_dict(), Pyy, Px_tot, n_iter=n_iter, T_tot=T_tot,
        rec_progress=True,
    )

    # The Fisher matrices are integrals dominated by the power-carrying bins -> match.
    np.testing.assert_allclose(gamma_rec, gamma_rec0, rtol=1e-5, atol=0)
    # On the bins legacy spends real power, the floored design still reproduces it.
    for it in range(n_iter):
        sig = Pxx_rec0[it] > 1e-6 * Pxx_rec0[it].max()
        np.testing.assert_allclose(Pxx_rec[it][sig], Pxx_rec0[it][sig], rtol=1e-5)
        np.testing.assert_allclose(nu_rec[it][sig], nu_rec0[it][sig], rtol=1e-5)
        # ...and no bin is left starved (the property the floor guarantees).
        assert Pxx_rec[it].min() >= 0.5 * _EXC_FLOOR_FRAC * Pxx_rec[it].max()


def test_excitation_floor_survives_concentrated_design():
    """Regression: a sharp resonance over a wide band makes the optimal drive
    concentrate so hard that off-resonance bins underflow to exactly 0 — the
    dispersion function then divided 0/0 → NaN → ``pinv`` 'SVD did not converge'
    (the pitch/yaw HSTS DoFs). The ``_EXC_FLOOR_FRAC`` floor must keep every pass
    finite and floored."""
    from system_ident.design.pintelon import optimal_excitation, _EXC_FLOOR_FRAC
    from system_ident.model import TFModel

    freq = np.linspace(0.1, 50.0, 2000)             # wide band, one narrow mode
    model = TFModel.from_resonances([(1.0, 200.0)], 1.0)   # very high Q
    Pyy = np.ones_like(freq)

    Pxx_rec, nu_rec, gamma_rec = optimal_excitation(
        freq, model, Pyy, 1.0, n_iter=6, rec_progress=True
    )
    assert np.all(np.isfinite(Pxx_rec)) and np.all(np.isfinite(nu_rec))
    assert np.all(np.isfinite(gamma_rec))
    assert np.all(Pxx_rec > 0)
    for it in range(Pxx_rec.shape[0]):
        assert Pxx_rec[it].min() >= 0.5 * _EXC_FLOOR_FRAC * Pxx_rec[it].max()


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
