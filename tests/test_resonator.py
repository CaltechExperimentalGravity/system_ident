"""ResonatorModel — physical (f0, Q, gain) parameterisation.

Validates that the physical model's response matches its expanded TFModel form,
that the parameter vector round-trips, and that the Jacobian is finite and
correctly shaped (the numeric surface the Fisher / Bayesian machinery needs).
"""

import numpy as np
import pytest

from system_ident.model import TFModel
from system_ident.resonator import ResonatorModel, resonator_from_tf


def test_params_roundtrip():
    m = ResonatorModel(f0=[0.6, 1.5], Q=[20.0, 30.0], gain=300.0)
    np.testing.assert_allclose(m.params, [0.6, 1.5, 20.0, 30.0, 300.0])
    m2 = m.with_params(m.params)
    np.testing.assert_allclose(m2.f0, m.f0)
    np.testing.assert_allclose(m2.Q, m.Q)
    assert m2.gain == m.gain


def test_eval_matches_expanded_tf():
    """The physical response equals its expanded num/den TFModel response."""
    m = ResonatorModel(f0=[0.6, 1.5], Q=[20.0, 30.0], gain=300.0)
    tf = m.to_tf()
    freq = np.logspace(-1, 1, 400)
    np.testing.assert_allclose(m.eval(freq), tf.eval(freq), rtol=1e-9, atol=1e-12)


def test_dc_gain():
    # num = gain (constant), so the DC value is gain / prod(w_i^2)
    m = ResonatorModel(f0=[1.0], Q=[20.0], gain=42.0)
    expected = 42.0 / (2 * np.pi * 1.0) ** 2
    assert abs(m.eval(np.array([1e-6]))[0] - expected) < 1e-3 * abs(expected)


def test_jacobian_shape_and_finite():
    m = ResonatorModel(f0=[1.0], Q=[20.0], gain=100.0)
    freq = np.linspace(0.2, 3.0, 50)
    J = m.jacobian(freq)
    assert J.shape == (3, 50)  # [f0, Q, gain] x n_bin
    assert np.all(np.isfinite(J))
    # finite-difference dH/dgain equals H/gain (H is linear in gain)
    np.testing.assert_allclose(J[2], m.eval(freq) / m.gain, rtol=1e-5)


def test_from_resonances_matches_tfmodel_constructor():
    m = ResonatorModel.from_resonances([(0.6, 20.0), (1.5, 30.0)], gain=300.0)
    tf = TFModel.from_resonances([(0.6, 20.0), (1.5, 30.0)], 300.0)
    freq = np.logspace(-1, 1, 200)
    np.testing.assert_allclose(m.eval(freq), tf.eval(freq), rtol=1e-9)


def test_resonator_from_tf_preserves_magnitude():
    """The conversion recovers f0, Q, and a gain that reproduces |H|."""
    tf = TFModel.from_resonances([(0.99, 19.4)], 97.0)
    rm = resonator_from_tf(tf)
    assert abs(float(rm.f0[0]) - 0.99) < 1e-3
    assert abs(float(rm.Q[0]) - 19.4) / 19.4 < 0.02
    freq = np.linspace(0.2, 3.0, 200)
    np.testing.assert_allclose(np.abs(rm.eval(freq)), np.abs(tf.eval(freq)), rtol=1e-6)


def test_resonator_from_tf_unstable_den_gain_does_not_collapse():
    """Regression: a marginally-UNSTABLE lock (RHP poles, negative middle den
    coefficient — as a noisy invfreqs fit can produce) must still yield a gain
    that reproduces |H|. A complex-LS gain cancels here and collapses to ~0,
    starting the hybrid refine from a degenerate (gain~0) model.
    """
    tf = TFModel(num=np.array([97.0731]), den=np.array([1.0, -0.3212, 38.7132]))
    rm = resonator_from_tf(tf)
    assert float(rm.gain) > 1.0, f"gain collapsed to {float(rm.gain):.3f}"
    freq = np.linspace(0.2, 3.0, 200)
    np.testing.assert_allclose(np.abs(rm.eval(freq)), np.abs(tf.eval(freq)), rtol=1e-3)
