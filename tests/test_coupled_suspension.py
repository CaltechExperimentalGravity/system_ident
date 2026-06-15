"""Coupled longitudinal/pitch suspension model (plant.coupled_suspension)."""
import numpy as np

from system_ident.model import TFModel
from system_ident.plant import coupled_suspension


LONG = [(0.43, 20.0), (1.00, 30.0)]
PITCH = [(0.56, 18.0), (1.31, 28.0)]


def _modes(model):
    p = np.roots(np.asarray(model.den, float))
    p = p[p.imag > 1e-9]
    return sorted(abs(z) / (2 * np.pi) for z in p)


def test_returns_four_elements():
    H = coupled_suspension(LONG, PITCH)
    assert set(H) == {("POS", "POS"), ("PIT", "PIT"), ("PIT", "POS"), ("POS", "PIT")}
    assert all(isinstance(v, TFModel) for v in H.values())


def test_all_elements_share_the_normal_mode_poles():
    H = coupled_suspension(LONG, PITCH)
    ref = _modes(H[("POS", "POS")])
    assert len(ref) == 4
    # the 4 normal-mode frequencies (~0.43, 0.56, 1.00, 1.31) appear in every element
    for key in H:
        np.testing.assert_allclose(_modes(H[key]), ref, rtol=1e-6)
    np.testing.assert_allclose(ref, sorted([0.43, 0.56, 1.00, 1.31]), rtol=0.05)


def test_diagonal_has_an_antiresonance_notch():
    """A diagonal element dips into a notch between the two same-DoF modes."""
    H = coupled_suspension(LONG, PITCH, coupling=0.2)
    f = np.linspace(0.3, 1.4, 4000)
    mag = np.abs(H[("POS", "POS")].eval(f))
    # a real notch (local minimum well below the neighbouring peaks)
    i = np.argmin(mag)
    assert 0 < i < len(f) - 1
    assert mag[i] < 0.3 * max(mag[: i + 1].max(), mag[i:].max())


def test_offdiagonal_is_not_a_scaled_diagonal():
    """Cross term has its own shape — not a rescaled copy of the diagonal."""
    H = coupled_suspension(LONG, PITCH, coupling=0.2)
    f = np.logspace(np.log10(0.2), np.log10(3), 2000)
    diag = np.abs(H[("POS", "POS")].eval(f))
    cross = np.abs(H[("PIT", "POS")].eval(f))
    ratio = cross / diag
    # if it were a scaled diagonal the ratio would be ~constant; it is not
    assert ratio.std() / ratio.mean() > 0.3


def test_reciprocity():
    H = coupled_suspension(LONG, PITCH)
    f = np.logspace(np.log10(0.2), np.log10(3), 500)
    np.testing.assert_allclose(H[("PIT", "POS")].eval(f), H[("POS", "PIT")].eval(f), rtol=1e-9)


def test_coupling_zero_is_block_diagonal():
    H = coupled_suspension(LONG, PITCH, coupling=0.0)
    f = np.logspace(np.log10(0.2), np.log10(3), 500)
    assert np.max(np.abs(H[("PIT", "POS")].eval(f))) < 1e-9
