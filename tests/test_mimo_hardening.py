"""A5 hardening: recover_open_loop conditioning guard + assemble_campaign resolution guard."""
import numpy as np
import pytest

from system_ident.mimo_loop import recover_open_loop
from system_ident.mimo_campaign import assemble_campaign


def test_recover_open_loop_exact_when_well_conditioned():
    # pinv == inv for well-conditioned X: recovery is exact (regression on the vectorization)
    rng = np.random.default_rng(0)
    G = rng.standard_normal((5, 3, 3)) + 1j * rng.standard_normal((5, 3, 3))
    X = rng.standard_normal((5, 3, 3)) + 1j * rng.standard_normal((5, 3, 3))
    Y = np.einsum('fij,fjk->fik', G, X)
    np.testing.assert_allclose(recover_open_loop(X, Y), G, atol=1e-9)


def test_recover_open_loop_warns_on_ill_conditioned_bin():
    X = np.tile(np.eye(3, dtype=complex), (3, 1, 1))
    X[1, 2, 2] = 1e-14                                  # one near-singular drive matrix
    Y = np.zeros((3, 3, 3), complex)
    with pytest.warns(RuntimeWarning, match="cond"):
        recover_open_loop(X, Y, cond_warn=1e6)
    # and it does NOT blow up (pinv-guarded) — the recovery stays finite
    assert np.all(np.isfinite(recover_open_loop(X, Y, cond_warn=1e18)))


def test_assemble_campaign_rejects_colliding_lines():
    # two lines 0.05 Hz apart with df = 0.25 Hz collapse onto one bin -> ValueError (raised
    # before any backend use, so a dummy backend is fine)
    with pytest.raises(ValueError, match="collapse|distinct"):
        assemble_campaign(None, ["e"], ["d"], ["s"], [1.0, 1.05],
                          fs=256.0, nperseg=1024, n_periods=4, drive_psd=np.ones(1))
