from __future__ import annotations
import numpy as np, control, pytest
from system_ident.mimo_plant import mimo_suspension, input_matrix, output_matrix


def test_plant_shapes_square_and_rectangular():
    Gsq = mimo_suspension([(0.6, 20), (1.5, 30)], n_sens=2, n_act=2, coupling=0.25)
    assert Gsq.noutputs == 2 and Gsq.ninputs == 2
    Grect = mimo_suspension([(0.6, 20), (1.5, 30)], n_sens=3, n_act=2, coupling=0.25)
    assert Grect.noutputs == 3 and Grect.ninputs == 2


def test_plant_is_coupled_with_shared_poles():
    G = mimo_suspension([(0.6, 20), (1.5, 30)], n_sens=2, n_act=2, coupling=0.3)
    f = np.geomspace(0.3, 8, 200)
    H = G(2j * np.pi * f)  # (2,2,200)
    # off-diagonal is non-trivial (genuine coupling)
    assert np.max(np.abs(H[0, 1])) / np.max(np.abs(H[0, 0])) > 0.05
    # all elements share the same poles (same denominator roots) -> resonances line up
    pk00 = f[np.argmax(np.abs(H[0, 0]))]
    pk11 = f[np.argmax(np.abs(H[1, 1]))]
    assert min(abs(pk00 - 0.6), abs(pk00 - 1.5)) < 0.1 and min(abs(pk11 - 0.6), abs(pk11 - 1.5)) < 0.1


def test_decoupling_matrix_shapes():
    Min = input_matrix(2, 3, kind="perturbed", seed=1)
    assert Min.shape == (2, 3)
    G = mimo_suspension([(0.6, 20), (1.5, 30)], n_sens=2, n_act=2)
    Mo = output_matrix(G, n_act=2, n_dof=2, basis="eigenmode")
    assert Mo.ninputs == 2 and Mo.noutputs == 2


# ---------------------------------------------------------------------------
# Task 2: CoupledLoop
# ---------------------------------------------------------------------------
from system_ident.mimo_loop import CoupledLoop, velocity_damper


def _square_loop(fs=64.0, basis="euler"):
    G = mimo_suspension([(0.6, 20), (1.5, 30)], n_sens=2, n_act=2, coupling=0.25)
    C = [velocity_damper(0.5, 20.0) for _ in range(2)]
    Min = input_matrix(2, 2, kind="identity")
    Mout = output_matrix(G, n_act=2, n_dof=2, basis=basis)
    return CoupledLoop(G, C, Min, Mout, fs=fs)


def test_loop_is_stable():
    assert _square_loop().is_stable()


def test_loop_oracle_is_discrete_plant():
    lp = _square_loop()
    f = np.array([0.4, 3.0])
    z = np.exp(2j * np.pi * f / lp.fs)
    expect = lp.Gd(z)  # (2,2,2)
    np.testing.assert_allclose(lp.oracle(f), expect, rtol=1e-9)


def test_sensitivity_identity():
    # S = (I+L)^-1  =>  shape must be n_act×n_act in discrete form
    lp = _square_loop()
    assert lp.Sd.ninputs == 2 and lp.Sd.noutputs == 2
