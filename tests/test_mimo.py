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
