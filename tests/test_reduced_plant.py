import numpy as np
import pytest
from system_ident.reduced_plant import ReducedStateSpacePlant


def test_load_shapes_and_labels():
    p = ReducedStateSpacePlant.load("hsts")
    assert p.A.shape == (36, 36)
    assert len(p.inputs) == p.B.shape[1] == 24
    assert len(p.outputs) == p.C.shape[0] == 18
    assert "m1.drive.L" in p.inputs and "m1.disp.L" in p.outputs


def test_modes_are_the_srm_physics():
    p = ReducedStateSpacePlant.load("hsts")
    f0 = sorted(f for f, q in p.modes())
    # the SRM spatial doublet + a triple, all present
    assert any(abs(f - 0.672) < 1e-3 for f in f0)
    assert any(abs(f - 0.676) < 1e-3 for f in f0)
    assert sum(1 for f in f0 if 1.50 < f < 1.53) == 3
    assert all(abs(q - 50.0) < 1e-6 for f, q in p.modes())


def test_eval_frf_shape_and_finite():
    p = ReducedStateSpacePlant.load("hsts")
    freq = np.linspace(0.3, 5.0, 200)
    G = p.eval(freq)
    assert G.shape == (200, 18, 24)
    assert np.all(np.isfinite(G))


def test_frf_peaks_at_a_mode():
    p = ReducedStateSpacePlant.load("hsts")
    i, j = p.outputs.index("m1.disp.L"), p.inputs.index("m1.drive.L")
    freq = np.linspace(0.6, 0.75, 4000)
    mag = np.abs(p.eval(freq)[:, i, j])
    f_peak = freq[np.argmax(mag)]
    assert abs(f_peak - 0.672) < 0.01  # peaks at the fundamental


def test_subplant_selects_block():
    p = ReducedStateSpacePlant.load("hsts")
    dofs_in = ["m1.drive.L", "m1.drive.P"]
    dofs_out = ["m1.disp.L", "m1.disp.P"]
    sub = p.subplant(sensors=dofs_out, actuators=dofs_in)
    assert sub.inputs == dofs_in and sub.outputs == dofs_out
    freq = np.linspace(0.3, 3.0, 50)
    # subplant FRF equals the selected rows/cols of the full FRF
    full = p.eval(freq)
    fi = [p.inputs.index(x) for x in dofs_in]
    fo = [p.outputs.index(x) for x in dofs_out]
    assert np.allclose(sub.eval(freq), full[:, fo][:, :, fi])
