"""Oracle utility unit tests — analytic plant from a scenario, no twin needed.

These exercise :mod:`system_ident.backends.rtsfreerun_oracle` on a synthetic
scenario dict, so they run on a plain SciPy stack (the *realized-SOS* cross-check
that needs a built model lives in ``test_rtsfreerun_real_model.py``).
"""
import numpy as np
import pytest

from system_ident.backends import rtsfreerun_oracle as orc


def _scenario(modules=("PLANT_A",)):
    """A two-mode plant: poles at ~2 Hz (Q 20) and ~5 Hz (Q 50), one zero pair."""
    return {"init": [
        {"fm": "PLANT_A", "section": "FM1", "zpk": {
            "z": [], "k": 1.0, "plane": "f",
            "p": [{"re": 0.05, "im": 2.0}, {"re": 0.05, "im": -2.0}]}},
        {"fm": "PLANT_A", "section": "FM2", "zpk": {
            "k": 1.0, "plane": "f",
            "z": [{"re": 0.10, "im": 3.0}, {"re": 0.10, "im": -3.0}],
            "p": [{"re": 0.05, "im": 5.0}, {"re": 0.05, "im": -5.0}]}},
        {"fm": "PLANT_A", "section": "FM1",
         "switches": {"INPUT": True, "OUTPUT": True, "FM1": True, "FM2": True}},
    ]}


def test_coerce_zpk_plane_f_negates_and_scales():
    z, p, k = orc._coerce_zpk({"z": [], "p": [{"re": 0.05, "im": 2.0}], "k": 3.0, "plane": "f"})
    assert k == 3.0 and z == []
    assert np.isclose(p[0], -2 * np.pi * (0.05 + 2.0j))     # plane 'f' -> s, LHP


def test_coerce_zpk_plane_s_passthrough():
    z, p, k = orc._coerce_zpk({"z": [], "p": [-0.3 - 12.0j], "k": 1.0, "plane": "s"})
    assert np.isclose(p[0], -0.3 - 12.0j)


def test_analytic_plant_recovers_modes():
    tf = orc.analytic_plant(_scenario(), modules=("PLANT_A",))
    modes = orc.plant_modes(tf)
    assert len(modes) == 2
    (f0, q0), (f1, q1) = modes
    assert np.isclose(f0, 2.0, rtol=0.02) and np.isclose(q0, 20.0, rtol=0.05)
    assert np.isclose(f1, 5.0, rtol=0.02) and np.isclose(q1, 50.0, rtol=0.05)
    # real coefficients (conjugate-symmetric roots), order 4 poles / 2 zeros
    assert np.isrealobj(tf.den) and np.isrealobj(tf.num)
    assert tf.den.size == 5 and tf.num.size == 3


def test_prior_from_scenario_perturb_zero_is_truth():
    scen = _scenario()
    truth = orc.analytic_plant(scen, modules=("PLANT_A",))
    prior = orc.prior_from_scenario(scen, modules=("PLANT_A",), perturb=0.0)
    np.testing.assert_allclose(prior.eval(np.linspace(0.5, 8, 50)),
                               truth.eval(np.linspace(0.5, 8, 50)), rtol=1e-9)


def test_prior_from_scenario_perturb_keeps_real_coeffs():
    prior = orc.prior_from_scenario(_scenario(), modules=("PLANT_A",), perturb=0.15,
                                    rng=np.random.default_rng(0))
    # conjugate symmetry preserved -> coefficients stay real, modes still ~2
    assert np.isrealobj(prior.num) and np.isrealobj(prior.den)
    modes = orc.plant_modes(prior)
    assert len(modes) == 2 and 1.0 < modes[0][0] < 3.0


def test_analytic_plant_ignores_other_modules():
    """Only the named drive modules contribute to the plant."""
    scen = _scenario()
    scen["init"].append({"fm": "OTHER", "section": "FM1", "zpk": {
        "z": [], "p": [{"re": 0.01, "im": 1.0}, {"re": 0.01, "im": -1.0}],
        "k": 9.0, "plane": "f"}})
    assert orc.plant_modes(orc.analytic_plant(scen, modules=("PLANT_A",))).__len__() == 2
