"""40m SOS Stage-0 plant: feasibility gate, construction consistency, and the OSEM layer."""
import math

import numpy as np
import pytest

from system_ident.sos_plant import (
    DOFS, MEASURED_MODES_HZ, SOSOptic, build_plant, fit_suspension, inertia_vector,
    labelled_modes, modes_hz, stiffness_matrix,
)


@pytest.fixture(scope="module")
def fitted():
    optic = SOSOptic()
    return optic, fit_suspension(optic)


@pytest.fixture(scope="module")
def plant(fitted):
    optic, susp = fitted
    return build_plant(optic, susp)


def _labelled_modes(sys):
    """{DOF: f0}, using the module's inertia-weighted labelling."""
    return {d: f0 for d, (f0, _q) in labelled_modes(sys).items()}


def test_mode_labelling_needs_the_inertia_weighting():
    """Guards a real trap: a raw |v| argmax mislabels the 1 Hz L mode as P.

    The eigenvector mixes metres and radians, so the raw components are not comparable —
    the L mode's pitch component is numerically larger. Weighting by M_ii fixes it.
    """
    optic = SOSOptic()
    sys = build_plant(optic, fit_suspension(optic))
    w, v = np.linalg.eig(np.asarray(sys.A, float))
    naive, weighted = {}, {}
    for i, z in enumerate(w):
        if z.imag <= 1e-9:
            continue
        f0 = abs(z) / (2 * math.pi)
        naive[DOFS[int(np.argmax(np.abs(v[:6, i])))] ] = f0
        weighted[DOFS[int(np.argmax(inertia_vector(optic) * np.abs(v[:6, i]) ** 2))]] = f0
    assert set(weighted) == set(DOFS)          # every DOF accounted for
    assert set(naive) != set(DOFS)             # the naive metric loses one
    assert weighted["L"] == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------- optic


def test_optic_matches_40m_geometry():
    o = SOSOptic()
    # fused silica cylinder, R=37.5 mm, H=25 mm
    assert o.mass_kg == pytest.approx(0.243, abs=5e-4)
    # independently, T000134's single-loop-suspended mirror quotes I_theta = 9.78e-5
    assert o.I_pitch == pytest.approx(9.78e-5, rel=5e-3)
    assert o.I_yaw == pytest.approx(o.I_pitch)
    # roll is about the symmetry axis: I = m R^2 / 2, strictly larger than the transverse
    assert o.I_roll > o.I_pitch


# ------------------------------------------------------------------- feasibility gate


def test_gate_all_six_modes_present(plant):
    f = _labelled_modes(plant)
    assert set(f) == set(DOFS), f"missing DOF: {set(DOFS) - set(f)}"


@pytest.mark.parametrize("dof", ["L", "P", "V", "R", "Y"])
def test_gate_fitted_modes_hit_measured(plant, dof):
    """These are fit targets, so agreement is by construction — this guards the solver."""
    f = _labelled_modes(plant)
    assert f[dof] == pytest.approx(MEASURED_MODES_HZ[dof], rel=1e-6)


def test_side_mode_is_predicted_not_fitted(plant):
    """T is the real test: omega_T^2 = g/(l+b) follows from the pinned l and b."""
    f = _labelled_modes(plant)
    assert f["T"] == pytest.approx(MEASURED_MODES_HZ["T"], rel=3e-3)
    # ...and it is genuinely not exact, i.e. it really was not fitted
    assert f["T"] != pytest.approx(MEASURED_MODES_HZ["T"], rel=1e-9)


def test_wire_separation_lands_at_the_optic_diameter(fitted):
    """roll/vertical = sqrt(2) is equivalent to wires at the optic edge (d = 2R)."""
    optic, susp = fitted
    assert susp.wire_sep_m == pytest.approx(2 * optic.radius_m, rel=1e-9)


def test_fitted_length_and_breakoff_are_physical(fitted):
    optic, susp = fitted
    # ~1 Hz pendulum -> ~25 cm wire; T000134 independently quotes 0.248 m
    assert susp.wire_length_m == pytest.approx(0.248, rel=5e-3)
    # break-off sits just above the CoM, and below T000134's (our pitch is lower: 0.6 vs 0.78)
    assert 0.0 < susp.breakoff_m < 0.985e-3


# ------------------------------------------------------------------------ state space


def test_plant_shape_and_stability(plant):
    assert plant.nstates == 12
    assert plant.ninputs == 6
    assert plant.noutputs == 6
    assert np.all(np.real(np.linalg.eigvals(np.asarray(plant.A, float))) < 0)


def test_structural_q_is_the_convention(plant):
    for _f0, q in modes_hz(plant):
        assert q == pytest.approx(50.0, rel=0.05)


def test_only_longitudinal_pitch_is_coupled(fitted):
    optic, susp = fitted
    K = stiffness_matrix(optic, susp)
    off = [(i, j) for i in range(6) for j in range(6)
           if i != j and abs(K[i, j]) > 0]
    lp = {(DOFS.index("L"), DOFS.index("P")), (DOFS.index("P"), DOFS.index("L"))}
    assert set(off) == lp


def test_frf_is_finite_and_right_shape(plant):
    import control
    freq = np.logspace(-1, 2, 50)
    mag, _phase, _w = control.frequency_response(plant, 2 * np.pi * freq)
    assert mag.shape == (6, 6, 50)
    assert np.all(np.isfinite(mag))


# ------------------------------------------------------------------------------ OSEM


def test_osem_round_trip_is_identity_on_actuated_dofs():
    from system_ident.osem import euler_to_osem, osem_to_euler, actuated_mask
    RT = osem_to_euler() @ euler_to_osem()
    expected = np.diag(actuated_mask().astype(float))
    assert np.allclose(RT, expected, atol=1e-12)


def test_osem_cannot_reach_vertical_or_roll():
    from system_ident.osem import euler_to_osem, UNACTUATED_DOFS
    M = euler_to_osem()
    for dof in UNACTUATED_DOFS:
        assert np.allclose(M[:, DOFS.index(dof)], 0.0)
    assert np.linalg.matrix_rank(M) == 4


def test_butterfly_pattern_produces_no_dof():
    from system_ident.osem import BUTTERFLY, osem_to_euler
    assert np.allclose(osem_to_euler() @ np.array(BUTTERFLY), 0.0, atol=1e-12)


def test_osem_projected_plant_is_five_by_five(plant):
    from system_ident.osem import project_plant
    p5 = project_plant(plant)
    assert p5.ninputs == 5
    assert p5.noutputs == 5
    assert p5.nstates == plant.nstates


def test_coil_force_to_dof_is_the_transpose_not_the_pinv():
    from system_ident.osem import coil_force_to_dof, euler_to_osem, osem_to_euler
    assert np.allclose(coil_force_to_dof(), euler_to_osem().T)
    # the pinv would rescale the 4x coil gain away; the transpose must not equal it
    assert not np.allclose(coil_force_to_dof(), osem_to_euler())


@pytest.mark.parametrize("dof,expected", [
    ("L", (1, 1, 1, 1, 0)),      # all face coils together
    ("P", (1, 1, -1, -1, 0)),    # upper vs lower
    ("Y", (1, -1, -1, 1, 0)),    # left vs right
    ("T", (0, 0, 0, 0, 1)),      # side coil alone
])
def test_osem_patterns_match_cds_geometry(dof, expected):
    """Signs agree with the salvaged X1:SUS-ITMXP_COIL_IN_C2DOF matrix."""
    from system_ident.osem import euler_to_osem
    assert np.allclose(euler_to_osem()[:, DOFS.index(dof)], expected)


# ---------------------------------------------------------------- loader compatibility


@pytest.mark.parametrize("name", ["hsts", "quad"])
def test_default_suffix_still_loads_the_twin_plants(name):
    """Generalizing load() must not disturb the modal-truncation plants."""
    from system_ident.reduced_plant import ReducedStateSpacePlant
    p = ReducedStateSpacePlant.load(name)
    assert p.A.shape[0] == p.A.shape[1] > 0
    assert p.B.shape == (p.A.shape[0], len(p.inputs))
    assert p.C.shape == (len(p.outputs), p.A.shape[0])
    assert len(p.modes()) > 0
