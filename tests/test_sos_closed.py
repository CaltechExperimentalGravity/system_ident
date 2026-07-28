"""Stage 2 (pyctl): the 40m SOS closed-loop identification through its nominal loops.

The campaign result: with the nominal OPT_CTRL_SUS* controllers closed and an in-loop
disturbance, the naive FRF estimate is biased on every damped DOF while P&S is consistent —
and the undamped V/R modes (open loops) show naive==P&S as a built-in null control.
"""
import numpy as np
import control
import pytest

from system_ident import sos_closed as sc
from system_ident.sos_plant import DOFS


def test_nominal_controllers_load_and_reconstruct():
    """The committed controllers reproduce the salvaged SUSSIDE bank response (mag+phase)."""
    K = sc.load_nominal_controllers()
    assert set(K) == set(sc.DAMPED)
    fr = control.frequency_response(K["T"], 2 * np.pi * np.array([0.3, 1.0, 3.0])).frdata.ravel()
    # from the 16384 Hz digital bank: |K|,phase = 0.100@78.7, 0.343@51.8, 1.116@-37.1
    assert abs(fr[0]) == pytest.approx(0.1003, rel=1e-2)
    assert np.degrees(np.angle(fr[1])) == pytest.approx(51.8, abs=1.0)
    assert abs(fr[2]) == pytest.approx(1.116, rel=1e-2)


def test_loop_stable_and_damps_only_LTPY():
    """Full 6-DOF loop stable; nominal controllers damp L/T/P/Y; V/R stay at open-loop Q=50."""
    G, Kfull, kappa = sc.build_closed_loop()
    assert np.all(np.real(sc.closed_loop_poles(G, Kfull)) < 0)
    Q = sc.damped_Q(G, Kfull)
    for d in ("L", "T", "P", "Y"):
        assert 5.0 < Q[d] < 25.0, f"{d} closed-loop Q={Q[d]:.1f} (expected damped from 50)"
    for d in ("V", "R"):
        assert Q[d] == pytest.approx(50.0, abs=1.0), f"{d} should be undamped (Q~50), got {Q[d]:.1f}"


@pytest.fixture(scope="module")
def experiment():
    return sc.assemble(fs=2048.0, kappa_ref=3.0, seed=0)


@pytest.fixture(scope="module")
def errors(experiment):
    lo = experiment.per_dof_errors(n_periods=16, sigma=0.5, seeds=20)
    hi = experiment.per_dof_errors(n_periods=128, sigma=0.5, seeds=20)
    return {"lo": lo, "hi": hi}


def test_naive_is_biased_on_damped_dofs(errors):
    """Naive error is a bias floor: essentially unchanged from P=16 to P=128 on L/T/P/Y."""
    nlo, _ = errors["lo"]; nhi, _ = errors["hi"]
    for d in ("L", "T", "P", "Y"):
        assert abs(nhi[d] - nlo[d]) / nlo[d] < 0.15, (
            f"{d} naive not a flat floor: {nlo[d]:.3f} (P16) -> {nhi[d]:.3f} (P128)")


def test_ps_consistent_and_beats_naive_on_damped_dofs(errors):
    """P&S error falls with P and overtakes the biased naive estimate on every damped DOF."""
    _, plo = errors["lo"]; nhi, phi = errors["hi"]
    for d in ("L", "T", "P", "Y"):
        assert phi[d] < plo[d], f"{d} P&S did not fall with P: {plo[d]:.3f} -> {phi[d]:.3f}"
        assert phi[d] < 0.6 * nhi[d], f"{d} P&S ({phi[d]:.3f}) did not beat naive ({nhi[d]:.3f})"


def test_undamped_dofs_show_no_closed_loop_bias(errors):
    """V and R have no controller -> open loop -> naive == P&S (the built-in null control)."""
    nlo, plo = errors["lo"]; nhi, phi = errors["hi"]
    for d in ("V", "R"):
        assert abs(nlo[d] - plo[d]) / nlo[d] < 0.1, f"{d} P16 naive!=P&S: {nlo[d]:.3f} vs {plo[d]:.3f}"
        assert abs(nhi[d] - phi[d]) / nhi[d] < 0.1, f"{d} P128 naive!=P&S: {nhi[d]:.3f} vs {phi[d]:.3f}"


def test_balancing_conditions_the_recovery(experiment):
    """The balanced plant is O(1) on every diagonal — otherwise the 1e4x DOF gain spread makes
    the MIMO recovery singular. (Guards the closed_loop_id.balance fix.)"""
    peaks = [np.max(np.abs(experiment.Gk[:, i, i])) for i in range(6)]
    assert max(peaks) / min(peaks) < 10.0, f"diagonal peak spread {max(peaks)/min(peaks):.1f}, not balanced"
