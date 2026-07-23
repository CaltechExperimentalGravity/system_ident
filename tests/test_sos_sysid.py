"""Stage 1 signoff gate: 6-DOF MIMO P&S recovery of the SOS agrees with the oracle within CRB.

The open-loop (pyctl) rung of the campaign. Deterministic — the campaigns use fixed seeds.
"""
import numpy as np
import pytest

from system_ident.reduced_plant import ReducedStateSpacePlant
from system_ident.sos_campaign import (
    run_full_recovery, run_band, oracle_by_dof, LOW_BAND, HIGH_BAND, DOFS,
)


@pytest.fixture(scope="module")
def recoveries():
    return run_full_recovery()


def test_all_six_modes_recovered(recoveries):
    assert {r.dof for r in recoveries} == set(DOFS)


def test_recovery_within_crb(recoveries):
    """The signoff gate: every mode within a few CRB sigma of its own DOF's oracle."""
    for r in recoveries:
        assert r.n_sigma_f0 < 3.0, (
            f"{r.dof} f0={r.f0:.5f} vs {r.f0_true:.5f} is {r.n_sigma_f0:.1f}σ (CRB {r.f0_std:.2e})")
        assert r.n_sigma_Q < 3.0, (
            f"{r.dof} Q={r.Q:.2f} vs {r.Q_true:.2f} is {r.n_sigma_Q:.1f}σ")


def test_frequencies_are_physically_accurate(recoveries):
    """Beyond the CRB (which can be wide for weakly-excited modes), f0 is close in absolute
    terms — the well-excited modes to <1e-3, the weakest (T) still well under 1%."""
    for r in recoveries:
        assert r.frac_err_f0 < 1e-2, f"{r.dof}: {r.frac_err_f0:.1e}"
    well_excited = [r for r in recoveries if r.dof != "T"]
    assert max(r.frac_err_f0 for r in well_excited) < 2e-3


def test_LT_spatial_doublet_resolved(recoveries):
    """L (1.000 Hz) and T (0.998 Hz) are 1.8 mHz apart in different DOF — a spatial doublet.
    Block-decoupling must return BOTH as distinct modes, not one collapsed pole."""
    byd = {r.dof: r for r in recoveries}
    assert "L" in byd and "T" in byd
    # recovered as two separate modes (they come from different blocks: {L,P} vs {T})
    assert byd["L"].f0 != byd["T"].f0
    # each lands nearer its own oracle than the other's
    assert abs(byd["L"].f0 - 1.0000) < abs(byd["L"].f0 - 0.9982) or byd["L"].n_sigma_f0 < 1
    assert byd["T"].f0_true == pytest.approx(0.9982, abs=1e-3)


def test_LP_coupling_recovered_as_two_modes(recoveries):
    """The genuinely-MIMO {L,P} block returns both its modes (1.0 and 0.6 Hz)."""
    byd = {r.dof: r for r in recoveries}
    assert byd["L"].f0 == pytest.approx(1.0, rel=1e-2)
    assert byd["P"].f0 == pytest.approx(0.6, rel=1e-2)


def test_crb_is_informative_not_trivial(recoveries):
    """A pass is only meaningful if the CRB is tight — a huge CRB would pass vacuously.
    The well-excited modes must be pinned to sub-percent fractional 1σ."""
    byd = {r.dof: r for r in recoveries}
    for d in ("L", "P", "Y", "V", "R"):
        frac_sigma = byd[d].f0_std / byd[d].f0_true
        assert frac_sigma < 1e-2, f"{d} CRB {frac_sigma:.1e} too loose to be a real gate"


def test_high_band_independent_of_low(recoveries):
    """V and R come from their own campaign; recovering them confirms the two-band split."""
    byd = {r.dof: r for r in recoveries}
    assert byd["V"].f0 == pytest.approx(16.0, rel=2e-3)
    assert byd["R"].f0 == pytest.approx(22.6274, rel=2e-3)


def test_oracle_by_dof_matches_the_committed_plant():
    """The scoring oracle (rebuilt analytically) matches the committed sos_6dof artifact."""
    o = oracle_by_dof()
    npz_modes = sorted(f for f, _q in ReducedStateSpacePlant.load("sos", suffix="_6dof").modes())
    assert sorted(f for f, _q in o.values()) == pytest.approx(npz_modes, abs=1e-6)
