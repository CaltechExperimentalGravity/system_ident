"""DARM hierarchical actuation using the twin's nested-offload distribution filters.

`default_reduced(hierarchical=True)` populates DARMLoop.distribution with the offload filters
reproduced from digital_twin's cavity_arm_lsc_hierarchical experiment (D_TST=1, D_PUM=O_A,
D_M0=O_A·O_B) and sets κ to the physical per-stage authorities (STAGE_GAINS). The strong, slow
M0 then dominates the DARM actuation at low frequency and hands off up the chain, so adjacent
stages cross over — the crossover the calibration lines exist to measure.
"""
import numpy as np
import control as ct
import pytest

from system_ident.darm import DARMLoop
from system_ident import darm_tv as tv
from system_ident import darm_actuation as da


def test_offload_filters_reproduce_twin_design():
    """O_A, O_B match the documented nested-offload design (integrator + leads/lowpass + margin
    biquads). Guards the reproduction against drift."""
    O_A, O_B = da.offload_filters()
    f = np.geomspace(0.1, 100, 30); w = 2 * np.pi * f
    hA = np.abs(np.asarray(ct.frequency_response(O_A, w).frdata).ravel())
    hB = np.abs(np.asarray(ct.frequency_response(O_B, w).frdata).ravel())
    # integral action: high gain at low frequency, rolling down
    assert hA[0] > hA[-1] and hB[0] > hB[-1]
    # O_A crosses unity well above O_B (ESD/PUM handoff above PUM/TOP handoff)
    fA = f[np.argmin(np.abs(hA - 1))]; fB = f[np.argmin(np.abs(hB - 1))]
    assert fA > fB


def test_hierarchical_sets_distribution_and_authorities():
    loop = DARMLoop.default_reduced(hierarchical=True)
    assert set(loop.distribution) == {"M0", "PUM", "TST"}
    # TST is the direct (unity) stage; M0/PUM carry the offload filters
    f = np.array([1.0, 10.0])
    np.testing.assert_allclose(loop.distribution["TST"].eval(f), np.ones(2, complex))
    O_A, O_B = da.offload_filters()
    np.testing.assert_allclose(loop.distribution["PUM"].eval(f),
                               np.asarray(ct.frequency_response(O_A, 2*np.pi*f).frdata).ravel(), rtol=1e-9)


def test_non_hierarchical_default_has_no_distribution():
    assert DARMLoop.default_reduced().distribution == {}


def test_actuation_hierarchy_M0_low_f_then_hands_off():
    """M0 (strong, slow) dominates DARM actuation at low frequency; a higher stage takes over
    above the M0/PUM crossover — the hierarchical hand-off."""
    loop = DARMLoop.default_reduced(fmin=0.1, hierarchical=True)
    f = np.geomspace(0.1, 300, 6000)
    mags = {s: np.abs(loop.stage(s, f)) for s in loop.stages}
    dom = lambda fq: max(mags, key=lambda s: mags[s][int(np.argmin(np.abs(f - fq)))])
    assert dom(0.3) == "M0"                    # slow strong stage owns low frequency
    assert dom(50.0) != "M0"                   # handed off by mid-band
    d = mags["M0"] - mags["PUM"]
    assert np.any(np.diff(np.sign(d)) != 0), "M0/PUM actuation should cross"


def test_G_identity_and_kappa_recovery_with_distribution():
    """Loop identity holds with the distribution, and the Pcal ruler still recovers each stage's
    κ (the ruler uses the full D_i·N_i shape, so the distribution doesn't bias it)."""
    loop = DARMLoop.default_reduced(fmin=10.0, hierarchical=True)
    f = np.geomspace(10, 1500, 2000)
    np.testing.assert_allclose(loop.G(f), loop.A(f) * loop.D(f) * loop.C(f), rtol=1e-8)
    loop.sensor_asd = 300.0; loop.disturbance_asd = 3e-4
    # recover the fast (TST) and mid (PUM) stages in the band where they are active
    for name in ("TST", "PUM"):
        _, kappa0 = loop.stages[name]
        khat, sig = tv.snapshot_kappa(loop, name, kappa0, n_periods=16, seed=7)
        assert abs(khat - kappa0) < 4 * sig, f"{name}: {khat:.4g} vs {kappa0:.4g} ({(khat-kappa0)/sig:.1f}σ)"


def test_cal_lines_measure_per_stage_actuation_with_distribution():
    """Cal lines return each stage's ruler-calibrated actuation A_i = κ_i·D_i·N_i (= loop.stage)
    with the hierarchical distribution active."""
    loop = DARMLoop.default_reduced(fmin=10.0, hierarchical=True)
    loop.sensor_asd = 300.0; loop.disturbance_asd = 3e-4
    freqs = np.array([15.0, 40.0, 100.0])
    resp = tv.cal_line_response(loop, freqs, n_periods=16, seed=3)
    for name, (lines, A, sigma) in resp.items():
        truth = loop.stage(name, lines)
        assert np.all(np.abs(A - truth) < 5 * sigma), f"{name} cal-line actuation off truth"
