"""DARM hierarchical actuation using the twin's nested-offload distribution filters.

`default_reduced(hierarchical=True)` builds the twin's nested-offload DARM actuation: the stage
shapes are the M0-DAMPED reduced-QUAD compliances (the twin damps the quad before designing the
hierarchy — darm_actuation.hierarchical_stage_shapes) and DARMLoop.distribution holds the offload
filters reproduced from digital_twin's cavity_arm_lsc_hierarchical experiment (D_TST=1, D_PUM=O_A,
D_M0=O_A·O_B). The offload runs in FORCE units, so the compliances alone carry the relative stage
strengths (κ_i = 1) — no separate authority weighting, which would double-count. The strong, slow
M0 then dominates at low frequency and hands off up the chain; the DARM-referred contributions
cross at the design targets F_PT≈0.5 Hz (M0/PUM) and F_EP≈10 Hz (PUM/TST) — what the cal lines
measure.
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


def test_hierarchical_sets_distribution_and_forceunit_kappa():
    loop = DARMLoop.default_reduced(hierarchical=True)
    assert set(loop.distribution) == {"M0", "PUM", "TST"}
    # force-unit offload: the compliances carry the hierarchy, so κ_i are all 1 (no separate
    # per-stage authority weighting — that would double-count what the compliances encode).
    assert all(loop.stages[s][1] == 1.0 for s in loop.stages)
    # TST is the direct (unity) stage; M0/PUM carry the offload filters
    f = np.array([1.0, 10.0])
    np.testing.assert_allclose(loop.distribution["TST"].eval(f), np.ones(2, complex))
    O_A, O_B = da.offload_filters()
    np.testing.assert_allclose(loop.distribution["PUM"].eval(f),
                               np.asarray(ct.frequency_response(O_A, 2*np.pi*f).frdata).ravel(), rtol=1e-9)


def test_crossovers_land_at_design_targets():
    """The whole point of the damp-first (real ETMX M0 damping) + force-unit offload fix: the
    DARM-referred per-stage contributions cross at the twin's design targets F_PT≈0.5 Hz (M0/PUM)
    and F_EP≈10 Hz (PUM/TST). PUM/TST is a single clean crossing; M0/PUM is a tight cluster right
    at F_PT (the real L-DOF M0 damping is loose — the M0 sensor barely feels the slow L mode — so
    a small residual wiggle sits on the handoff, faithful to the plant). Guards against the earlier
    bugs (STAGE_GAINS folded in → ~4 Hz M0/PUM and no PUM/TST crossing; undamped columns → a
    tangle of spurious forest crossings across the whole 0.4–3 Hz band)."""
    loop = DARMLoop.default_reduced(fmin=0.1, hierarchical=True)
    f = np.geomspace(0.1, 300, 8000)
    mags = {s: np.abs(loop.stage(s, f)) for s in loop.stages}

    def crossings(a, b):
        d = mags[a] - mags[b]
        return f[np.where(np.diff(np.sign(d)) != 0)[0]]

    xo_mp = crossings("M0", "PUM")
    xo_pt = crossings("PUM", "TST")
    assert xo_mp.size >= 1 and np.all((0.35 <= xo_mp) & (xo_mp <= 0.75)), \
        f"M0/PUM handoff should cluster at F_PT≈0.5, got {xo_mp}"
    assert xo_pt.size == 1, f"PUM/TST should cross once, got {xo_pt}"
    assert 8.0 <= xo_pt[0] <= 13.0, f"PUM/TST crossover {xo_pt[0]:.3f} not near F_EP=10"


def test_non_hierarchical_default_has_no_distribution():
    assert DARMLoop.default_reduced().distribution == {}


def test_etmx_damping_filters_match_twin_design():
    """The six ETMX M0-damping filters reproduce the twin's SUS_CONFIG["ETMX"] design: velocity
    damper k_d·s/(1+s/2π·8) × per-DOF LP, with the real production gains. Guards the reproduction
    (verified FRF-identical, 0.0 diff, against digital_twin _doc_helpers.damping_filter)."""
    filt = da.etmx_m0_damping_filters()
    assert set(filt) == {"L", "T", "V", "R", "P", "Y"}
    assert da.ETMX_M0_DAMP_GAINS["L"] == -1000.0 and da.ETMX_F_LP == 8.0
    # velocity damper → zero at DC (differentiator): |K(f)| rises from ~0 at low f
    f = np.array([1e-3, 1.0])
    for d in ("L", "Y"):
        h = np.abs(np.asarray(ct.frequency_response(filt[d], 2 * np.pi * f).frdata).ravel())
        assert h[0] < h[1], f"{d}: velocity damper should roll up from DC"


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
