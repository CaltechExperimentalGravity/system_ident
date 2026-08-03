"""Fisher-optimal DARM calibration-line design (`darm_callines`).

The 7 TDCFs (κ_C, f_cc, δ, τ, κ_M0/PUM/TST) are measured by a Pcal reference plus one actuator
line per hierarchical stage. These tests check the CRB engine (identifiability, 1/√T scaling, an
analytic single-line check), the sizing to a 0.1%-in-5-min target, and that the Fisher-optimal
placement is no worse than the fixed O3/O4 line positions at equal drive.
"""
import numpy as np
import pytest

from system_ident import darm_callines as cl


@pytest.fixture(scope="module")
def loop():
    return cl.default_cal_loop(delta_deg=5.0)


def test_all_seven_tdcfs_identifiable(loop):
    """The Pcal + M0/PUM/ESD roster gives every TDCF a finite CRB (Fisher full-rank) — no
    degenerate placement leaves a parameter unmeasured."""
    lines = cl.seed_lines(loop)
    for ln in lines:
        ln.amp = 1.0
    sig = cl.tdcf_sigma(loop, lines, T=60.0)
    assert set(sig) == set(cl.TDCF_PARAMS)
    assert all(np.isfinite(s) and s > 0 for s in sig.values())


def test_crb_scales_as_one_over_sqrt_T(loop):
    ls = cl.build_lineset(loop, cl.seed_lines(loop))
    amps = np.ones(len(cl.TDCF_PARAMS))
    s1 = cl.sigma(ls, amps, 50.0)
    s4 = cl.sigma(ls, amps, 200.0)                       # 4× time → σ halves
    for k in cl.TDCF_PARAMS:
        assert s1[k] / s4[k] == pytest.approx(2.0, rel=1e-6)


def test_single_stage_line_matches_analytic_crb(loop):
    """One actuator line measuring its own κ (∂lnH/∂lnκ=1) has the textbook single-parameter CRB
    σ = 1/(√2·SNR), where the achievable DARM amplitude is drive·authority(f) — a closed-form check
    on the Fisher normalisation including the actuator-range weighting."""
    f, drive, T = 30.0, 2.0e-4, 120.0
    ls = cl.build_lineset(loop, [cl.Line(f, "TST", drive)])
    # isolate κ_TST information (single line, single informative param via the ruler column)
    gamma = cl.fisher(ls, np.array([drive]), T)
    idx = cl.TDCF_PARAMS.index("kappa_TST")
    snr = drive * ls.authority[0] * np.sqrt(T) / cl.floor_asd(loop, np.array([f]))[0]
    assert np.sqrt(1.0 / gamma[idx, idx]) == pytest.approx(1.0 / (np.sqrt(2) * snr), rel=1e-6)


def test_sizing_hits_target_and_reports_feasibility(loop):
    """The sized roster reaches 0.1% on every TDCF at its reported T_req, and the feasibility flag
    is consistent with the binding time (amplitude-only optimisation keeps the test fast)."""
    res = cl.size_lines_for_target(loop, A_tot=1.0, target=1e-3, T_ref=60.0, optimize_freq=False)
    # each param's σ at its own T_req equals the target (σ ∝ 1/√T)
    for k in cl.TDCF_PARAMS:
        s_at_treq = res["sigma"][k] * np.sqrt(res["T_ref"] / res["t_req"][k])
        assert s_at_treq == pytest.approx(res["target"], rel=1e-6)
    assert res["binding"] in cl.TDCF_PARAMS
    assert res["feasible"] == (res["t_req_max"] <= 300.0)


def test_response_budget_propagation(loop):
    """Propagating the TDCF CRB into δR/R(f): the response model matches loop.R at nominal, every
    parameter (including the κ's, with D held fixed) moves R, and the response uncertainty is
    finite, scales as 1/√T, and lands well inside the O3 budget at the 0.1% design point."""
    f = np.geomspace(10.0, 2000.0, 200)
    D = loop.D(f)
    R_model = (1.0 + loop.A(f) * D * loop.C(f)) / loop.C(f)
    np.testing.assert_allclose(R_model, loop.R(f), rtol=1e-9)          # D-fixed model == loop.R
    J = cl.response_log_jacobian(loop, f)
    assert J.shape == (len(f), len(cl.TDCF_PARAMS))
    assert np.all(np.max(np.abs(J), axis=0) > 0)                       # every param moves R somewhere
    res = cl.size_lines_for_target(loop, A_tot=1.0, target=1e-3, T_ref=60.0, optimize_freq=False)
    ls, amps = res["lineset"], np.array([ln.amp for ln in res["lines"]])
    m1, p1 = cl.response_budget(loop, ls, amps, res["t_req_max"], f)
    m4, p4 = cl.response_budget(loop, ls, amps, 4.0 * res["t_req_max"], f)
    assert np.all(np.isfinite(m1)) and np.all(m1 > 0)
    np.testing.assert_allclose(m1 / m4, 2.0, rtol=1e-6)                # σ ∝ 1/√T
    band = (f >= 20) & (f <= 2000)
    assert m1[band].max() < cl.O3_BUDGET["total_mag_pct"]             # inside the O3 budget
    assert p1[band].max() < cl.O3_BUDGET["total_phase_deg"]


def test_response_optimal_is_gentler_than_baselines(loop):
    """The measurement-design thesis: a response-optimal P&S scheme reaches a given random-error
    level with less injected energy K=A²·T than the O3/O4 fixed-line placement AND a naive broadband
    injection (K is scheme-characteristic since δR/R ∝ 1/√(A²T))."""
    tm, tp = cl.TARGET_LEVELS["O3 random"]
    rt = cl.rho_of_target(tm, tp)
    pns = cl.size_lines_for_response(loop, A_tot=1.0, T_ref=60.0, seed=0)
    o3 = cl.reference_scheme(loop, cl.O3_LINES, A_tot=1.0)
    nb_ls, nb_amps = cl.naive_broadband(loop, A_tot=1.0)
    K_pns = cl.pareto_cost(loop, pns["lineset"], pns["amps"], rt)
    K_fix = cl.pareto_cost(loop, o3["lineset"], np.array([l.amp for l in o3["lines"]]), rt)
    K_nb = cl.pareto_cost(loop, nb_ls, nb_amps, rt)
    assert K_pns < K_fix and K_pns < K_nb                          # gentler/faster than both
    assert K_fix / K_pns > 2.0                                     # a substantial factor
    # the reduction factor is target-independent (K ∝ 1/ρ_target²)
    rt2 = cl.rho_of_target(*cl.TARGET_LEVELS["0.1% stretch"])
    K_pns2 = cl.pareto_cost(loop, pns["lineset"], pns["amps"], rt2)
    K_fix2 = cl.pareto_cost(loop, o3["lineset"], np.array([l.amp for l in o3["lines"]]), rt2)
    assert (K_fix2 / K_pns2) == pytest.approx(K_fix / K_pns, rel=1e-6)


def test_optimal_placement_beats_fixed_o3_o4(loop):
    """Optimising line frequency + amplitude is no worse than the fixed O3/O4 positions (with
    their amplitudes optimally allocated) at equal total drive — optimal is optimal."""
    opt = cl.size_lines_for_target(loop, A_tot=1.0, target=1e-3, T_ref=60.0, seed=0)
    o3 = cl.reference_scheme(loop, cl.O3_LINES, A_tot=1.0)
    o4 = cl.reference_scheme(loop, cl.O4_LINES, A_tot=1.0)
    assert opt["t_req_max"] <= o3["t_req_max"] * 1.001
    assert opt["t_req_max"] <= o4["t_req_max"] * 1.001
    # the roster is 3 actuator + 4 Pcal lines
    kinds = [ln.kind for ln in opt["lines"]]
    assert sorted(k for k in kinds if k != "PCAL") == ["M0", "PUM", "TST"]
    assert kinds.count("PCAL") == 4
