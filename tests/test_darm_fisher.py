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
    σ = 1/(√2·SNR), SNR = amp·√T/floor — a closed-form check on the Fisher normalisation."""
    f, amp, T = 30.0, 2.0e-4, 120.0
    ls = cl.build_lineset(loop, [cl.Line(f, "TST", amp)])
    # isolate κ_TST information (single line, single informative param via the ruler column)
    gamma = cl.fisher(ls, np.array([amp]), T)
    idx = cl.TDCF_PARAMS.index("kappa_TST")
    snr = amp * np.sqrt(T) / cl.floor_asd(loop, np.array([f]))[0]
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
