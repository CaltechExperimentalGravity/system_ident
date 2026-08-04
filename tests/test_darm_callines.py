"""DARM calibration lines: per-stage actuation measurement for the hierarchical quad drive.

DARM is actuated hierarchically across quad masses (M0/PUM/TST). Calibration lines injected on
each stage, measured against a Pcal reference, recover each stage's actuation A_i = κ_i·D_i·N_i
(the ruler cancels sensing + loop) — the quantity you compare across stages to find where each
stage crosses over with the one below. The crossover itself lives in the `distribution` filters
(supplied from the twin actuation design); without them the raw mechanical columns don't cross.
"""
import numpy as np
import pytest

from system_ident.darm import DARMLoop
from system_ident import darm_tv as tv


class _LP:
    """First-order low-pass distribution filter (unit DC gain)."""
    def __init__(self, fc): self.fc = fc
    def eval(self, f): return 1.0 / (1.0 + 1j * np.asarray(f, float) / self.fc)


class _HP:
    def __init__(self, fc): self.fc = fc
    def eval(self, f):
        f = np.asarray(f, float); return (1j * f / self.fc) / (1.0 + 1j * f / self.fc)


def _noisy(loop):
    loop.sensor_asd = 300.0
    loop.disturbance_asd = 3e-4
    return loop


def test_stage_set_is_M0_PUM_TST():
    loop = DARMLoop.default_reduced()
    assert list(loop.stages) == ["M0", "PUM", "TST"]
    assert loop.ports == ["PCAL", "M0", "PUM", "TST"]


def test_cal_lines_recover_per_stage_actuation():
    """The ruler H_stage/H_pcal returns A_i = κ_i·D_i·N_i = loop.stage(name, f) at each cal line,
    within the measurement CRB."""
    loop = _noisy(DARMLoop.default_reduced(fmin=10.0))
    freqs = np.array([15.0, 30.0, 60.0, 120.0])
    resp = tv.cal_line_response(loop, freqs, n_periods=16, seed=3)
    assert set(resp) == {"M0", "PUM", "TST"}
    for name, (lines, A, sigma) in resp.items():
        truth = loop.stage(name, lines)                      # the actuation the ruler should give
        assert np.all(sigma > 0)
        assert np.all(np.abs(A - truth) < 5 * sigma), f"{name} cal-line actuation off truth"


def test_cal_lines_snap_into_band():
    loop = DARMLoop.default_reduced(fmin=10.0)
    resp = tv.cal_line_response(loop, [15.0, 30.0], n_periods=16, seed=0)
    lines = resp["TST"][0]
    assert np.all(lines >= loop.fmin) and np.all(lines <= loop.fmax)
    with pytest.raises(ValueError):
        tv.cal_line_response(loop, [5.0], n_periods=16, seed=0)   # below band → no lines


def test_distribution_filters_create_a_measurable_crossover():
    """With hierarchical distribution filters set, adjacent stages' measured |A_i| cross — the
    crossover the cal lines exist to find. (A placeholder complementary set stands in for the
    twin's real actuation design.)"""
    loop = _noisy(DARMLoop.default_reduced(fmin=10.0))
    # PUM rolls off above ~40 Hz, TST takes over above it → a PUM/TST crossover in [40,120] Hz
    loop.distribution = {"M0": _LP(15.0), "PUM": _LP(40.0), "TST": _HP(40.0)}
    freqs = np.geomspace(20.0, 200.0, 12)
    resp = tv.cal_line_response(loop, freqs, n_periods=16, seed=3)
    f = resp["PUM"][0]
    diff = np.abs(resp["PUM"][1]) - np.abs(resp["TST"][1])
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    assert len(sign_changes) >= 1, "PUM and TST actuation should cross with the distribution set"
    f_x = f[sign_changes[0]]
    assert 30.0 < f_x < 160.0, f"PUM/TST crossover at {f_x:.1f} Hz outside expected range"


def test_darm_o4_asd_matches_the_vendored_curve():
    """The O4 displacement floor is the vendored aligo_O4high strain × 4 km, log-log interpolated:
    it must reproduce the tabulated points and show the O4 shape (best ~1.2e-20 m/√Hz mid-band, a
    steep seismic wall below ~20 Hz, shot noise rising above ~1 kHz)."""
    from system_ident.darm import darm_o4_asd, _o4_strain_table, _ALIGO_ARM_LENGTH_M

    tf, ts = _o4_strain_table()
    # On the tabulated grid the interpolation is exact (× arm length).
    got = darm_o4_asd(tf)
    assert np.allclose(got, _ALIGO_ARM_LENGTH_M * ts, rtol=1e-12)

    best = darm_o4_asd(np.geomspace(50.0, 500.0, 200)).min()
    assert 8e-21 < best < 2e-20, f"mid-band best {best:.2e} off the ~1.2e-20 m/√Hz O4 bucket"
    # Steep seismic wall: 10 Hz is far noisier than the mid-band minimum.
    assert darm_o4_asd(10.0) > 100 * best
    # Shot noise climbs above ~1 kHz, and out-of-table frequencies clamp (finite, no blow-up).
    assert darm_o4_asd(3000.0) > darm_o4_asd(300.0)
    assert np.isfinite(darm_o4_asd(1.0)) and np.isfinite(darm_o4_asd(9000.0))


def test_pcal_free_mass_range_matches_hardware():
    """Pcal actuator range: radiation-pressure force from the ±200 mW power modulation on the 40 kg
    free test mass, x = F/(M(2πf)²) ∝ 1/f². Check the magnitude and the free-mass slope."""
    from system_ident import darm_callines as cl
    x100 = float(cl.pcal_range_disp(100.0))
    # F_rms = (0.2 W / c)/√2 ≈ 4.71e-10 N; x(100 Hz) = F/(40·(2π·100)²) ≈ 2.98e-17 m
    assert 2.5e-17 < x100 < 3.5e-17, f"Pcal range at 100 Hz = {x100:.2e} m, expected ~3e-17"
    # 1/f² free-mass roll-off: a decade up drops the range by ~100×
    assert np.isclose(cl.pcal_range_disp(100.0) / cl.pcal_range_disp(1000.0), 100.0, rtol=0.02)


# ── joint TDCF Fisher / A-optimal line design + designed-line readout ─────────────────────────────
def _o4_loop():
    from system_ident.darm import darm_o4_asd
    loop = DARMLoop.default_reduced(fmin=10.0, hierarchical=True).with_params(delta=np.radians(5.0))
    loop.noise_asd = darm_o4_asd
    return loop


# The identifiable joint set for the O4 floor + ≥10 Hz lines: sensing gain κ_C (g_c), SRC detuning
# δ, and the two in-band actuation stages. (κ_M0 acts <0.5 Hz — its line is out of this band — and
# f_cc/τ need a wideband high-f Pcal spread; the design CRB quantifies those separately.)
_SCOPE = ("g_c", "delta", "kappa_PUM", "kappa_TST")
_PRIORS = {"g_c": 0.02, "delta": 0.02, "kappa_PUM": 0.02, "kappa_TST": 0.02}


def test_joint_fisher_crb_scales_as_one_over_sqrt_T():
    from system_ident import darm_callines as cl
    loop = _o4_loop()
    caps = cl.stage_force_caps(loop, names=_SCOPE)
    roster = [(f, p, cl.line_displacement(loop, p, f, caps, pcal_weight=0.5))
              for f, p in [(30.0, "PCAL"), (200.0, "PCAL"), (28.0, "PUM"), (210.0, "TST")]]
    _, cov1, _, _ = cl.joint_fisher(loop, roster, T=10.0, names=_SCOPE)
    _, cov4, _, _ = cl.joint_fisher(loop, roster, T=40.0, names=_SCOPE)
    ratio = np.sqrt(np.diag(cov1)) / np.sqrt(np.diag(cov4))
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-6)             # σ ∝ 1/√T


def test_a_optimal_design_beats_broadband_and_is_well_conditioned():
    from system_ident import darm_callines as cl
    loop = _o4_loop()
    caps = cl.stage_force_caps(loop, names=_SCOPE)
    nom = cl._nominal(loop, _SCOPE)
    frac = {"g_c", "kappa_PUM", "kappa_TST"}
    abs_std = {n: (_PRIORS[n] * abs(nom[i]) if n in frac else _PRIORS[n]) for i, n in enumerate(_SCOPE)}
    P = cl._prior_cov(abs_std, _SCOPE)
    d = cl.design_lines(loop, _PRIORS, T=60.0, n_pcal=3, names=_SCOPE)
    # broadband on every port (fair: measures the same params, but spreads the power thin)
    bb = cl.broadband_roster(loop, _SCOPE, caps, n_per_port=10)
    cost_bb = cl.a_optimal_cost(cl.joint_fisher(loop, bb, 60.0, names=_SCOPE)[0], P)
    assert d["cost"] < cost_bb, f"A-optimal {d['cost']:.3f} not better than broadband {cost_bb:.3f}"
    assert d["full_rank"], "designed roster is rank-deficient"
    # data-A cost = Σ (snapshot σ / drift prior)² = Σ 1/margin²; < n_par ⇔ every drift resolved
    assert 0.0 < d["cost"] < len(_SCOPE)
    assert all(m > 1.0 for m in d["margins"].values())            # every drift resolved


def test_designed_line_readout_recovers_joint_params_at_predicted_crb():
    """End-to-end: design the lines, inject+read them leakage-free on the twin, and recover the joint
    parameters — each within a few σ of truth, and the empirical σ near the joint_fisher CRB."""
    from system_ident import darm_callines as cl
    from system_ident import darm_tv as tv
    loop = _o4_loop()
    T = 32 * 4096 / loop.fs
    d = cl.design_lines(loop, _PRIORS, T=T, n_pcal=3, names=_SCOPE)
    truth = {"g_c": 1.04e6, "delta": np.radians(5.3), "kappa_PUM": 1.03, "kappa_TST": 1.05}
    theta, sig, corr, names = tv.joint_snapshot_lines(loop, truth, d["roster"],
                                                      nperseg=4096, n_periods=32, seed=11)
    _, cov, _, _ = cl.joint_fisher(loop, d["roster"], T, names=_SCOPE)
    pred = dict(zip(_SCOPE, np.sqrt(np.diag(cov))))
    for n in names:
        assert abs(theta[n] - truth[n]) < 5 * sig[n], f"{n}: {theta[n]:.5g} vs {truth[n]:.5g}"
        # Order-of-magnitude cross-check: the simulator's empirical (JᵀJ)⁻¹ (from the FRF errors) and
        # the analytic displacement-SNR Fisher use different noise-normalisation conventions, so agree
        # to ~1 decade, not exactly (tightest on g_c/κ; loosest on δ, whose info sits at high f).
        assert 0.05 < sig[n] / pred[n] < 20.0, f"{n}: emp σ {sig[n]:.2e} vs CRB {pred[n]:.2e}"
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)
    assert np.all(np.abs(corr) <= 1.0 + 1e-6)


def test_stage_force_caps_are_grounded_derived():
    from system_ident import darm_callines as cl
    from system_ident import provenance as prov
    loop = _o4_loop()
    cl.stage_force_caps(loop, names=_SCOPE)
    reg = prov.registry()
    for st in ("PUM", "TST"):
        assert reg[f"stage_force_cap_{st}"].kind == prov.DERIVED


def test_pcal_budget_crosscheck_within_an_order_of_magnitude():
    """The ±200 mW Pcal displacement at 17.1 Hz agrees with the O4 Fig. 2 line height (× 4 km) to
    within ~1 order of magnitude — a real grounding cross-check, not a fitted number."""
    from system_ident import darm_callines as cl
    r = cl.pcal_budget_crosscheck(_o4_loop())["ratio"]
    assert 0.1 < r < 10.0, f"Pcal budget vs O4 Fig.2 ratio {r:.2f} off by >1 decade"


def test_provenance_gate_flags_a_planted_assumption():
    from system_ident import provenance as prov
    import system_ident.darm, system_ident.darm_callines  # noqa: F401  (populate the registry)
    prov.require_grounded()                                        # the shipped inputs are grounded
    prov.record("planted_placeholder", 1.23, prov.ASSUMED, "test: a number we do not have")
    try:
        with __import__("pytest").raises(AssertionError):
            prov.require_grounded()
    finally:
        prov._REGISTRY.pop("planted_placeholder", None)


def test_o4_floor_clamped_vs_seismic_wall():
    """The simulation floor (darm_o4_asd) clamps flat below the ~10 Hz table start; the DESIGN floor
    (darm_o4_asd_seismic) extrapolates the seismic wall (≈ f^−7) — steeply rising, finite to DC, and
    identical to the clamped floor at/above 10 Hz (so every line ≥10 Hz is simulated exactly)."""
    from system_ident.darm import darm_o4_asd, darm_o4_asd_seismic
    f_hi = np.geomspace(11.0, 4000.0, 40)
    np.testing.assert_allclose(darm_o4_asd(f_hi), darm_o4_asd_seismic(f_hi), rtol=1e-9)
    assert np.isclose(darm_o4_asd(1.0), darm_o4_asd(9.0))                  # clamped flat below table
    assert darm_o4_asd_seismic(1.0) > 100.0 * darm_o4_asd_seismic(9.0)    # wall rises steeply
    assert np.all(np.isfinite(darm_o4_asd_seismic(np.array([0.0, 0.1, 1.0, 10.0]))))
