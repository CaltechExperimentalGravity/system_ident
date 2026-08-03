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


def test_lineset_authority_is_absolute_pcal_and_rolled_off_stages():
    """build_lineset authority = full-range DARM displacement [m]: Pcal from pcal_range_disp, and
    each stage's authority rolls off from its low-frequency peak (the upper masses drive only low)."""
    from system_ident import darm_callines as cl
    loop = cl.default_cal_loop()
    ls = cl.build_lineset(loop, [cl.Line(100.0, "PCAL"), cl.Line(15.0, "M0"),
                                 cl.Line(150.0, "M0")])
    assert np.isclose(ls.authority[0], cl.pcal_range_disp(100.0), rtol=1e-6)   # Pcal = free-mass range
    assert ls.authority[1] > ls.authority[2]                                   # M0 stronger low than high
    assert np.all(ls.authority > 0)
