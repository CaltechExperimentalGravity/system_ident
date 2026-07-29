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
