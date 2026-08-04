"""Step-5: physical safety watchdog + safe-state handoff."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from system_ident.backends.twin import TwinBackend
from system_ident.plant import SuspensionPlant, double_pendulum
from system_ident.safety import SafetyAbort, SafetyLimits, Watchdog

FS = 32.0
EXC = {"X1:EXC_POS": "POS"}
RB = {"X1:RESP_POS": "POS"}
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "system_ident" / "configs" / "three_dof_twin.yml"
)


def _twin():
    plant = SuspensionPlant({"POS": double_pendulum()}, fs=FS)
    return TwinBackend(plant, EXC, RB, fs=FS)


def _watchdog(backend, actuator_sat=1.0, rms_ceiling=1.0, ramp=2.0):
    limits = SafetyLimits(
        actuator_sat=actuator_sat,
        rms_ceiling={"POS": rms_ceiling},
        ramp_down_secs=ramp,
    )
    return Watchdog(backend, limits)


# --- config -------------------------------------------------------------------
def test_limits_from_config():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    limits = SafetyLimits.from_config(config)
    assert limits.actuator_sat == pytest.approx(1e-4)
    assert limits.rms_ceiling["POS"] == pytest.approx(1e-6)
    assert limits.rms_ceiling["YAW"] == pytest.approx(1e-7)
    assert limits.ramp_down_secs == pytest.approx(2.0)


# --- snapshot / restore -------------------------------------------------------
def test_snapshot_restore_round_trip():
    twin = _twin()
    snap = twin.snapshot_state(["X1:EXC_POS"])  # pre-run: no drive
    twin.inject("X1:EXC_POS", np.ones(int(4 * FS)), FS)
    assert np.any(twin.read(["X1:RESP_POS"], 4.0)["X1:RESP_POS"] != 0)

    twin.restore_state(snap)
    np.testing.assert_array_equal(twin.read(["X1:RESP_POS"], 4.0)["X1:RESP_POS"], 0.0)


# --- evaluation (no side effects) ---------------------------------------------
def test_evaluate_flags_saturation_and_rms():
    wd = _watchdog(_twin(), actuator_sat=1.0, rms_ceiling=1.0)
    segment = {
        "X1:EXC_POS": np.array([0.5, -2.0, 0.1]),   # peak 2.0 > 1.0 -> breach
        "X1:RESP_POS": np.full(100, 3.0),           # rms 3.0 > 1.0 -> breach
    }
    report = wd.evaluate(segment)
    assert not report.ok
    assert len(report.breaches) == 2
    assert report.drive_peaks["X1:EXC_POS"] == pytest.approx(2.0)
    assert report.output_rms["POS"] == pytest.approx(3.0)
    assert not wd.aborted  # evaluate must not trigger the handoff


def test_evaluate_ok_within_limits():
    wd = _watchdog(_twin(), actuator_sat=10.0, rms_ceiling=10.0)
    segment = {"X1:EXC_POS": np.array([0.5, -0.4]), "X1:RESP_POS": np.full(50, 0.2)}
    report = wd.evaluate(segment)
    assert report.ok and not report.breaches


# --- check + handoff ----------------------------------------------------------
def test_check_aborts_and_hands_off_on_breach():
    twin = _twin()
    wd = _watchdog(twin, actuator_sat=1.0)
    wd.snapshot(["X1:EXC_POS"])  # capture quiet pre-run state

    twin.inject("X1:EXC_POS", np.full(int(8 * FS), 5.0), FS)  # over saturation
    segment = twin.read(["X1:EXC_POS", "X1:RESP_POS"], 8.0)

    with pytest.raises(SafetyAbort, match="actuator saturation"):
        wd.check(segment)

    assert wd.aborted
    assert "actuator saturation" in wd.abort_reason
    # handoff restored the pre-run (quiet) state
    np.testing.assert_array_equal(twin.read(["X1:RESP_POS"], 8.0)["X1:RESP_POS"], 0.0)


def test_operator_stop_ramps_and_restores():
    twin = _twin()
    wd = _watchdog(twin)
    wd.snapshot(["X1:EXC_POS"])
    twin.inject("X1:EXC_POS", np.ones(int(8 * FS)), FS)

    wd.abort("operator STOP")

    assert wd.aborted and wd.abort_reason == "operator STOP"
    np.testing.assert_array_equal(twin.read(["X1:RESP_POS"], 8.0)["X1:RESP_POS"], 0.0)


def test_abort_is_idempotent():
    twin = _twin()
    wd = _watchdog(twin)
    wd.snapshot(["X1:EXC_POS"])
    wd.abort("first reason")
    wd.abort("second reason")  # no-op
    assert wd.abort_reason == "first reason"
