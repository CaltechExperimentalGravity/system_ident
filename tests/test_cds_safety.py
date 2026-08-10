"""Stage E: safety enforcement -- the per-injection approval token (#17) and
pre-injection amplitude limits (#4), against the Stage A fakes.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.signal as sig

from system_ident.backends.cds import CDSBackend, _default_authorizer
from system_ident.model import TFModel
from system_ident.safety import SafetyAbort, SafetyLimits

from _fake_cds import FakeArbitraryLoop, FakeArbitraryStream, FakeCDSWorld, install

FS = 256.0
NPERSEG, N_PERIODS = 1024, 4
EXC, SENSOR, DRIVE = "X1:COIL_EXC", "X1:SENSOR_DQ", "X1:COIL_MON_DQ"


@pytest.fixture(autouse=True)
def _reset():
    FakeArbitraryLoop.reset()
    FakeArbitraryStream.reset()
    os.environ.setdefault("IFO", "X1")
    yield
    FakeArbitraryLoop.reset()
    FakeArbitraryStream.reset()


def _world_with_plant():
    world = FakeCDSWorld()
    plant = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    b, a = sig.bilinear(plant.num, plant.den, FS)
    world.getdata.add_exc_channel(EXC, rate=FS, testpoint=True)
    world.getdata.add_readback_channel(SENSOR, exc_channel=EXC, plant=(b, a), rate=FS,
                                       testpoint=True)
    world.getdata.add_readback_channel(DRIVE, exc_channel=EXC, plant=([1.0], [1.0]), rate=FS,
                                       testpoint=True)
    return world


def _backend(world, **kw):
    from system_ident.backends.cds_transport import AWGNDSTransport
    t = AWGNDSTransport()
    return CDSBackend(t, exc_channels={EXC: "POS"}, readback_channels={SENSOR: "POS"},
                      drive_channels={"POS": DRIVE}, fs=FS, exc_mode="loop",
                      segment_duration=NPERSEG / FS, n_segments=N_PERIODS, warmup_s=0.0, **kw)


def _one_period():
    return np.sin(2 * np.pi * 1.0 * np.arange(NPERSEG) / FS)


def _drive():
    return np.tile(_one_period(), N_PERIODS)


# --------------------------------------------------------------------------- #
# The per-injection approval token (#17, spec S5)
# --------------------------------------------------------------------------- #

def test_denying_authorizer_means_the_loop_is_never_constructed():
    world = _world_with_plant()
    with install(world):
        be = _backend(world, authorizer=lambda prompt: False)
        be.inject(EXC, _drive(), FS)
        assert FakeArbitraryLoop.n_constructed == 0     # denial happens before staging starts
        with pytest.raises(SafetyAbort):
            be.read([SENSOR], NPERSEG * N_PERIODS / FS)
        # the gate rejected it BEFORE reaching the AWG API -- construction
        # counts as "reached", even if .start() was never called
        assert FakeArbitraryLoop.n_constructed == 0
        assert FakeArbitraryLoop.n_started == 0


def test_authorizer_called_once_per_inject_not_per_read():
    world = _world_with_plant()
    with install(world):
        calls = []

        def authorizer(prompt):
            calls.append(prompt)
            return True

        be = _backend(world, authorizer=authorizer)
        be.inject(EXC, _drive(), FS)
        be.read([SENSOR], NPERSEG * N_PERIODS / FS)
        be.read([SENSOR], NPERSEG * N_PERIODS / FS)      # same generation, cached -> no re-prompt
        be.read([SENSOR], NPERSEG * N_PERIODS / FS)
        assert len(calls) == 1

        be.inject(EXC, _drive(), FS)                     # a NEW inject() -> a NEW prompt
        be.read([SENSOR], NPERSEG * N_PERIODS / FS)
        assert len(calls) == 2


def test_default_authorizer_denies_on_eoferror():
    import builtins
    import sys

    class _TTYStdin:
        def isatty(self):
            return True                  # reach input(), which then raises EOFError

    def raising_input(prompt):
        raise EOFError

    saved_input, saved_stdin = builtins.input, sys.stdin
    builtins.input, sys.stdin = raising_input, _TTYStdin()
    try:
        assert _default_authorizer("prompt: ") is False
    finally:
        builtins.input, sys.stdin = saved_input, saved_stdin


def test_default_authorizer_denies_on_non_tty():
    import sys

    class _NonTTYStdin:
        def isatty(self):
            return False

    saved = sys.stdin
    sys.stdin = _NonTTYStdin()
    try:
        assert _default_authorizer("prompt: ") is False
    finally:
        sys.stdin = saved


def test_new_inject_invalidates_a_previously_denied_token():
    world = _world_with_plant()
    with install(world):
        decisions = iter([False, True])
        be = _backend(world, authorizer=lambda prompt: next(decisions))
        be.inject(EXC, _drive(), FS)
        with pytest.raises(SafetyAbort):
            be.read([SENSOR], NPERSEG * N_PERIODS / FS)
        be.inject(EXC, _drive(), FS)                      # fresh token, fresh prompt
        seg = be.read([SENSOR], NPERSEG * N_PERIODS / FS)   # now approved -> succeeds
        assert seg[SENSOR].size == NPERSEG * N_PERIODS


# --------------------------------------------------------------------------- #
# Pre-injection amplitude limits (#4, spec S5)
# --------------------------------------------------------------------------- #

def test_drive_limits_scale_down_a_small_excess_and_report_the_factor(capsys):
    world = _world_with_plant()
    with install(world):
        limits = SafetyLimits(actuator_sat=1e12, rms_ceiling={"POS": 1e12},
                              max_exc_peak=0.8)             # one_period peak is 1.0 -> needs 0.8x
        be = _backend(world, authorizer=lambda p: True, safety_limits=limits)
        be.inject(EXC, _drive(), FS)
        out = capsys.readouterr().out
        assert "scaling drive" in out
        staged_peak = np.max(np.abs(be._staged[EXC]["array"]))
        assert staged_peak <= 0.8 + 1e-9


def test_drive_limits_raise_safety_abort_past_2x_shrink():
    world = _world_with_plant()
    with install(world):
        limits = SafetyLimits(actuator_sat=1e12, rms_ceiling={"POS": 1e12},
                              max_exc_peak=0.1)             # needs a 10x shrink -- design is wrong
        be = _backend(world, authorizer=lambda p: True, safety_limits=limits)
        with pytest.raises(SafetyAbort):
            be.inject(EXC, _drive(), FS)
        assert FakeArbitraryLoop.n_constructed == 0         # raises before ever staging/starting


def test_drive_limits_are_opt_in_and_off_by_default():
    world = _world_with_plant()
    with install(world):
        be = _backend(world, authorizer=lambda p: True)      # safety_limits=None (default)
        be.inject(EXC, _drive() * 1e9, FS)                    # absurd amplitude, no limits set
        np.testing.assert_allclose(be._staged[EXC]["array"], _drive() * 1e9, rtol=1e-4, atol=1.0)


def test_nonfinite_drive_is_rejected_before_reaching_the_transport():
    world = _world_with_plant()
    with install(world):
        be = _backend(world, authorizer=lambda p: True)
        bad = _drive().copy()
        bad[10] = np.nan
        with pytest.raises(SafetyAbort):
            be.inject(EXC, bad, FS)
        assert FakeArbitraryLoop.n_constructed == 0


# --------------------------------------------------------------------------- #
# Sanity: coherence/data-integrity faults never enter Watchdog breaches
# (S4.3.6 guard rail -- true by construction; regression-guard it explicitly)
# --------------------------------------------------------------------------- #

def test_watchdog_has_no_coherence_logic():
    import inspect
    from system_ident.safety import Watchdog
    src = inspect.getsource(Watchdog.evaluate)
    assert "coh" not in src.lower()
