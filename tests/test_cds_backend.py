"""Stage D: ``CDSBackend`` itself -- construction/pre-flight (#13), the
``read()`` invariants (#14), the read cache (#15), and lifecycle (#16).

FRF-recovery scenarios run against ``TwinTransport`` (fast, no awg/cdsutils
needed); staged-array assertions (untapered loop tiling with
``ramptime==ramp_s``, the tapered stream envelope) run against
``AWGNDSTransport`` + the Stage A fakes, which expose exactly that detail.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
import scipy.signal as sig

from system_ident.backends.cds import CDSBackend
from system_ident.backends.cds_transport import (
    AWGNDSTransport,
    DataIntegrityError,
    TransportUnavailable,
    TwinTransport,
)
from system_ident.excitation import multisine_from_psd
from system_ident.loop import SysIDLoop
from system_ident.model import TFModel
from system_ident.safety import SafetyLimits, Watchdog

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


# --------------------------------------------------------------------------- #
# TwinTransport-backed FRF recovery (both modes)
# --------------------------------------------------------------------------- #

class _Buf:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)


class _MockRTSModel:
    def __init__(self, plant, fs=FS):
        self.sample_rate = float(fs)
        self._b, self._a = sig.bilinear(plant.num, plant.den, fs)
        self._zi = np.zeros(max(len(self._a), len(self._b)) - 1)
        self._pending = None
        self._captured: list = []

    def fetch_later(self, t0, t1, names):
        want = int(round((t1 - t0) * self.sample_rate))
        self._pending = (list(names), want)
        return lambda: self._captured

    def run(self, cycles, excitations=None, excitation_data=None):
        n = int(cycles)
        drive = np.zeros(n)
        if excitations and excitation_data is not None:
            ed = np.asarray(excitation_data, dtype=float)
            for j, ch in enumerate(excitations):
                if ch == EXC:
                    drive = drive + (ed[:, j] if ed.ndim == 2 else ed)[:n]
        y, self._zi = sig.lfilter(self._b, self._a, drive, zi=self._zi)
        probes = {SENSOR: y, DRIVE: drive}
        if self._pending is not None:
            names, want = self._pending
            self._captured = [_Buf(probes.get(nm, np.zeros(n))[:want]) for nm in names]
            self._pending = None


def _backend(exc_mode="stream", warmup_s=0.0, t_ramp=3.0, **kw):
    from system_ident.model import TFModel
    plant = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    transport = TwinTransport(_MockRTSModel(plant, fs=FS))
    be = CDSBackend(
        transport,
        exc_channels={EXC: "POS"}, readback_channels={SENSOR: "POS"},
        drive_channels={"POS": DRIVE},
        fs=FS, exc_mode=exc_mode, t_ramp=t_ramp, warmup_s=warmup_s,
        segment_duration=NPERSEG / FS, n_segments=N_PERIODS, **kw)
    return be, plant


def _grid():
    f_all = np.fft.rfftfreq(NPERSEG, d=1 / FS)
    band = (f_all >= 0.3) & (f_all <= 8.0)
    return band, f_all[band]


def test_stream_mode_recovers_frf():
    band, freq = _grid()
    # Q=20 @ 1 Hz -> ringdown tau ~= 6.4 s; need t >> tau*ln(1/1e-6) ~= 88 s of
    # settle before the 1e-6 check below (same reasoning as the Stage A
    # harness self-test).
    be, plant = _backend(exc_mode="stream", warmup_s=150.0)
    drive = multisine_from_psd(np.ones_like(freq), FS, NPERSEG, N_PERIODS, freq, seed=0)
    be.inject(EXC, drive, FS)
    seg = be.read([DRIVE, SENSOR], NPERSEG * N_PERIODS / FS)
    assert seg[SENSOR].size == seg[DRIVE].size == NPERSEG * N_PERIODS
    H, _, _ = SysIDLoop._estimate_tf_periodic(seg[DRIVE], seg[SENSOR], FS, NPERSEG, band)
    b, a = sig.bilinear(plant.num, plant.den, FS)
    _, H_true = sig.freqz(b, a, worN=freq, fs=FS)
    assert np.max(np.abs(H - H_true) / np.abs(H_true)) < 1e-6


def test_loop_mode_recovers_frf():
    band, freq = _grid()
    be, plant = _backend(exc_mode="loop", warmup_s=150.0)
    drive = multisine_from_psd(np.ones_like(freq), FS, NPERSEG, N_PERIODS, freq, seed=0)
    be.inject(EXC, drive, FS)
    seg = be.read([DRIVE, SENSOR], NPERSEG * N_PERIODS / FS)
    H, _, _ = SysIDLoop._estimate_tf_periodic(seg[DRIVE], seg[SENSOR], FS, NPERSEG, band)
    b, a = sig.bilinear(plant.num, plant.den, FS)
    _, H_true = sig.freqz(b, a, worN=freq, fs=FS)
    assert np.max(np.abs(H - H_true) / np.abs(H_true)) < 1e-6


def test_exc_mode_defaults_to_stream_and_loop_warns():
    be, _ = _backend()
    assert be.exc_mode == "stream"
    plant = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    transport = TwinTransport(_MockRTSModel(plant, fs=FS))
    with pytest.warns(UserWarning, match="LINEAR"):
        CDSBackend(transport, exc_channels={EXC: "POS"}, readback_channels={SENSOR: "POS"},
                  drive_channels={"POS": DRIVE}, fs=FS, exc_mode="loop",
                  segment_duration=NPERSEG / FS, n_segments=N_PERIODS)


def test_drive_channel_is_mandatory():
    plant = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    transport = TwinTransport(_MockRTSModel(plant, fs=FS))
    with pytest.raises(ValueError, match="channels.drive"):
        CDSBackend(transport, exc_channels={EXC: "POS"}, readback_channels={SENSOR: "POS"},
                  drive_channels={}, fs=FS)


def test_nothing_staged_is_the_quiet_read():
    be, _ = _backend()
    seg = be.read([SENSOR], 1.0)
    assert np.all(seg[SENSOR] == 0.0)


def test_missing_channel_raises():
    be, _ = _backend()
    with pytest.raises(TransportUnavailable):
        be.read(["X1:NOT_A_REAL_CHANNEL"], 1.0)


def test_non_integer_second_duration_raises():
    be, _ = _backend()
    with pytest.raises(TransportUnavailable):
        be.read([SENSOR], 1.5)


def test_read_cache_serves_repeat_calls_within_a_generation():
    be, _ = _backend(warmup_s=5.0)
    drive = np.tile(np.sin(2 * np.pi * 1.0 * np.arange(NPERSEG) / FS), N_PERIODS)
    be.inject(EXC, drive, FS)
    dur = NPERSEG * N_PERIODS / FS
    first = be.read([SENSOR], dur)
    assert len(be._cache) == 1
    second = be.read([SENSOR], dur)
    assert len(be._cache) == 1                        # no new fetch -> no new cache entry
    np.testing.assert_array_equal(first[SENSOR], second[SENSOR])


def test_new_inject_bumps_generation_and_invalidates_cache():
    be, _ = _backend(warmup_s=5.0)
    drive = np.tile(np.sin(2 * np.pi * 1.0 * np.arange(NPERSEG) / FS), N_PERIODS)
    be.inject(EXC, drive, FS)
    dur = NPERSEG * N_PERIODS / FS
    be.read([SENSOR], dur)
    gen1 = be._generation
    be.inject(EXC, drive, FS)
    assert be._generation == gen1 + 1
    assert len(be._cache) == 0


def test_ramp_down_is_idempotent():
    be, _ = _backend(warmup_s=1.0)
    drive = np.tile(np.sin(2 * np.pi * 1.0 * np.arange(NPERSEG) / FS), N_PERIODS)
    be.inject(EXC, drive, FS)
    be.read([SENSOR], NPERSEG * N_PERIODS / FS)
    be.ramp_down(EXC, 1.0)
    assert EXC not in be._live
    be.ramp_down(EXC, 1.0)                              # second call: no raise


def test_snapshot_restore_stops_live_excitation():
    be, _ = _backend(warmup_s=1.0)
    drive = np.tile(np.sin(2 * np.pi * 1.0 * np.arange(NPERSEG) / FS), N_PERIODS)
    be.inject(EXC, drive, FS)
    be.read([SENSOR], NPERSEG * N_PERIODS / FS)
    assert be._live
    snap = be.snapshot_state([EXC])
    be.restore_state(snap)
    assert not be._live


# --------------------------------------------------------------------------- #
# Simultaneous-mode staging (spec S9.3)
# --------------------------------------------------------------------------- #

def test_multi_channel_stage_then_assemble():
    """inject() on two channels before any read(); the first read() of the
    generation starts BOTH together (S9.3's stash-then-assemble)."""
    from system_ident.backends.mimo_twin import MIMOTwinBackend  # noqa: F401  (unused; just checking import safety)
    plant = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    world_mdl = _MockRTSModel(plant, fs=FS)
    transport = TwinTransport(world_mdl)
    be = CDSBackend(transport, exc_channels={EXC: "POS", "X1:COIL2_EXC": "PIT"},
                    readback_channels={SENSOR: "POS", "X1:SENSOR2_DQ": "PIT"},
                    drive_channels={"POS": DRIVE, "PIT": "X1:COIL2_MON_DQ"},
                    fs=FS, exc_mode="stream", warmup_s=1.0,
                    segment_duration=NPERSEG / FS, n_segments=N_PERIODS)
    drive = np.tile(np.sin(2 * np.pi * 1.0 * np.arange(NPERSEG) / FS), N_PERIODS)
    be.inject(EXC, drive, FS)
    be.inject("X1:COIL2_EXC", drive, FS)
    assert not be._live                                  # staged, not started
    be.read([SENSOR], NPERSEG * N_PERIODS / FS)
    assert set(be._live) == {EXC, "X1:COIL2_EXC"}         # both started together


# --------------------------------------------------------------------------- #
# Fault classification / retry policy (S4.3.6/S4.3.7)
# --------------------------------------------------------------------------- #

def _world_with_plant():
    world = FakeCDSWorld()
    from system_ident.model import TFModel
    plant = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    b, a = sig.bilinear(plant.num, plant.den, FS)
    world.getdata.add_exc_channel(EXC, rate=FS, testpoint=True)
    world.getdata.add_readback_channel(SENSOR, exc_channel=EXC, plant=(b, a), rate=FS,
                                       testpoint=True)
    world.getdata.add_readback_channel(DRIVE, exc_channel=EXC, plant=([1.0], [1.0]), rate=FS,
                                       testpoint=True)
    return world


def _awg_backend(world, **kw):
    t = AWGNDSTransport()
    be = CDSBackend(t, exc_channels={EXC: "POS"}, readback_channels={SENSOR: "POS"},
                    drive_channels={"POS": DRIVE}, fs=FS, exc_mode="loop",
                    segment_duration=NPERSEG / FS, n_segments=N_PERIODS, warmup_s=0.0, **kw)
    return be


def test_excited_reject_tier_fault_raises_immediately_no_retry():
    world = _world_with_plant()
    with install(world):
        be = _awg_backend(world, passive_read_retries=3)
        be.inject(EXC, np.ones(NPERSEG), FS)
        world.getdata.make_nonfinite(SENSOR, 0, 5)
        with pytest.raises(DataIntegrityError):
            be.read([SENSOR], NPERSEG * N_PERIODS / FS)


def test_passive_reject_tier_fault_retries_then_succeeds():
    world = _world_with_plant()
    with install(world):
        be = _awg_backend(world, passive_read_retries=3)
        world.getdata.make_nonfinite(SENSOR, 0, 5)         # one-shot: fails once, then clean
        seg = be.read([SENSOR], NPERSEG * N_PERIODS / FS)    # nothing staged -> passive
        assert seg[SENSOR].size == NPERSEG * N_PERIODS


# --------------------------------------------------------------------------- #
# Staged-array assertions against the Stage A fakes (#9's acceptance test)
# --------------------------------------------------------------------------- #

def test_loop_mode_stages_untapered_tiling_with_ramptime_eq_ramp_s():
    world = _world_with_plant()
    with install(world):
        be = _awg_backend(world, t_ramp=3.0)
        # Backends always receive an ALREADY-TILED multi-period array from the
        # caller (matching every other backend's convention -- loop.py's
        # _make_drive tiles before calling inject()); CDSBackend does not
        # tile a single period up itself.
        one_period = np.sin(2 * np.pi * 1.0 * np.arange(NPERSEG) / FS)
        tiled_drive = np.tile(one_period, N_PERIODS)
        be.inject(EXC, tiled_drive, FS)
        be.read([SENSOR], NPERSEG * N_PERIODS / FS)
        handle = be._live[EXC]
        assert isinstance(handle, FakeArbitraryLoop)
        assert handle.start_ramptime == 3.0
        periods = handle.array.reshape(N_PERIODS, NPERSEG)
        # untapered: every period has (numerically) identical energy
        e = (periods ** 2).sum(axis=1)
        assert np.allclose(e, e[0], rtol=1e-9)


def test_stream_mode_stages_a_tapered_array_outside_the_analysed_window():
    world = _world_with_plant()
    with install(world):
        t = AWGNDSTransport()
        be = CDSBackend(t, exc_channels={EXC: "POS"}, readback_channels={SENSOR: "POS"},
                        drive_channels={"POS": DRIVE}, fs=FS, exc_mode="stream",
                        t_ramp=3.0, warmup_s=2.0,
                        segment_duration=NPERSEG / FS, n_segments=N_PERIODS)
        drive_one_period = np.sin(2 * np.pi * 1.0 * np.arange(NPERSEG) / FS)
        be.inject(EXC, drive_one_period, FS)
        be.read([SENSOR], NPERSEG * N_PERIODS / FS)
        handle = be._live[EXC]
        assert isinstance(handle, FakeArbitraryStream) and not isinstance(handle, FakeArbitraryLoop)
        commanded = handle.commanded
        n_ramp = int(round(3.0 * FS))
        n_settle = int(round(2.0 * FS))
        # ramp-on IS tapered (first sample near zero)...
        assert abs(commanded[0]) < 0.05 * np.max(np.abs(commanded))
        # ...but the analysed window (segment 3, offset past ramp-on+settle) is flat.
        main_start = n_ramp + n_settle
        main = commanded[main_start: main_start + NPERSEG * N_PERIODS]
        main_periods = main.reshape(N_PERIODS, NPERSEG)
        e_main = (main_periods ** 2).sum(axis=1)
        assert np.allclose(e_main, e_main[0], rtol=1e-9)  # untapered inside the analysed window


# --------------------------------------------------------------------------- #
# Lifecycle: SIGINT stops every started channel exactly once (subprocess)
# --------------------------------------------------------------------------- #

def test_sigint_stops_started_channel_exactly_once():
    import subprocess
    import sys
    code = """
import os, signal, sys, time
sys.path.insert(0, "tests")
os.environ["IFO"] = "X1"
import numpy as np
from system_ident.backends.cds import CDSBackend
from system_ident.backends.cds_transport import AWGNDSTransport
from _fake_cds import FakeArbitraryLoop, install, FakeCDSWorld

world = FakeCDSWorld()
world.getdata.add_exc_channel("X1:COIL_EXC", rate=256.0, testpoint=True)
world.getdata.add_readback_channel("X1:SENSOR_DQ", exc_channel="X1:COIL_EXC",
                                   plant=([1.0], [1.0]), rate=256.0, testpoint=True)
with install(world):
    t = AWGNDSTransport()
    be = CDSBackend(t, exc_channels={"X1:COIL_EXC": "POS"},
                    readback_channels={"X1:SENSOR_DQ": "POS"},
                    drive_channels={"POS": "X1:SENSOR_DQ"}, fs=256.0, exc_mode="loop",
                    segment_duration=4.0, n_segments=1, warmup_s=0.0)
    be.inject("X1:COIL_EXC", np.ones(1024), 256.0)
    be.read(["X1:SENSOR_DQ"], 4.0)
    assert FakeArbitraryLoop.n_started == 1
    try:
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.1)
    except KeyboardInterrupt:
        pass  # the handler's job is _stop_all(); re-raising to the default
              # handler afterward is correct behavior, not a bug to swallow
              # in production -- this test just needs to observe the state.
    # FakeArbitraryLoop does not override abort() -- it increments the base
    # FakeArbitraryStream's own counter, not a Loop-specific one.
    from _fake_cds import FakeArbitraryStream
    assert FakeArbitraryStream.n_aborted == 1
    assert not be._live
    print("OK")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "OK" in result.stdout, result.stderr
