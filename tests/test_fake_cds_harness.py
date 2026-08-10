"""Self-test for the Stage A fake-transport harness (issue #5).

Per the plan: "the harness is exercised by every test in Stage G. No
standalone assertions needed beyond a self-test that the fake plant's freqz
matches the FRF the loop recovers from it." That's the load-bearing test
here (both excitation modes); the rest are narrow checks that the
fault-injection switches actually do what they claim, since an unverified
fake fault is worse than no fake fault at all.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import scipy.signal as sig

from system_ident.loop import SysIDLoop

from _fake_cds import (
    FakeArbitraryLoop,
    FakeArbitraryStream,
    FakeCDSWorld,
    FakeChannelNotFound,
    install,
)

FS = 256.0
NPERSEG, N_PERIODS = 1024, 8
EXC, SENSOR = "X1:COIL_EXC", "X1:SENSOR_DQ"
# Q=20 @ 1 Hz -> ringdown tau = Q/(pi f0) ~= 6.4 s; >=200 s of settle before the
# analysed record decays the startup transient well past the 1e-8 check below
# (real backends do this via warmup_s; this harness test does it explicitly).
SETTLE_S = 240.0


def _grid():
    f_all = np.fft.rfftfreq(NPERSEG, 1 / FS)
    band = (f_all >= 0.3) & (f_all <= 8.0)
    return band, f_all[band]


def _plant_ba():
    # Q=20 resonance at 1 Hz, matching the spec's S2 measurement setup.
    from system_ident.model import TFModel
    plant = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    return sig.bilinear(plant.num, plant.den, FS)


def _multisine(band, freq):
    from system_ident.excitation import multisine_from_psd
    return multisine_from_psd(np.ones_like(freq), FS, NPERSEG, N_PERIODS, freq, seed=0)


@pytest.fixture(autouse=True)
def _reset_awg_counts():
    FakeArbitraryLoop.reset()
    FakeArbitraryStream.reset()
    yield
    FakeArbitraryLoop.reset()
    FakeArbitraryStream.reset()


def _world_with_plant(testpoint=True):
    world = FakeCDSWorld()
    b, a = _plant_ba()
    world.getdata.add_exc_channel(EXC, rate=FS, testpoint=testpoint)
    world.getdata.add_readback_channel(SENSOR, exc_channel=EXC, plant=(b, a), rate=FS,
                                       testpoint=testpoint)
    return world, (b, a)


# ---------------------------------------------------------------------------
# The load-bearing self-test: fake plant's freqz == FRF the loop recovers.
# ---------------------------------------------------------------------------

def test_loop_mode_freqz_matches_recovered_frf():
    band, freq = _grid()
    world, (b, a) = _world_with_plant()
    drive = _multisine(band, freq)
    start = world.gpstime.now().gps()
    world.open_loop(EXC, drive, rate=FS, start=start)

    with world.getdata.open_stream([EXC, SENSOR]):
        world.getdata([EXC, SENSOR], SETTLE_S)          # discard: let the resonance ring down
        [buf_x, buf_y] = world.getdata([EXC, SENSOR], NPERSEG * N_PERIODS / FS)

    assert buf_x.data.size == buf_y.data.size == NPERSEG * N_PERIODS
    H, H_err, coh = SysIDLoop._estimate_tf_periodic(buf_x.data, buf_y.data, FS, NPERSEG, band)
    _, H_true = sig.freqz(b, a, worN=freq, fs=FS)

    rel_err = np.max(np.abs(H - H_true) / np.abs(H_true))
    assert rel_err < 1e-8, f"loop-mode FRF diverged from freqz truth: {rel_err:.3e}"


def test_stream_mode_freqz_matches_recovered_frf():
    """Same check, driving the excitation through chunked ``append`` calls
    instead of a one-shot tiled ``ArbitraryLoop`` array — #31's stream path."""
    band, freq = _grid()
    world, (b, a) = _world_with_plant()
    drive = _multisine(band, freq)
    # A stream has no implicit looping (unlike ArbitraryLoop) -- tile the
    # periodic drive out to settle+record length and feed it via chunked
    # append(), the way a long-record stream injection actually works (S2.3).
    total_n = int(round((SETTLE_S + NPERSEG * N_PERIODS / FS) * FS))
    tiled = np.resize(drive, total_n)
    handle = world.open_stream(EXC, rate=FS)
    chunk = NPERSEG  # arbitrary chunk size -> exercises multiple append() calls
    for i in range(0, total_n, chunk):
        handle.append(tiled[i:i + chunk])

    with world.getdata.open_stream([EXC, SENSOR]):
        world.getdata([EXC, SENSOR], SETTLE_S)          # discard: let the resonance ring down
        [buf_x, buf_y] = world.getdata([EXC, SENSOR], NPERSEG * N_PERIODS / FS)

    H, H_err, coh = SysIDLoop._estimate_tf_periodic(buf_x.data, buf_y.data, FS, NPERSEG, band)
    _, H_true = sig.freqz(b, a, worN=freq, fs=FS)
    rel_err = np.max(np.abs(H - H_true) / np.abs(H_true))
    assert rel_err < 1e-8, f"stream-mode FRF diverged from freqz truth: {rel_err:.3e}"


def test_quiet_read_with_no_live_injection_is_zero():
    world, _ = _world_with_plant(testpoint=False)
    [buf] = world.getdata([EXC], 1.0)
    assert np.all(buf.data == 0.0)


# ---------------------------------------------------------------------------
# Fault-injection switches (#32) — narrow checks that each one fires.
# ---------------------------------------------------------------------------

def test_adjacent_reads_have_contiguous_start_times():
    world, _ = _world_with_plant(testpoint=False)
    [b1] = world.getdata([EXC], 1.0)
    [b2] = world.getdata([EXC], 1.0)
    assert b2.start_time == b1.start_time + 1


def test_insert_gap_breaks_adjacency():
    world, _ = _world_with_plant(testpoint=False)
    [b1] = world.getdata([EXC], 1.0)
    world.getdata.insert_gap(5.0)
    [b2] = world.getdata([EXC], 1.0)
    assert b2.start_time == b1.start_time + 1 + 5


def test_testpoint_bare_read_succeeds_but_not_after_it_was_streamed_and_closed():
    world, _ = _world_with_plant(testpoint=True)
    world.getdata([EXC], 1.0)              # a bare, self-contained read -> fine (no bracket needed)
    with world.getdata.open_stream([EXC]):
        world.getdata([EXC], 1.0)          # still fine while the stream is open
    with pytest.raises(FakeChannelNotFound):
        world.getdata([EXC], 1.0)          # closed after streaming -> no second fetch


def test_recorded_channel_needs_no_open_stream():
    world, _ = _world_with_plant(testpoint=False)
    world.getdata([EXC], 1.0)              # no open_stream needed -> no raise


def test_missing_channel_and_short_and_nonfinite():
    world, _ = _world_with_plant(testpoint=False)
    world.getdata.make_missing(SENSOR)
    [only_x] = world.getdata([EXC, SENSOR], 1.0)
    assert only_x is not None

    world.getdata.make_short(EXC, by_samples=3)
    [buf] = world.getdata([EXC], 1.0)
    assert buf.data.size == int(FS) - 3

    world.getdata.make_nonfinite(EXC, start=0, stop=5)
    [buf] = world.getdata([EXC], 1.0)
    assert np.all(np.isnan(buf.data[:5])) and np.all(np.isfinite(buf.data[5:]))


def test_raise_on_next_and_rate_change():
    world, _ = _world_with_plant(testpoint=False)
    world.getdata.raise_on_next(EXC, FakeChannelNotFound("gone"))
    with pytest.raises(FakeChannelNotFound):
        world.getdata([EXC], 1.0)
    # the fault is one-shot: the very next call is clean again
    world.getdata([EXC], 1.0)

    world.getdata.change_rate(EXC, 8192.0)
    [buf] = world.getdata([EXC], 1.0)
    assert buf.sample_rate == 8192.0


def test_stream_underrun_raises_on_configured_append():
    world, _ = _world_with_plant(testpoint=False)
    handle = world.open_stream(EXC, rate=FS)
    handle.fail_on_append(2)
    handle.append(np.ones(4))
    with pytest.raises(Exception):
        handle.append(np.ones(4))


def test_loop_stop_before_start_raises():
    world, _ = _world_with_plant(testpoint=False)
    handle = world.open_loop(EXC, np.ones(4), rate=FS, start=world.gpstime.now().gps())
    with pytest.raises(Exception):
        handle.stop(ramptime=1.0)


# ---------------------------------------------------------------------------
# Construction counting (class-level, must be assertable and resettable).
# ---------------------------------------------------------------------------

def test_construction_counts_are_class_level_and_resettable():
    assert FakeArbitraryLoop.n_constructed == 0
    world, _ = _world_with_plant(testpoint=False)
    world.open_loop(EXC, np.ones(4), rate=FS, start=world.gpstime.now().gps())
    assert FakeArbitraryLoop.n_constructed == 1
    assert FakeArbitraryStream.n_constructed == 1   # a Loop IS a Stream underneath
    FakeArbitraryLoop.reset()
    assert FakeArbitraryLoop.n_constructed == 0


# ---------------------------------------------------------------------------
# sys.modules stubbing.
# ---------------------------------------------------------------------------

def test_install_stubs_and_restores_sys_modules():
    for name in ("awg", "cdsutils", "gpstime"):
        assert name not in sys.modules
    with install() as world:
        import awg
        import cdsutils
        import gpstime as gpstime_mod
        h = awg.ArbitraryLoop(EXC, np.ones(4), rate=FS, start=world.gpstime.now().gps())
        assert isinstance(h, FakeArbitraryLoop)
        assert cdsutils.getdata is world.getdata
        assert gpstime_mod.gpstime is world.gpstime
    for name in ("awg", "cdsutils", "gpstime"):
        assert name not in sys.modules
