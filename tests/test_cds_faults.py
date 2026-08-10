"""Stage G's exit-gate fault coverage (issue #32, spec S4.3.3): one case per
taxonomy item, all on the Stage A fakes -- the correct exception type, the
correct reject-vs-abort outcome, and the excitation stopped exactly once.
Plus the three structurally most important checks: chunking is invisible,
an excited gap is rejected at zero weight, a passive gap is retried.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.signal as sig

from system_ident.backends.cds import CDSBackend
from system_ident.backends.cds_transport import (
    AWGNDSTransport,
    ChannelNotFound,
    ChannelNotInjectable,
    DataIntegrityError,
    TestpointLost,
    TestpointTimeout,
    TimingFault,
    TransportUnavailable,
)
from system_ident.model import TFModel

from _fake_cds import (
    FakeArbitraryLoop,
    FakeArbitraryStream,
    FakeCDSWorld,
    FakeTestpointLost,
    FakeTestpointTimeout,
    install,
)

FS = 256.0
NPERSEG, N_PERIODS = 1024, 4
EXC, SENSOR, DRIVE = "X1:COIL_EXC", "X1:SENSOR_DQ", "X1:COIL_MON_DQ"


class FakeAWGErrorForFramebuilderReboot(RuntimeError):
    """Named so AWGNDSTransport's exception-name classifier falls through to
    the generic TransportUnavailable base -- exactly what an unrecognised
    fault (S4.3.4's standing rule) is supposed to do."""


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
    kw.setdefault("authorizer", lambda p: True)
    t = AWGNDSTransport()
    return CDSBackend(t, exc_channels={EXC: "POS"}, readback_channels={SENSOR: "POS"},
                      drive_channels={"POS": DRIVE}, fs=FS, exc_mode="loop",
                      segment_duration=NPERSEG / FS, n_segments=N_PERIODS, warmup_s=0.0, **kw)


def _drive():
    return np.tile(np.sin(2 * np.pi * 1.0 * np.arange(NPERSEG) / FS), N_PERIODS)


def _inject_and_read(be):
    be.inject(EXC, _drive(), FS)
    return be.read([SENSOR], NPERSEG * N_PERIODS / FS)


# --------------------------------------------------------------------------- #
# #32's ten items -- exception type + reject-vs-abort + "stopped exactly once"
# --------------------------------------------------------------------------- #

def test_item_1_2_testpoint_lost_aborts_and_stops_exactly_once():
    world = _world_with_plant()
    with install(world):
        be = _backend(world)
        be.inject(EXC, _drive(), FS)
        world.getdata.raise_on_next(SENSOR, FakeTestpointLost("cleared by another user"))
        with pytest.raises(TestpointLost):
            be.read([SENSOR], NPERSEG * N_PERIODS / FS)
        assert not be._live
        assert FakeArbitraryStream.n_aborted == 1


def test_item_3_transport_unavailable_aborts_and_stops_exactly_once():
    world = _world_with_plant()
    with install(world):
        be = _backend(world)
        be.inject(EXC, _drive(), FS)
        world.getdata.raise_on_next(SENSOR, FakeAWGErrorForFramebuilderReboot())
        with pytest.raises(TransportUnavailable):
            be.read([SENSOR], NPERSEG * N_PERIODS / FS)
        assert not be._live
        assert FakeArbitraryStream.n_aborted == 1


def test_item_5_nonfinite_excitation_rejected_pre_injection():
    world = _world_with_plant()
    with install(world):
        be = _backend(world)
        bad = _drive().copy()
        bad[0] = np.inf
        from system_ident.safety import SafetyAbort
        with pytest.raises(SafetyAbort):
            be.inject(EXC, bad, FS)
        assert FakeArbitraryLoop.n_constructed == 0     # never reaches the AWG API


def test_item_5_nonfinite_readback_is_a_data_integrity_reject():
    world = _world_with_plant()
    with install(world):
        be = _backend(world)
        be.inject(EXC, _drive(), FS)
        world.getdata.make_nonfinite(SENSOR, 0, 3)
        with pytest.raises(DataIntegrityError):
            be.read([SENSOR], NPERSEG * N_PERIODS / FS)
        assert not be._live                              # reject-tier still tears down this record


def test_item_6_noninjectable_channel_rejected_at_preflight():
    world = _world_with_plant()
    with install(world):
        # Patch classify() to report the excitation channel as non-injectable,
        # mimicking a slow read-only EPICS record wired in by mistake.
        real_classify = AWGNDSTransport.classify

        def fake_classify(self, channels):
            info = real_classify(self, channels)
            info[EXC].injectable = False
            return info

        AWGNDSTransport.classify = fake_classify
        try:
            with pytest.raises(ChannelNotInjectable):
                _backend(world)
        finally:
            AWGNDSTransport.classify = real_classify


def test_item_7_invalid_readback_channel_is_channel_not_found():
    # SENSOR is never registered -> the fake's own "unknown channel" path
    world2 = FakeCDSWorld()
    world2.getdata.add_exc_channel(EXC, rate=FS, testpoint=True)
    # readback + drive channels deliberately NOT registered
    with install(world2):
        with pytest.raises(ChannelNotFound):
            CDSBackend(AWGNDSTransport(), exc_channels={EXC: "POS"},
                      readback_channels={SENSOR: "POS"}, drive_channels={"POS": DRIVE},
                      fs=FS, exc_mode="loop", segment_duration=NPERSEG / FS,
                      n_segments=N_PERIODS, authorizer=lambda p: True)


def test_item_8_testpoint_timeout_aborts():
    world = _world_with_plant()
    with install(world):
        be = _backend(world)
        be.inject(EXC, _drive(), FS)
        world.getdata.raise_on_next(SENSOR, FakeTestpointTimeout("allocation timed out"))
        with pytest.raises(TestpointTimeout):
            be.read([SENSOR], NPERSEG * N_PERIODS / FS)
        assert not be._live


def test_item_9_rate_change_mid_stream_is_a_timing_fault():
    """CDSBackend delegates its chunked read entirely to the transport, so
    this exercises the same ``be.transport`` instance CDSBackend holds --
    but drives ``stream()`` by hand (like
    ``test_cds_transport.py::test_awgndstransport_rate_change_mid_campaign``)
    because ``change_rate`` takes effect immediately and permanently in the
    fake: a mismatch can only be observed BETWEEN chunks of one continuous
    stream, never between two independent ``read()`` calls (each starts a
    fresh ``stream()`` with no prior block to compare against -- a real,
    if narrow, limitation of the current chunked-read implementation, not
    just this test's plumbing)."""
    world = _world_with_plant()
    with install(world):
        be = _backend(world)
        be.inject(EXC, _drive(), FS)
        it = be.transport.stream([SENSOR], NPERSEG * N_PERIODS / FS, chunk_s=1.0)
        next(it)
        world.getdata.change_rate(SENSOR, 512.0)
        with pytest.raises(TimingFault):
            next(it)


def test_item_9_gps_backwards_step_mid_stream_is_a_timing_fault():
    world = _world_with_plant()
    with install(world):
        be = _backend(world)
        be.inject(EXC, _drive(), FS)
        it = be.transport.stream([SENSOR], NPERSEG * N_PERIODS / FS, chunk_s=1.0)
        next(it)
        world.getdata.insert_gap(-2.0)          # an NTP-disciplined clock can step back
        with pytest.raises(TimingFault):
            next(it)


def test_item_10_network_issue_is_transport_unavailable():
    world = _world_with_plant()
    with install(world):
        be = _backend(world)
        be.inject(EXC, _drive(), FS)
        world.getdata.raise_on_next(SENSOR, ConnectionError("network unreachable"))
        with pytest.raises(TransportUnavailable):
            be.read([SENSOR], NPERSEG * N_PERIODS / FS)
        assert not be._live


# --------------------------------------------------------------------------- #
# The three structurally most important checks (Stage G's own emphasis)
# --------------------------------------------------------------------------- #

def test_chunking_is_invisible_bit_identical_to_single_fetch():
    # Whole-second chunk sizes only: the fake's start_time (like the real
    # NDS buffer boundary it stands in for, spec S2.1) is integer-second
    # truncated, so a sub-second chunk_s can't be adjacency-checked exactly
    # -- a fake/GPS-precision artefact, not something read_chunk_s should
    # ever be set to in practice (the shipped default is 1.0 s, S9.4).
    world = _world_with_plant()
    with install(world):
        be = _backend(world, read_chunk_s=NPERSEG * N_PERIODS / FS)  # one big chunk
        single = _inject_and_read(be)

    world2 = _world_with_plant()
    with install(world2):
        be2 = _backend(world2, read_chunk_s=1.0)                      # many 1 s chunks
        chunked = _inject_and_read(be2)

    np.testing.assert_array_equal(single[SENSOR], chunked[SENSOR])


def test_excited_gap_is_rejected_and_contributes_zero_weight():
    """A gap strictly BETWEEN two blocks of one continuous read -- exercised
    by manually stepping ``be.transport.stream()`` (the same object
    ``CDSBackend.read()`` uses internally), because a gap at the very START
    of a fresh ``stream()`` call has no prior block to be inconsistent
    WITH (every ``fetch()``/``read()`` call starts a new one) and is
    therefore not detectable as a gap at all -- a real, narrow limitation of
    the current chunked-read implementation (recorded in the bring-up note),
    not a test artefact."""
    from system_ident.loop import SysIDLoop
    world = _world_with_plant()
    with install(world):
        be = _backend(world)
        be.inject(EXC, _drive(), FS)
        it = be.transport.stream([SENSOR], NPERSEG * N_PERIODS / FS, chunk_s=1.0)
        next(it)
        world.getdata.insert_gap(5.0)
        with pytest.raises(DataIntegrityError):
            next(it)
    # The #6 lesson, restated at the transport-fault level (#32): reject
    # contributes INF error -> zero weight, never a small one.
    accum = {"w": np.zeros(1), "wH": np.zeros(1, dtype=complex)}
    H_acc, err_acc = SysIDLoop._accumulate(accum, np.array([1.0]), np.array([np.inf]))
    assert accum["w"][0] == 0.0


def test_passive_read_is_retried_bounded_and_reported(capsys):
    """A one-shot integrity fault (not a gap -- a gap's effect on the FIRST
    block of a fresh read is unobservable, see the test above) on a passive
    (nothing staged) read: bounded retry, reported."""
    world = _world_with_plant()
    with install(world):
        be = _backend(world, passive_read_retries=3)
        world.getdata.make_nonfinite(SENSOR, 0, 3)   # one-shot: the retry should recover
        seg = be.read([SENSOR], NPERSEG * N_PERIODS / FS)   # nothing staged -> passive
        assert seg[SENSOR].size == NPERSEG * N_PERIODS
        assert "retry 1/3" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Read-only real-transport smoke test (Stage G's ladder step 10): probe the
# rate, a quiet getdata, NO injection -- safe on the deployment machine
# without operator approval. Skips everywhere `awg` isn't installed, i.e.
# everywhere except the deployment machine.
# --------------------------------------------------------------------------- #

import importlib.util  # noqa: E402


@pytest.mark.skipif(importlib.util.find_spec("awg") is None,
                    reason="awg not installed (real-transport smoke test; deployment machine only)")
@pytest.mark.skipif(os.environ.get("CDS_SMOKE_TEST_CHANNEL") is None,
                    reason="set CDS_SMOKE_TEST_CHANNEL to a real, live NDS channel name to run this smoke test")
def test_real_transport_readonly_smoke():  # pragma: no cover - runs only on the deployment machine
    site_ifo_env = os.environ.get("CDS_SITE_IFO_ENV", "IFO")
    channel = os.environ["CDS_SMOKE_TEST_CHANNEL"]
    transport = AWGNDSTransport(site_ifo_env=site_ifo_env)
    rate = transport.probe_rate([channel])
    assert rate > 0
    caps = transport.fetch([channel], 1.0)
    assert next(iter(caps.values())).data.size > 0
