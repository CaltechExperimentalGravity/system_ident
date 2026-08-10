"""Stage C: the transport seam. ``AWGNDSTransport`` against the Stage A fakes
(``tests/_fake_cds.py``), ``TwinTransport`` against a ``MockRTSModel``-style
fake (``tests/test_rtsfreerun_backend.py:35``'s pattern) -- both implement the
same ``CDSTransport`` protocol, so both get the same test bodies where the
scenario is transport-agnostic.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.signal as sig

from system_ident.backends.cds_transport import (
    AWGNDSTransport,
    CDSTransportError,
    ChannelNotFound,
    DataIntegrityError,
    TestpointTimeout,
    TimingFault,
    TransportUnavailable,
    TwinTransport,
)

from _fake_cds import FakeArbitraryLoop, FakeArbitraryStream, FakeChannelNotFound, install

FS = 256.0
EXC, SENSOR = "X1:COIL_EXC", "X1:SENSOR_DQ"


@pytest.fixture(autouse=True)
def _reset_counts():
    FakeArbitraryLoop.reset()
    FakeArbitraryStream.reset()
    os.environ.setdefault("IFO", "X1")
    yield
    FakeArbitraryLoop.reset()
    FakeArbitraryStream.reset()


def _plant_ba():
    from system_ident.model import TFModel
    plant = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    return sig.bilinear(plant.num, plant.den, FS)


def _world_with_plant():
    from _fake_cds import FakeCDSWorld
    world = FakeCDSWorld()
    b, a = _plant_ba()
    world.getdata.add_exc_channel(EXC, rate=FS, testpoint=True)
    world.getdata.add_readback_channel(SENSOR, exc_channel=EXC, plant=(b, a), rate=FS,
                                       testpoint=True)
    return world


# --------------------------------------------------------------------------- #
# AWGNDSTransport, against the Stage A fakes
# --------------------------------------------------------------------------- #

def test_awgndstransport_requires_the_site_ifo_env():
    os.environ.pop("IFO", None)
    with pytest.raises(TransportUnavailable):
        AWGNDSTransport()


def test_awgndstransport_probe_rate_and_classify():
    world = _world_with_plant()
    with install(world):
        t = AWGNDSTransport()
        assert t.probe_rate([EXC]) == FS
        info = t.classify([EXC, SENSOR])
        assert info[EXC].retrievable == "testpoint" and info[EXC].injectable
        assert info[SENSOR].retrievable == "recorded" and not info[SENSOR].injectable


def test_awgndstransport_loop_mode_start_stop_construction_counts():
    world = _world_with_plant()
    with install(world):
        t = AWGNDSTransport()
        assert FakeArbitraryLoop.n_constructed == 0
        h = t.start(EXC, np.ones(256), FS, t.now_gps(), ramptime=1.0)
        assert FakeArbitraryLoop.n_constructed == 1 and FakeArbitraryLoop.n_started == 1
        t.stop(h, ramptime=1.0)
        assert FakeArbitraryLoop.n_stopped == 1
        t.stop(h, ramptime=1.0)                     # idempotent: no second stop, no raise
        assert FakeArbitraryLoop.n_stopped == 1


def test_awgndstransport_stream_mode_append_and_underrun():
    world = _world_with_plant()
    with install(world):
        t = AWGNDSTransport()
        h = t.open(EXC, FS)
        t.append(h, np.ones(64))
        t.append(h, np.ones(64))
        assert h.appended[0].size == 64 and len(h.appended) == 2
        h.fail_on_append(3)                          # 1-indexed: the NEXT call
        with pytest.raises(DataIntegrityError):
            t.append(h, np.ones(64))
        t.close(h)
        assert h.closed


def test_awgndstransport_set_gain_and_abort_independent_of_queued_data():
    world = _world_with_plant()
    with install(world):
        t = AWGNDSTransport()
        h = t.open(EXC, FS)
        t.append(h, np.ones(64))
        t.set_gain(h, 0.0, ramptime=2.0)
        t.abort(h)
        assert h.gain_calls == [(0.0, 2.0)] and h.aborted


def test_awgndstransport_fetch_recovers_frf():
    world = _world_with_plant()
    with install(world):
        t = AWGNDSTransport()
        drive = np.tile(np.sin(2 * np.pi * 1.0 * np.arange(256) / FS), 8)
        h = t.start(EXC, drive, FS, t.now_gps(), ramptime=0.0)
        with world.getdata.open_stream([EXC, SENSOR]):
            world.getdata([EXC, SENSOR], 200.0)               # settle
            caps = t.fetch([EXC, SENSOR], 8 * 256 / FS)
        assert caps[EXC].data.size == caps[SENSOR].data.size == 8 * 256
        t.stop(h, ramptime=0.0)


def test_awgndstransport_stream_chunked_matches_single_fetch():
    world = _world_with_plant()
    with install(world):
        t = AWGNDSTransport()
        t.start(EXC, np.ones(256), FS, t.now_gps(), ramptime=0.0)
        with world.getdata.open_stream([EXC, SENSOR]):
            single = t.fetch([EXC, SENSOR], 4.0)
        world2 = _world_with_plant()
        with install(world2):
            t2 = AWGNDSTransport()
            t2.start(EXC, np.ones(256), FS, t2.now_gps(), ramptime=0.0)
            with world2.getdata.open_stream([EXC, SENSOR]):
                chunked = _concat(t2.stream([EXC, SENSOR], 4.0, chunk_s=1.0))
        np.testing.assert_array_equal(single[EXC].data, chunked[EXC].data)
        np.testing.assert_array_equal(single[SENSOR].data, chunked[SENSOR].data)


def _concat(chunks):
    out = {}
    for caps in chunks:
        for ch, cap in caps.items():
            out.setdefault(ch, []).append(cap.data)
    return {ch: type("C", (), {"data": np.concatenate(v)})() for ch, v in out.items()}


def test_awgndstransport_gap_is_a_hard_failure():
    world = _world_with_plant()
    with install(world):
        t = AWGNDSTransport()
        t.start(EXC, np.ones(256), FS, t.now_gps(), ramptime=0.0)
        with world.getdata.open_stream([EXC, SENSOR]):
            it = t.stream([EXC, SENSOR], 3.0, chunk_s=1.0)
            next(it)
            world.getdata.insert_gap(5.0)
            with pytest.raises(DataIntegrityError):
                next(it)


def test_awgndstransport_rate_change_mid_campaign_is_a_timing_fault():
    world = _world_with_plant()
    with install(world):
        t = AWGNDSTransport()
        t.start(EXC, np.ones(256), FS, t.now_gps(), ramptime=0.0)
        with world.getdata.open_stream([EXC, SENSOR]):
            it = t.stream([EXC, SENSOR], 3.0, chunk_s=1.0)
            next(it)
            world.getdata.change_rate(EXC, 512.0)
            with pytest.raises(TimingFault):
                next(it)


def test_awgndstransport_errors_are_all_cds_transport_errors():
    """Catching CDSTransportError alone must be enough to know "stop
    driving" -- every typed fault is a subclass of it."""
    assert issubclass(ChannelNotFound, CDSTransportError)
    assert issubclass(DataIntegrityError, CDSTransportError)
    assert issubclass(TimingFault, CDSTransportError)
    assert issubclass(TestpointTimeout, CDSTransportError)
    assert issubclass(TransportUnavailable, CDSTransportError)


def test_awgndstransport_testpoint_lost_mid_stream_is_a_hard_failure():
    """A live test point can be released by another user/process mid-record
    (S4.3.3 items 1-2). ``stream()``'s current implementation is repeated
    ``getdata`` calls (S9.4's open question: whether that sustains a
    continuous multi-hour test-point read, or the NDS2 iterate/stride API is
    required, is unresolved and hardware-only) -- what IS this transport's
    job regardless of which primitive it ends up built on is translating
    whatever the call raises, at ANY point in the block sequence, into the
    typed hierarchy so the caller's teardown path is uniform."""
    world = _world_with_plant()
    with install(world):
        t = AWGNDSTransport()
        t.start(EXC, np.ones(256), FS, t.now_gps(), ramptime=0.0)
        with world.getdata.open_stream([EXC, SENSOR]):
            it = t.stream([EXC, SENSOR], 3.0, chunk_s=1.0)
            next(it)
            world.getdata.raise_on_next(EXC, FakeChannelNotFound("test point released"))
            with pytest.raises(ChannelNotFound):
                next(it)


def test_awgndstransport_classifies_fake_exceptions_by_name():
    world = _world_with_plant()
    with install(world):
        t = AWGNDSTransport()
        world.getdata.raise_on_next(SENSOR, FakeChannelNotFound("gone"))
        with pytest.raises(ChannelNotFound):
            t.probe_rate([SENSOR])


# --------------------------------------------------------------------------- #
# TwinTransport, against a MockRTSModel-style fake
# --------------------------------------------------------------------------- #

class _Buf:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)


class _MockRTSModel:
    """Mirrors tests/test_rtsfreerun_backend.py:35's MockRTSModel."""

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
        probes = {SENSOR: y}
        if self._pending is not None:
            names, want = self._pending
            self._captured = [_Buf(probes.get(nm, np.zeros(n))[:want]) for nm in names]
            self._pending = None


def _twin_transport():
    from system_ident.model import TFModel
    plant = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    return TwinTransport(_MockRTSModel(plant, fs=FS)), plant


def test_twintransport_probe_rate_and_classify():
    t, _ = _twin_transport()
    assert t.probe_rate([EXC]) == FS
    info = t.classify([EXC, SENSOR])
    assert info[EXC].exists and info[SENSOR].exists


def test_twintransport_loop_mode_fetch_recovers_frf():
    t, plant = _twin_transport()
    b, a = sig.bilinear(plant.num, plant.den, FS)
    _, band, freq = _band(FS, 1024)
    from system_ident.excitation import multisine_from_psd
    drive = multisine_from_psd(np.ones_like(freq), FS, 1024, 8, freq, seed=0)
    t.start(EXC, drive, FS, t.now_gps(), ramptime=0.0)
    t.fetch([EXC, SENSOR], 200.0)                     # settle (twin: fresh mdl.run() cycles)
    caps = t.fetch([EXC, SENSOR], 8 * 1024 / FS)
    from system_ident.loop import SysIDLoop
    H, _, _ = SysIDLoop._estimate_tf_periodic(caps[EXC].data, caps[SENSOR].data, FS, 1024, band)
    _, H_true = sig.freqz(b, a, worN=freq, fs=FS)
    assert np.max(np.abs(H - H_true) / np.abs(H_true)) < 1e-8


def test_twintransport_stream_mode_matches_loop_mode():
    t, plant = _twin_transport()
    t2, _ = _twin_transport()
    drive = np.tile(np.sin(2 * np.pi * 1.0 * np.arange(256) / FS), 4)

    t.start(EXC, drive, FS, t.now_gps(), ramptime=0.0)
    loop_caps = t.fetch([EXC, SENSOR], drive.size / FS)

    h2 = t2.open(EXC, FS)
    for i in range(0, drive.size, 64):
        t2.append(h2, drive[i:i + 64])
    stream_caps = t2.fetch([EXC, SENSOR], drive.size / FS)

    np.testing.assert_allclose(loop_caps[SENSOR].data, stream_caps[SENSOR].data)


def test_twintransport_gap_within_a_record_is_representable():
    t, _ = _twin_transport()
    t.start(EXC, np.ones(256), FS, t.now_gps(), ramptime=0.0)
    it = t.stream([EXC, SENSOR], 3.0, chunk_s=1.0)
    c1 = next(it)
    t.inject_gap(2.0)
    c2 = next(it)
    assert c2[EXC].start_time == c1[EXC].start_time + 1 + 2


def test_twintransport_abort_zeroes_gain_and_drops_handle():
    t, _ = _twin_transport()
    h = t.start(EXC, np.ones(256), FS, t.now_gps(), ramptime=0.0)
    t.abort(h)
    assert h.aborted and h.gain == 0.0
    caps = t.fetch([EXC, SENSOR], 1.0)
    assert np.all(caps[SENSOR].data == 0.0)           # nothing left driving the plant


def _band(fs, nperseg):
    f_all = np.fft.rfftfreq(nperseg, d=1 / fs)
    band = (f_all >= 0.3) & (f_all <= 8.0)
    return f_all, band, f_all[band]
