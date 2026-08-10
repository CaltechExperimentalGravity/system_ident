"""Fakes for the ``awg`` / ``cdsutils`` / ``gpstime`` CDS libraries.

None of the three is installed on the dev machine, so every CDS test fakes
them via ``sys.modules`` stubs (:func:`install`). Modeled on
``tests/test_rtsfreerun_backend.py:35``'s ``MockRTSModel``: a small
in-process world that drives an injected excitation through a known plant,
so a test can check what comes back rather than trusting a black box.

Stage A, issue #5. Amended by #31 (``ArbitraryStream`` — the default
excitation mode, alongside ``ArbitraryLoop`` as a peer) and #32 (fault
injection: the harness must be able to *cause* every fault in spec S4.3.3
from day one — retrofitting fault injection into happy-path-only fakes is
exactly the rebuild #32 exists to avoid).

None of this hierarchy is ``CDSTransportError`` (spec S4.1) — that
translation is ``AWGNDSTransport``'s job (Stage C, not yet built). These
fakes raise plausible stand-ins for what the *real* ``awg``/``cdsutils``
would raise (``FakeAWGError``, ``FakeNDSError`` and its subclasses); the
transport is what turns those into the typed hierarchy callers see.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import scipy.signal as sig


# ---------------------------------------------------------------------------
# Fake-library exceptions — stand-ins for what the real libraries raise.
# ---------------------------------------------------------------------------

class FakeAWGError(RuntimeError):
    """Stands in for an ``awg``-raised error (slot busy, invalid channel, a
    stop before a start — ``backend_rtcds.py:118-144``'s ``"cannot join
    thread before it is started"`` is reproduced verbatim below)."""


class FakeNDSError(RuntimeError):
    """Stands in for a ``cdsutils``/``nds2``-raised connectivity error."""


class FakeTestpointTimeout(FakeNDSError):
    """Stands in for a test-point-allocation timeout (S4.3.3 item 8)."""


class FakeChannelNotFound(FakeNDSError):
    """Stands in for ``getdata`` naming a channel NDS doesn't know, *and*
    for a test-point re-fetch after its stream has closed (S4.3.1: a test
    point has no "second fetch")."""


# ---------------------------------------------------------------------------
# gpstime
# ---------------------------------------------------------------------------

@dataclass
class _FakeGpsNow:
    _t: float

    def gps(self) -> float:
        return self._t


class FakeGpstime:
    """A controllable GPS clock — no test ever sleeps a real ``start_buffer``.

    Real ``getdata`` blocks in wall-clock time collecting live data; faking
    that would make the suite slow and nondeterministic. Instead every fake
    that consumes time (``getdata``, ``ArbitraryLoop.start``) advances this
    clock synthetically, so "waiting" costs nothing and is fully
    reproducible.
    """

    def __init__(self, start: float = 1_000_000_000.0):
        self._t = float(start)

    def advance(self, secs: float) -> None:
        """Simulate ``secs`` of wall-clock time passing."""
        self._t += float(secs)

    def jump(self, secs: float) -> None:
        """An abrupt GPS step — forward (an NTP correction) or, per S4.3.3's
        "GPS jump, skew and backwards step" note, negative (an
        NTP-disciplined clock can step back; not hypothetical)."""
        self._t += float(secs)

    def now(self) -> _FakeGpsNow:
        return _FakeGpsNow(self._t)


# ---------------------------------------------------------------------------
# awg.ArbitraryStream / awg.ArbitraryLoop
# ---------------------------------------------------------------------------

class FakeArbitraryStream:
    """Fakes ``awg.ArbitraryStream`` (issue #31's default excitation mode).

    Records ``open``/``close``/``abort``/``set_gain(gain, ramptime)`` and,
    critically, the **size and order of every** ``append``, so chunked
    feeding and a simulated **underrun** (a gap in the appended data) are
    both testable off-hardware.

    Construction counts are class-level, and tests must assert on them
    (not merely on ``.start``/``.append`` never having been called): a gate
    that constructs the object and then declines is still a gate that
    reached the AWG API. Call :meth:`reset` between tests (e.g. an autouse
    fixture) so counts don't leak across cases.
    """

    n_constructed = 0
    n_opened = 0
    n_closed = 0
    n_aborted = 0

    def __init__(self, channel: str, rate: float):
        FakeArbitraryStream.n_constructed += 1
        FakeArbitraryStream.n_opened += 1
        self.channel = channel
        self.rate = float(rate)
        self.appended: list[np.ndarray] = []
        self.gain_calls: list[tuple[float, float]] = []
        self.closed = False
        self.aborted = False
        self._underrun_at: int | None = None   # 1-indexed append() call to fail on

    def append(self, data, scale: float = 1.0) -> None:
        if self.closed or self.aborted:
            raise FakeAWGError(f"append on a closed/aborted stream ({self.channel!r})")
        call_no = len(self.appended) + 1
        if self._underrun_at is not None and call_no == self._underrun_at:
            raise FakeAWGError(f"stream underrun on {self.channel!r} (append #{call_no})")
        self.appended.append(np.asarray(data, dtype=float) * float(scale))

    def set_gain(self, gain: float, ramptime: float = 0.0) -> None:
        self.gain_calls.append((float(gain), float(ramptime)))

    def close(self) -> None:
        FakeArbitraryStream.n_closed += 1
        self.closed = True

    def abort(self) -> None:
        FakeArbitraryStream.n_aborted += 1
        self.aborted = True

    def fail_on_append(self, n: int) -> None:
        """Test hook: the ``n``-th ``append`` call (1-indexed) raises,
        simulating a starved stream (GC pause, NFS, network) — #31/S4.3.3."""
        self._underrun_at = n

    @property
    def commanded(self) -> np.ndarray:
        """The concatenation of every successful ``append`` so far — what a
        readback of the drive-monitor channel (``_EXC``) should see."""
        return np.concatenate(self.appended) if self.appended else np.zeros(0)

    @classmethod
    def reset(cls) -> None:
        cls.n_constructed = cls.n_opened = cls.n_closed = cls.n_aborted = 0


class FakeArbitraryLoop(FakeArbitraryStream):
    """Fakes ``awg.ArbitraryLoop``, which **subclasses** ``ArbitraryStream``
    in real ``awg`` — mirrored here (spec S4.1) rather than duplicating
    ``append``/``set_gain``/``close``/``abort``.

    Stages a whole tiled array up front
    (``backend_rtcds.py``'s ``ArbitraryLoop(chan, array, rate=..., start=...)``),
    then ``start``/``stop`` a transport-level linear gain ramp around it
    (S2.2/S2.3, ``cds.exc_mode: loop``) — the peer mode to the default
    ``stream`` construction.
    """

    n_constructed = 0
    n_started = 0
    n_stopped = 0

    def __init__(self, channel: str, array, rate: float, start: float, ramptime: float = 0.0):
        super().__init__(channel, rate)          # a Loop IS a Stream underneath
        FakeArbitraryLoop.n_constructed += 1
        self.array = np.asarray(array, dtype=float)
        self.start_gps = float(start)
        self.started = False
        self.stopped = False
        self.start_ramptime: float | None = None
        self.stop_ramptime: float | None = None

    def start(self, ramptime: float = 0.0, wait: bool = False) -> None:
        if self.started:
            raise FakeAWGError(f"{self.channel!r} already started")
        FakeArbitraryLoop.n_started += 1
        self.started = True
        self.start_ramptime = float(ramptime)

    def stop(self, ramptime: float = 0.0) -> None:
        if not self.started:
            # backend_rtcds.py:118-144, verbatim: a stop before a start joins
            # a worker thread that was never spawned and masks the real error.
            raise FakeAWGError("cannot join thread before it is started")
        FakeArbitraryLoop.n_stopped += 1
        self.stopped = True
        self.stop_ramptime = float(ramptime)

    @property
    def commanded(self) -> np.ndarray:
        return self.array

    @classmethod
    def reset(cls) -> None:
        cls.n_constructed = cls.n_started = cls.n_stopped = 0
        FakeArbitraryStream.reset.__func__(cls)  # also clear the inherited counters


# ---------------------------------------------------------------------------
# The shared world: wires the AWG fakes to a plant, and cdsutils.getdata to
# both — so what comes back from a fake readback is what was actually
# commanded, filtered through a known plant, not an independent fiction.
# ---------------------------------------------------------------------------

@dataclass
class _Buf:
    """Mimics a real NDS buffer: ``.data`` / ``.sample_rate`` / ``.start_time``."""
    data: np.ndarray
    sample_rate: float
    start_time: int


@dataclass
class _ChannelConfig:
    role: str = "noise"          # "exc" | "readback" | "noise"
    exc_channel: str | None = None    # readback role only: which exc drives it
    plant: tuple | None = None        # readback role only: (b, a) discrete filter
    rate: float = 16384.0
    testpoint: bool = False           # True: no re-fetch once its stream closes (S4.3.1)
    noise_floor: float = 0.0


class FakeGetdata:
    """Fakes ``cdsutils.getdata``.

    A single shared clock and channel registry drive every call, so a fake
    plant filtered from a fake excitation is what a readback channel
    actually returns — not a value invented independently of what was
    injected. Amended by #32: every call applies whatever fault switches are
    currently armed, then clears the one-shot ones, so a test arms a fault,
    calls once, and asserts on the result.

    Retrievability (S4.3.1): a **test point** (``testpoint=True``) raises
    :class:`FakeChannelNotFound` unless it is currently inside an
    :meth:`open_stream` block — a recorded/``_DQ`` channel is always
    fetchable. This is what makes the "no second fetch" invariant testable.
    """

    def __init__(self, world: "FakeCDSWorld"):
        self._world = world
        self.channels: dict[str, _ChannelConfig] = {}
        self._open: set[str] = set()
        self._exhausted: set[str] = set()
        self._plant_state: dict[str, np.ndarray] = {}
        self._exc_cursor: dict[str, int] = {}   # samples already consumed, per exc channel

        # one-shot fault switches, armed via the methods below
        self._missing: set[str] = set()
        self._short_by: dict[str, int] = {}
        self._nonfinite: dict[str, slice] = {}
        self._gap_before_next_s: float = 0.0
        self._raise_on: dict[str, BaseException] = {}
        self._rate_override: dict[str, float] = {}

    # -- channel registry -----------------------------------------------
    def add_exc_channel(self, channel: str, rate: float, testpoint: bool = True) -> None:
        self.channels[channel] = _ChannelConfig(role="exc", rate=rate, testpoint=testpoint)

    def add_readback_channel(self, channel: str, exc_channel: str, plant, rate: float,
                             testpoint: bool = True, noise_floor: float = 0.0) -> None:
        """``plant``: an ``(b, a)`` discrete-filter tuple, e.g. from
        ``scipy.signal.bilinear``."""
        self.channels[channel] = _ChannelConfig(
            role="readback", exc_channel=exc_channel, plant=plant, rate=rate,
            testpoint=testpoint, noise_floor=noise_floor)
        self._plant_state[channel] = np.zeros(max(len(plant[0]), len(plant[1])) - 1)

    def add_noise_channel(self, channel: str, rate: float, noise_floor: float,
                          testpoint: bool = False) -> None:
        self.channels[channel] = _ChannelConfig(
            role="noise", rate=rate, testpoint=testpoint, noise_floor=noise_floor)

    # -- fault injection (one-shot unless noted; #32) ---------------------
    def make_missing(self, channel: str) -> None:
        """Next call: ``channel`` is silently absent from the result."""
        self._missing.add(channel)

    def make_short(self, channel: str, by_samples: int) -> None:
        """Next call: ``channel``'s block is ``by_samples`` short."""
        self._short_by[channel] = int(by_samples)

    def make_nonfinite(self, channel: str, start: int = 0, stop: int | None = None) -> None:
        """Next call: ``channel``'s block carries NaNs over ``[start, stop)``."""
        self._nonfinite[channel] = slice(start, stop)

    def insert_gap(self, seconds: float) -> None:
        """Next call: the world clock jumps ``seconds`` extra before the
        block starts, so this block's ``start_time`` is **not** adjacent to
        the previous block's end — the S4.3.2 gap case."""
        self._gap_before_next_s += float(seconds)

    def raise_on_next(self, channel: str, exc: BaseException) -> None:
        """Next call touching ``channel``: raise ``exc`` instead of
        returning data (``TransportUnavailable``/``TestpointLost``/
        ``TestpointTimeout``/``ChannelNotFound``-class faults, S4.3.3)."""
        self._raise_on[channel] = exc

    def change_rate(self, channel: str, new_rate: float) -> None:
        """From the next call onward, ``channel`` reports ``new_rate`` — a
        probed rate changing mid-campaign (S4.3.3 item 9)."""
        self._rate_override[channel] = float(new_rate)
        self.channels[channel].rate = float(new_rate)

    # -- retrievability (S4.3.1) ------------------------------------------
    @contextmanager
    def open_stream(self, channels):
        """A live request that stays open for its duration. A bare call with
        NO ``open_stream`` involvement at all is a normal, self-contained
        read (matching real ``cdsutils.getdata`` usage, e.g.
        ``backend_rtcds.py``'s bare ``getdata(chans, 1)`` probe — there is no
        separate "open" step in the real API). What ``open_stream`` models is
        narrower and specific: once a channel HAS been streamed and the
        block closes, a further call for it raises
        :class:`FakeChannelNotFound` — "no second fetch" of a test point
        (S4.3.1), not "every read needs a bracket"."""
        opened = set(channels) - self._open
        self._open |= opened
        try:
            yield self
        finally:
            self._open -= opened
            self._exhausted |= opened

    # -- the call ----------------------------------------------------------
    def __call__(self, channels, duration: float) -> list[_Buf]:
        for ch in channels:
            if ch not in self.channels:
                raise FakeChannelNotFound(f"unknown channel {ch!r}")
            exc = self._raise_on.pop(ch, None)
            if exc is not None:
                raise exc
            cfg = self.channels[ch]
            if cfg.testpoint and ch in self._exhausted and ch not in self._open:
                raise FakeChannelNotFound(
                    f"{ch!r} is a test point already streamed and closed "
                    "(a test point cannot be re-fetched, spec S4.3.1)")

        if self._gap_before_next_s:
            self._world.gpstime.advance(self._gap_before_next_s)
            self._gap_before_next_s = 0.0

        start_time = int(self._world.gpstime.now().gps())
        bufs: list[_Buf] = []
        for ch in channels:
            cfg = self.channels[ch]
            n = int(round(duration * cfg.rate))
            data = self._synthesize(ch, cfg, n)
            if ch in self._nonfinite:
                sl = self._nonfinite.pop(ch)
                data = data.copy()
                data[sl] = np.nan
            if ch in self._short_by:
                data = data[: max(0, len(data) - self._short_by.pop(ch))]
            bufs.append(_Buf(data=data, sample_rate=cfg.rate, start_time=start_time))

        self._world.gpstime.advance(duration)
        return [b for ch, b in zip(channels, bufs) if ch not in self._missing]

    # -- internals -----------------------------------------------------------
    def _synthesize(self, ch: str, cfg: _ChannelConfig, n: int) -> np.ndarray:
        if cfg.role == "exc":
            return self._exc_samples(ch, n)
        if cfg.role == "readback":
            drive = self._exc_samples(cfg.exc_channel, n, cursor_key=ch)
            b, a = cfg.plant
            y, zi = sig.lfilter(b, a, drive, zi=self._plant_state[ch])
            self._plant_state[ch] = zi
            if cfg.noise_floor:
                y = y + self._world.rng.normal(scale=cfg.noise_floor, size=n)
            return y
        # pure noise / disturbance channel
        return self._world.rng.normal(scale=cfg.noise_floor, size=n)

    def _exc_samples(self, exc_channel: str, n: int, cursor_key: str | None = None) -> np.ndarray:
        """The next ``n`` samples of whatever is currently commanded on
        ``exc_channel`` — from a live ``FakeArbitraryLoop`` (periodic tiling)
        or ``FakeArbitraryStream`` (the appended-so-far buffer), continuing
        from where the last read for ``cursor_key`` (default: the exc
        channel's own cursor) left off. No live injection -> zeros, matching
        ``backend_rtcds.py``'s ``excitation is None`` quiet read."""
        key = cursor_key or exc_channel
        handle = self._world.handles.get(exc_channel)
        cursor = self._exc_cursor.get(key, 0)
        if handle is None:
            self._exc_cursor[key] = cursor + n
            return np.zeros(n)
        commanded = handle.commanded
        if commanded.size == 0:
            out = np.zeros(n)
        elif isinstance(handle, FakeArbitraryLoop):
            idx = (cursor + np.arange(n)) % commanded.size
            out = commanded[idx]
        else:
            end = min(cursor + n, commanded.size)
            out = np.zeros(n)
            if end > cursor:
                out[: end - cursor] = commanded[cursor:end]
        self._exc_cursor[key] = cursor + n
        return out


@dataclass
class FakeCDSWorld:
    """Owns the shared clock, RNG and live-handle registry that
    :class:`FakeGetdata` and the AWG fakes are wired against."""

    gpstime: FakeGpstime = field(default_factory=FakeGpstime)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    handles: dict[str, FakeArbitraryStream] = field(default_factory=dict)
    getdata: FakeGetdata = field(init=False)

    def __post_init__(self) -> None:
        self.getdata = FakeGetdata(self)

    # -- registering live handles -----------------------------------------
    def open_loop(self, channel: str, array, rate: float, start: float,
                 ramptime: float = 0.0) -> FakeArbitraryLoop:
        h = FakeArbitraryLoop(channel, array, rate, start, ramptime)
        self.handles[channel] = h
        return h

    def open_stream(self, channel: str, rate: float) -> FakeArbitraryStream:
        h = FakeArbitraryStream(channel, rate)
        self.handles[channel] = h
        return h

    def clear_handle(self, channel: str) -> None:
        self.handles.pop(channel, None)


# ---------------------------------------------------------------------------
# sys.modules installation
# ---------------------------------------------------------------------------

def _make_awg_module(world: FakeCDSWorld) -> types.ModuleType:
    mod = types.ModuleType("awg")

    def _loop_ctor(channel, array, rate, start, ramptime=0.0):
        h = FakeArbitraryLoop(channel, array, rate, start, ramptime)
        world.handles[channel] = h
        return h

    def _stream_ctor(channel, rate):
        h = FakeArbitraryStream(channel, rate)
        world.handles[channel] = h
        return h

    mod.ArbitraryLoop = _loop_ctor
    mod.ArbitraryStream = _stream_ctor
    return mod


def _make_cdsutils_module(world: FakeCDSWorld) -> types.ModuleType:
    mod = types.ModuleType("cdsutils")
    mod.getdata = world.getdata
    return mod


def _make_gpstime_module(world: FakeCDSWorld) -> types.ModuleType:
    mod = types.ModuleType("gpstime")
    mod.gpstime = world.gpstime
    return mod


@contextmanager
def install(world: FakeCDSWorld | None = None):
    """Stub ``sys.modules['awg'/'cdsutils'/'gpstime']`` for the duration of
    the ``with`` block, so ``AWGNDSTransport``'s lazy ``import awg`` (etc.)
    resolves to these fakes instead of failing — none of the three is
    installed on the dev machine. Yields the :class:`FakeCDSWorld`.

    Restores whatever was in ``sys.modules`` before, so this never leaks
    into a test that didn't ask for it.
    """
    world = world or FakeCDSWorld()
    names = ("awg", "cdsutils", "gpstime")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["awg"] = _make_awg_module(world)
    sys.modules["cdsutils"] = _make_cdsutils_module(world)
    sys.modules["gpstime"] = _make_gpstime_module(world)
    try:
        yield world
    finally:
        for name in names:
            if saved[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved[name]
        FakeArbitraryLoop.reset()
        FakeArbitraryStream.reset()
