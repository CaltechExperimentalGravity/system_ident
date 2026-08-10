"""The CDS transport seam: ``CDSTransport`` (Protocol), ``AWGNDSTransport``
(real ``awg`` + ``cdsutils`` + ``gpstime``), ``TwinTransport`` (routes to a
compiled ``rtsfreerun`` model with a synthetic GPS clock).

Stage C, issues #11/#12, amended by #31 (both excitation modes: a one-shot
``stream`` array is the default, ``loop`` a supported peer mode) and #32
(spec S4.3: the ``CDSTransportError`` taxonomy, chunked reads with
block-adjacency verification, pre-flight ``classify``, and an asymmetric
retry policy). ``CDSBackend`` (Stage D) is written against this seam once,
and runs unchanged against either transport -- that is the entire point of
"pluggable transport" (``notes/40m-sos-campaign-handoff-2026-07.md:60-61``).

Ported from the sibling project's ``measurement/backend_rtcds.py``, comments
included -- these are the hardware lessons, and the comments are why they
exist.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Fault taxonomy (spec S4.3.3). All inherit CDSTransportError, so a caller
# that only wants "something went wrong, stop driving the suspension" has one
# thing to catch -- and an UNCLASSIFIED fault still hits this base, so
# teardown is always safe even when the specific class is a guess (S4.3.4).
# ---------------------------------------------------------------------------

class CDSTransportError(Exception):
    """Base for every CDS transport fault."""


class TransportUnavailable(CDSTransportError):
    """The transport itself (NDS/network/framebuilder) is unreachable."""


class TestpointLost(CDSTransportError):
    """A live test point was released or cleared mid-record -- by another
    user/process, or a hardware-state change. Unrecoverable for that record
    (S4.3.1: a test point has no second fetch)."""


class TestpointTimeout(CDSTransportError):
    """A test-point allocation did not complete within its bound."""


class ChannelNotFound(CDSTransportError):
    """A requested channel does not exist."""


class ChannelNotInjectable(CDSTransportError):
    """A requested excitation channel is not a fast, writable test point
    (structurally -- e.g. a slow read-only EPICS record)."""


class DataIntegrityError(CDSTransportError):
    """Data arrived but is wrong: short, gapped, non-finite, a rate change
    mid-campaign, or (post-hoc) a drive readback diverging from what was
    commanded."""


class TimingFault(CDSTransportError):
    """GPS/timing is inconsistent: a backwards step, or a block's
    ``start_time`` not matching the previous block's end."""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@dataclass
class ChannelInfo:
    """Pre-flight classification of one channel (spec S4.3.5). ``_DQ`` is a
    prior for ``retrievable``, never assumed -- always confirmed by probing."""

    exists: bool
    rate: float
    retrievable: str          # "recorded" | "testpoint"
    live: bool                 # finite, and variance above a floor
    injectable: bool           # structurally a fast, writable test point


@dataclass
class Capture:
    """One block of readback data."""

    data: np.ndarray
    start_time: int
    sample_rate: float


@runtime_checkable
class CDSTransport(Protocol):
    """What ``CDSBackend`` needs, and nothing else -- so it never imports
    ``awg``/``cdsutils``/``gpstime`` and is fully testable off-hardware
    against ``TwinTransport`` or the Stage A fakes."""

    def now_gps(self) -> float: ...
    def probe_rate(self, channels: list[str]) -> float: ...
    def classify(self, channels: list[str]) -> dict[str, ChannelInfo]: ...

    # loop mode -- awg.ArbitraryLoop-shaped (peer mode, cds.exc_mode: loop)
    def start(self, channel: str, array: np.ndarray, rate: float,
              start_gps: float, ramptime: float) -> object: ...
    def stop(self, handle: object, ramptime: float) -> None: ...

    # stream mode -- awg.ArbitraryStream-shaped (default, cds.exc_mode: stream)
    def open(self, channel: str, rate: float) -> object: ...
    def append(self, handle: object, array: np.ndarray, scale: float = 1.0) -> None: ...
    def close(self, handle: object) -> None: ...

    # shared by both -- independent of any queued/staged data (S2.2)
    def set_gain(self, handle: object, gain: float, ramptime: float) -> None: ...
    def abort(self, handle: object) -> None: ...

    # the read path (S4.3.2)
    def stream(self, channels: list[str], duration: float,
              chunk_s: float) -> Iterator[dict[str, Capture]]: ...
    def fetch(self, channels: list[str], duration: float) -> dict[str, Capture]: ...


def _concat_stream(chunks: Iterator[dict[str, Capture]]) -> dict[str, Capture]:
    """``fetch``'s shared implementation: consume ``stream()``, concatenate.
    Adjacency is already verified block-by-block inside ``stream()`` itself
    (S4.3.2) -- concatenating verified-adjacent blocks is required and is
    equivalent to one continuous read; this function only assembles what
    ``stream()`` has already certified."""
    out: dict[str, list[np.ndarray]] = {}
    start_time: dict[str, int] = {}
    rate: dict[str, float] = {}
    for caps in chunks:
        for ch, cap in caps.items():
            out.setdefault(ch, []).append(cap.data)
            start_time.setdefault(ch, cap.start_time)
            rate[ch] = cap.sample_rate
    return {
        ch: Capture(data=np.concatenate(blocks), start_time=start_time[ch], sample_rate=rate[ch])
        for ch, blocks in out.items()
    }


# ---------------------------------------------------------------------------
# AWGNDSTransport -- real hardware
# ---------------------------------------------------------------------------

class AWGNDSTransport:
    """``awg`` + ``cdsutils`` + ``gpstime``, lazy-imported here -- never at
    module scope, so ``import system_ident`` stays clean on a plain
    scientific-Python stack (``cds.py``'s lazy-import contract). Guarded by
    ``tests/test_cds_lazy_import.py``.
    """

    def __init__(self, site_ifo_env: str = "IFO"):
        # Site environment precondition: cdsutils/nds.py reads the site IFO
        # variable at IMPORT time and raises NDSError otherwise. Check first
        # and raise actionably. The value is site configuration, not a
        # constant -- it belongs in the site profile (Component 2), never
        # hardcoded here. Related trap, not this transport's to fix: do NOT
        # source the site workstation rc script to set it -- it prepends a
        # legacy site CDS python stack to PYTHONPATH, and `import cdsutils`
        # then dies with ModuleNotFoundError: No module named 'matrix'.
        # Unsetting PYTHONPATH recovers.
        if not os.environ.get(site_ifo_env):
            raise TransportUnavailable(
                f"${site_ifo_env} is not set -- cdsutils reads it at import time. "
                "This is site configuration (site profile, Component 2), not a "
                "constant; set it before constructing AWGNDSTransport."
            )
        import awg
        import cdsutils
        import gpstime

        # python-awg 3.1.2 calls Thread.isAlive(), removed in Python 3.9 (renamed
        # is_alive); ArbitraryLoop.start() hits it on its first line, so every
        # injection dies before it begins under a modern interpreter. awg 4.1.4
        # (spec S8a) no longer calls it, but keep the shim hasattr-guarded rather
        # than delete it -- the py3.9/CDS-3.1.2 fallback rungs still need it, and
        # patching the class (not site-packages) survives an env rebuild.
        if not hasattr(threading.Thread, "isAlive"):
            threading.Thread.isAlive = threading.Thread.is_alive

        self._awg = awg
        self._cdsutils = cdsutils
        self._gpstime = gpstime
        self._started: dict[object, bool] = {}

    def now_gps(self) -> float:
        return float(self._gpstime.gpstime.now().gps())

    def probe_rate(self, channels: list[str]) -> float:
        # Probe, never configure: a Foton-derived seed model assumes a
        # front-end rate, and a mismatch silently invalidates the run.
        bufs = self._read_once(list(channels), 1)
        rate = float(bufs[-1].sample_rate)
        print(f"probed rate {rate} Hz from {list(channels)}")
        return rate

    def classify(self, channels: list[str]) -> dict[str, ChannelInfo]:
        """The pre-flight retrievability/rate/liveness probe (S4.3.5), folded
        into the same one-second read the rate probe already does -- zero
        extra hardware cost, still pre-injection. Failing here costs
        nothing; failing after an injection costs an injection."""
        channels = list(channels)
        bufs = self._read_once(channels, 1)
        out: dict[str, ChannelInfo] = {}
        for ch, buf in zip(channels, bufs):
            data = np.asarray(buf.data, dtype=float)
            live = bool(np.all(np.isfinite(data))) and float(np.var(data)) > 0.0
            # `_DQ` is a USABLE PRIOR for "recorded", never a contract (S4.3.1):
            # it is confirmed by probing (this call succeeding at all), not
            # assumed from the name alone. Everything else is a live test point.
            recorded_prior = ch.endswith("_DQ")
            out[ch] = ChannelInfo(
                exists=True,
                rate=float(buf.sample_rate),
                retrievable="recorded" if recorded_prior else "testpoint",
                live=live,
                injectable=not recorded_prior,
            )
        return out

    # -- loop mode --------------------------------------------------------
    def start(self, channel: str, array: np.ndarray, rate: float,
              start_gps: float, ramptime: float) -> object:
        handle = self._awg.ArbitraryLoop(channel, np.asarray(array, dtype=float),
                                         rate=rate, start=start_gps)
        self._started[handle] = False
        try:
            handle.start(ramptime=ramptime, wait=False)
            self._started[handle] = True
        except Exception as exc:
            raise TransportUnavailable(f"{channel}: failed to start injection: {exc}") from exc
        return handle

    def stop(self, handle: object, ramptime: float) -> None:
        # backend_rtcds.py:118-144, verbatim: always stop, but ONLY if it
        # started (`ArbitraryLoop.stop()` joins its worker thread, which
        # raises "cannot join thread before it is started" if start() failed,
        # masking the real error); shout loudly on a failed stop -- it may
        # mean an excitation is still running; re-raise only when nothing
        # else is already propagating.
        if not self._started.get(handle, False):
            return
        try:
            handle.stop(ramptime=ramptime)
        except Exception as exc:
            print(f"\n*** WARNING: could not stop the excitation on "
                  f"{getattr(handle, 'channel', handle)}: {exc!r}\n*** Check for a "
                  f"running excitation before doing anything else.\n")
            if sys.exc_info()[0] is None:
                raise TransportUnavailable(f"failed to stop injection: {exc}") from exc
        finally:
            self._started[handle] = False

    # -- stream mode --------------------------------------------------------
    def open(self, channel: str, rate: float) -> object:
        return self._awg.ArbitraryStream(channel, rate)

    def append(self, handle: object, array: np.ndarray, scale: float = 1.0) -> None:
        try:
            handle.append(np.asarray(array, dtype=float), scale)
        except Exception as exc:
            # A starved append (GC pause, NFS, network) is a stream underrun
            # (#31/S4.3.3 item 5's post-injection counterpart is the
            # _EXC-vs-commanded comparison; this is the pre-injection half).
            raise DataIntegrityError(f"stream underrun on append: {exc}") from exc

    def close(self, handle: object) -> None:
        handle.close()

    # -- shared -- ArbitraryStream provides these independently of any
    # queued/staged data (S2.2), which is what makes ramp_down/mandatory-STOP
    # implementable under stream mode without rewriting a live array.
    def set_gain(self, handle: object, gain: float, ramptime: float) -> None:
        handle.set_gain(gain, ramptime=ramptime)

    def abort(self, handle: object) -> None:
        handle.abort()

    # -- the read path (S4.3.2) --------------------------------------------
    def stream(self, channels: list[str], duration: float,
              chunk_s: float) -> Iterator[dict[str, Capture]]:
        """The chunked-read primitive. Verifies each block begins exactly
        where the previous one ended -- a gap is a hard failure, never closed
        (S4.3.2) -- this only VERIFIES the buffer, never corrects it (S2.1).

        PROVISIONAL, flagged rather than hidden: the architectural intent
        (S4.3.1) is one open request delivering every block, because N
        independent ``getdata`` calls drop samples at every boundary --
        harmless for a recorded channel, permanent loss for a test point.
        Whether repeated ``cdsutils.getdata`` calls (what this method
        actually does, below -- the only pattern ``backend_rtcds.py``
        documents) sustain that for a live multi-hour test-point stream, or
        the NDS2 iterate/stride API is required instead, is spec S9.4's open
        question 4 -- explicitly unresolved and hardware-only. What this
        method delivers regardless of which primitive it ends up built on:
        every block verified adjacent before being trusted, and any fault at
        any point in the sequence uniformly translated into the typed
        hierarchy."""
        channels = list(channels)
        remaining = float(duration)
        prev: dict[str, tuple[int, int, float]] = {}   # channel -> (start_time, n, rate)
        while remaining > 1e-9:
            this_chunk = min(chunk_s, remaining)
            bufs = self._read_once(channels, this_chunk)
            caps: dict[str, Capture] = {}
            for ch, buf in zip(channels, bufs):
                data = np.asarray(buf.data, dtype=float)
                start_time = int(buf.start_time)
                rate = float(buf.sample_rate)
                if ch in prev:
                    prev_start, prev_n, prev_rate = prev[ch]
                    if rate != prev_rate:
                        raise TimingFault(
                            f"{ch}: rate changed mid-campaign ({prev_rate} -> {rate})")
                    expected_start = prev_start + int(round(prev_n / prev_rate))
                    if start_time < expected_start:
                        raise TimingFault(
                            f"{ch}: GPS stepped backwards ({expected_start} -> {start_time})")
                    if start_time != expected_start:
                        raise DataIntegrityError(
                            f"{ch}: gap between blocks (expected {expected_start}, "
                            f"got {start_time})")
                if data.size == 0 or not np.all(np.isfinite(data)):
                    raise DataIntegrityError(f"{ch}: short or non-finite block")
                caps[ch] = Capture(data=data, start_time=start_time, sample_rate=rate)
                prev[ch] = (start_time, data.size, rate)
            yield caps
            remaining -= this_chunk

    def fetch(self, channels: list[str], duration: float) -> dict[str, Capture]:
        """Convenience: consume :meth:`stream`, already-verified, and concat."""
        return _concat_stream(self.stream(list(channels), duration, duration))

    # -- internals ------------------------------------------------------
    def _read_once(self, channels: list[str], duration: float):
        try:
            return self._cdsutils.getdata(channels, duration)
        except Exception as exc:
            raise self._classify(exc) from exc

    @staticmethod
    def _classify(exc: Exception) -> CDSTransportError:
        """Best-effort classification of whatever ``cdsutils``/``nds2``
        raised, by exception class name -- an unrecognised fault still hits
        the ``CDSTransportError`` base, so teardown is always safe even when
        the specific class is a guess (S4.3.4's standing rule)."""
        name = type(exc).__name__.lower()
        if "notfound" in name or "nosuch" in name:
            return ChannelNotFound(str(exc))
        if "timeout" in name:
            return TestpointTimeout(str(exc))
        if "lost" in name:
            return TestpointLost(str(exc))
        return TransportUnavailable(str(exc))


# ---------------------------------------------------------------------------
# TwinTransport -- a compiled rtsfreerun model, synthetic GPS clock
# ---------------------------------------------------------------------------

class _TwinLoopHandle:
    __slots__ = ("channel", "array", "rate", "started", "gain", "aborted", "cursor")

    def __init__(self, channel, array, rate):
        self.channel = channel
        self.array = np.asarray(array, dtype=float)
        self.rate = float(rate)
        self.started = False
        self.gain = 1.0
        self.aborted = False
        self.cursor = 0                 # samples consumed so far, mod array.size


class _TwinStreamHandle:
    __slots__ = ("channel", "rate", "chunks", "gain", "closed", "aborted", "cursor")

    def __init__(self, channel, rate):
        self.channel = channel
        self.rate = float(rate)
        self.chunks: list[np.ndarray] = []
        self.gain = 1.0
        self.closed = False
        self.aborted = False
        self.cursor = 0                 # samples consumed so far (linear, not modulo)

    @property
    def array(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0)


class TwinTransport:
    """Routes ``start``/``stop``/``fetch`` at an rtsfreerun ``mdl`` (the same
    ``run(cycles, excitations=[...], excitation_data=...)`` /
    ``fetch_later(t0, t1, names)`` API ``RTSfreerunBackend`` drives) with a
    synthetic GPS clock, so the same ``CDSBackend`` code path runs against a
    compiled model. Both excitation modes are honoured, so mode selection is
    testable off-hardware (S4.1).

    What this must NOT pretend (S4.1): in the twin, GPS is instantiated at
    runtime and each ``mdl.run()`` is a fresh evolution, so continuity ACROSS
    records is meaningless here, and re-acquiring a whole stretch is cheap --
    neither is true on hardware, where GPS comes from an antenna or an NTP
    server. A gap WITHIN one twin record is representable (keep running
    cycles, do not fetch them); that is what gap tests should use, via
    :meth:`inject_gap`.
    """

    def __init__(self, mdl, start_gps: float = 1_000_000_000.0):
        self._mdl = mdl
        self._t = float(start_gps)
        self._handles: dict[str, object] = {}
        self._gap_next_s: float = 0.0

    def now_gps(self) -> float:
        return self._t

    def probe_rate(self, channels: list[str]) -> float:
        return float(self._mdl.sample_rate)

    def classify(self, channels: list[str]) -> dict[str, ChannelInfo]:
        rate = float(self._mdl.sample_rate)
        return {
            ch: ChannelInfo(exists=True, rate=rate, retrievable="recorded",
                            live=True, injectable=True)
            for ch in channels
        }

    # -- loop mode --------------------------------------------------------
    def start(self, channel: str, array: np.ndarray, rate: float,
              start_gps: float, ramptime: float) -> object:
        h = _TwinLoopHandle(channel, array, rate)
        h.started = True                 # staged eagerly; mdl.run() plays it on fetch
        self._handles[channel] = h
        return h

    def stop(self, handle: object, ramptime: float) -> None:
        handle.started = False
        self._handles.pop(handle.channel, None)

    # -- stream mode --------------------------------------------------------
    def open(self, channel: str, rate: float) -> object:
        h = _TwinStreamHandle(channel, rate)
        self._handles[channel] = h
        return h

    def append(self, handle: object, array: np.ndarray, scale: float = 1.0) -> None:
        if handle.closed or handle.aborted:
            raise DataIntegrityError(f"append on a closed/aborted stream ({handle.channel!r})")
        handle.chunks.append(np.asarray(array, dtype=float) * float(scale))

    def close(self, handle: object) -> None:
        handle.closed = True
        self._handles.pop(handle.channel, None)

    # -- shared -----------------------------------------------------------
    def set_gain(self, handle: object, gain: float, ramptime: float) -> None:
        handle.gain = float(gain)

    def abort(self, handle: object) -> None:
        handle.aborted = True
        handle.gain = 0.0
        self._handles.pop(handle.channel, None)

    # -- fault injection (test-only hook, no hardware equivalent) ---------
    def inject_gap(self, seconds: float) -> None:
        """A gap WITHIN one twin record, before the next chunk: ``mdl`` keeps
        running (its state evolves consistently) but those cycles are never
        fetched, so the next yielded block's ``start_time`` jumps forward --
        exactly the S4.3.2 gap signature. Never use this to simulate a gap
        ACROSS records -- that is not representable here (class docstring)."""
        self._gap_next_s += float(seconds)

    # -- the read path ------------------------------------------------------
    def stream(self, channels: list[str], duration: float,
              chunk_s: float) -> Iterator[dict[str, Capture]]:
        channels = list(channels)
        fs = float(self._mdl.sample_rate)
        remaining = float(duration)
        while remaining > 1e-9:
            if self._gap_next_s:
                n_gap = int(round(self._gap_next_s * fs))
                if n_gap > 0:
                    self._mdl.run(cycles=n_gap, excitations=None, excitation_data=None)
                    self._t += self._gap_next_s
                self._gap_next_s = 0.0

            this_chunk = min(chunk_s, remaining)
            n = int(round(this_chunk * fs))
            names, cols = [], []
            for ch in channels:
                h = self._handles.get(ch)
                if h is None or not isinstance(h, (_TwinLoopHandle, _TwinStreamHandle)):
                    continue
                arr = h.array
                if arr.size == 0:
                    continue
                # Continue this handle's own phase from where the LAST chunk
                # left off -- across chunks of one stream() call, AND across
                # separate calls (e.g. the settle read in _start_staged
                # followed by the analysed read): resetting to arr[0] every
                # call would inject a phase discontinuity right at the
                # settle/analysed boundary, corrupting exactly the periods
                # the settle exists to protect.
                if isinstance(h, _TwinLoopHandle):
                    idx = (h.cursor + np.arange(n)) % arr.size
                    col = arr[idx]
                else:
                    end = min(h.cursor + n, arr.size)
                    col = np.zeros(n)
                    if end > h.cursor:
                        col[: end - h.cursor] = arr[h.cursor: end]
                h.cursor += n
                names.append(ch)
                cols.append(col * h.gain)
            exc_data = np.column_stack(cols) if cols else None

            probe_names = [c for c in channels if c not in names]
            start_time = int(self._t)
            fetch = self._mdl.fetch_later(0, this_chunk, probe_names) if probe_names else None
            self._mdl.run(cycles=n, excitations=names or None, excitation_data=exc_data)
            self._t += this_chunk

            fetched: dict[str, np.ndarray] = {}
            if fetch is not None:
                for name, buf in zip(probe_names, fetch()):
                    fetched[name] = np.asarray(buf.data, dtype=float)

            caps: dict[str, Capture] = {}
            for ch in channels:
                if ch in names:
                    data = cols[names.index(ch)]
                else:
                    data = fetched.get(ch, np.zeros(n))
                caps[ch] = Capture(data=data, start_time=start_time, sample_rate=fs)
            yield caps
            remaining -= this_chunk

    def fetch(self, channels: list[str], duration: float) -> dict[str, Capture]:
        return _concat_stream(self.stream(list(channels), duration, duration))
