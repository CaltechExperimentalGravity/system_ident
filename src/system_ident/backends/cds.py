"""Real-hardware backend: transport-agnostic awg-inject + nds2-readback,
against real CDS hardware or a compiled twin, behind :mod:`.cds_transport`.

The CDS libraries (``awg``/``cdsutils``/``gpstime``) are provided by the LIGO
control-room environment. This module never imports them directly — that
happens only inside :class:`~.cds_transport.AWGNDSTransport`, itself
lazy-imported only when actually constructed — so ``import system_ident``
(and this module) never requires them; the twin/rtsfreerun path stays usable
on a plain scientific-Python stack.

Stage D, issues #13-#16, amended by #31 (both excitation modes) and #32
(spec S4.3: pre-flight probe, chunked read with a record verdict, read-cache
generation invalidation).

LIMITATIONS — what the periodic / ML path has and has NOT been validated
against
--------------------------------------------------------------------------
The Pintelon-Schoukens periodic-multisine measurement, the reference-based
closed-loop FRF, and the ML estimator are validated on the digital twin and
(via :class:`~.cds_transport.TwinTransport`) a compiled ``rtsfreerun`` model.
The following remain **UNTESTED** until this backend is run against real
:class:`~.cds_transport.AWGNDSTransport` hardware:

* **Real actuator nonlinearity** beyond a hard clip (DAC slew, analog
  saturation shape), which the Schroeder crest-factor choice mitigates but
  does not eliminate.
* **True controller dynamics** (the twin's feedback ``C(s)`` is a rational
  stand-in for the real digital servo + filters).
* Everything spec S6 lists as hardware-only: whether ``awg`` accepts an
  untapered integer-period array with ``ramptime`` start/stop semantics
  exactly as modelled here, AWG-slot release on a second injection,
  simultaneous-mode start skew, and whether the live NDS path sustains a
  continuous multi-hour test-point stream (S9.4).

What is **not** a gap, despite looking like one at first: a sample offset
between drive and response, constant OR one that varies by hundreds of
samples, cancels exactly in the ratio-of-averages FRF ``H = mean(Y)/mean(X)``
as long as X is a real readback (spec S2.1) — this is why :meth:`read` below
never rolls, trims, or GPS-aligns anything. Do not "fix" that.
"""

from __future__ import annotations

import atexit
import signal
import sys
import warnings
from fractions import Fraction

import numpy as np
import scipy.signal as sig

from ..safety import SafetyAbort, SafetyLimits
from .base import ChannelBackend
from .cds_transport import (
    CDSTransport,
    CDSTransportError,
    DataIntegrityError,
    TestpointTimeout,
    TestpointLost,
    TimingFault,
    TransportUnavailable,
)

_ABORT_TIER = (TestpointLost, TransportUnavailable, TestpointTimeout)
_REJECT_TIER = (DataIntegrityError, TimingFault)


def _default_authorizer(prompt: str) -> bool:
    """Rule 1/Rule 2 (spec S1): only a human may approve a hardware
    injection, and never by accident. Defaults to DENY on EOFError or a
    non-TTY stdin -- never proceed in batch."""
    if not sys.stdin.isatty():
        return False
    try:
        reply = input(prompt)
    except EOFError:
        return False
    return reply.strip().lower() == "yes"


class HardwareStateFault(CDSTransportError):
    """The plant changed under the measurement (S4.3.3 item 4) -- inferential
    only here (drive present while response RMS collapses against the
    quiet-time baseline); authoritative detection needs EPICS (Component 2)."""


class CDSBackend(ChannelBackend):
    """awg-inject + nds2-readback, against any :class:`CDSTransport`.

    Parameters
    ----------
    transport:
        A :class:`~.cds_transport.CDSTransport` -- ``AWGNDSTransport`` for
        real hardware, ``TwinTransport`` for a compiled model. Never
        constructed here; the caller (eventually ``config.py``'s
        ``build_cds_backend``) picks it, so this class is fully testable
        without ``awg``/``cdsutils`` ever loading.
    exc_channels, readback_channels:
        ``{channel: dof}`` -- the shape ``Watchdog`` and every other backend
        use (``RTSfreerunBackend``/``TwinBackend``). Without this exact
        shape ``Watchdog.evaluate`` silently never matches a channel, so
        auto-abort is dead and ``ramp_down`` is never called (spec S3.1).
    drive_channels:
        ``{dof: channel}`` (the opposite shape -- CDSBackend-internal, never
        consumed by ``Watchdog``, kept dof-keyed because that is the lookup
        this class actually needs: "this exc channel's dof -> its monitor").
        **Mandatory** (spec S2.1 item 3): ``loop.py``'s fallback to the
        excitation channel itself is exactly the 200%-error configuration on
        hardware (a stashed drive, not a real readback).
    fs:
        The sysID sample rate every channel decimates to (or, for a
        test-point excitation/monitor at the native hardware rate, IS).
    exc_mode:
        ``"stream"`` (default, spec S2.3) or ``"loop"`` (a supported peer
        mode). Selecting ``loop`` trades the cosine taper for a linear
        transport-level gain ramp -- harsher on the actuator.
    t_ramp, warmup_s, segment_duration, n_segments:
        The same knobs every backend already uses (``rtsfreerun``/twin
        ``t_ramp``, ``warmup_s``; ``measurement.segment_duration``,
        ``n_segments``) -- one meaning per knob across every backend (S2.3).
        No new configuration for the envelope.
    read_chunk_s, passive_read_retries:
        S4.3's two new (in-code-defaulted) knobs, both provisional pending a
        framebuilder measurement (spec S9.4).
    """

    def __init__(
        self,
        transport: CDSTransport,
        *,
        exc_channels: dict[str, str],
        readback_channels: dict[str, str],
        drive_channels: dict[str, str],
        fs: float,
        exc_mode: str = "stream",
        t_ramp: float = 3.0,
        warmup_s: float = 0.0,
        segment_duration: float | None = None,
        n_segments: int = 1,
        read_chunk_s: float = 1.0,
        passive_read_retries: int = 3,
        safety_limits: SafetyLimits | None = None,
        authorizer=_default_authorizer,
    ) -> None:
        if exc_mode not in ("stream", "loop"):
            raise ValueError(f"cds.exc_mode must be 'stream' or 'loop', got {exc_mode!r}")
        if exc_mode == "loop":
            warnings.warn(
                "cds.exc_mode: loop selects a LINEAR transport gain ramp -- harsher on "
                "the actuator than the default cosine taper (exc_mode: stream).",
                stacklevel=2,
            )
        if not drive_channels:
            raise ValueError(
                "CDSBackend requires channels.drive for every DoF (spec S2.1 item 3): "
                "falling back to the excitation channel returns the stashed drive, not "
                "a real readback -- the exact 200%-error configuration on hardware."
            )

        self.transport = transport
        self.exc_channels = dict(exc_channels)
        self.readback_channels = dict(readback_channels)
        self.drive_channels = dict(drive_channels)          # {dof: channel}
        self.fs = float(fs)
        self.exc_mode = exc_mode
        self.t_ramp = float(t_ramp)
        self.ramp_s = float(t_ramp)                          # ChannelBackend contract bookkeeping
        self.warmup_s = float(warmup_s)
        self.segment_duration = float(segment_duration) if segment_duration else None
        self.n_segments = int(n_segments)
        self.read_chunk_s = float(read_chunk_s)
        self.passive_read_retries = int(passive_read_retries)
        self.safety_limits = safety_limits
        self.authorizer = authorizer

        self._rates: dict[str, float] = {}                   # per-channel fs_hw (S4.3.1)
        self._generation = 0
        self._staged: dict[str, dict] = {}                    # channel -> {"array", "rate"}
        self._approval: dict[str, bool] = {}                  # channel -> single-use approval token
        self._live: dict[str, object] = {}                    # channel -> transport handle
        self._cache: dict[tuple[int, float], dict[str, np.ndarray]] = {}
        self._snapshot: dict | None = None

        self._all_channels = set(self.exc_channels) | set(self.readback_channels) \
            | set(self.drive_channels.values())

        # Zero-risk, pre-injection (#13): probe/classify every channel the
        # campaign will ever touch BEFORE any of them can be staged, so a bad
        # channel name or rate mismatch fails here -- not after an injection.
        self._preflight()
        self._register_lifecycle()

    # -- construction / pre-flight (#13, amended S4.3.5) ---------------------
    @classmethod
    def from_config(cls, config: dict, transport: CDSTransport) -> "CDSBackend":
        ch = config["channels"]
        exc = {dof: chan for dof, chan in ch["excitation"].items()}
        rb = {dof: chan for dof, chan in ch["readback"].items()}
        drive = {dof: chan for dof, chan in ch.get("drive", {}).items()}
        m = config["measurement"]
        cds_cfg = config.get("cds", {})
        return cls(
            transport,
            exc_channels={c: d for d, c in exc.items()},
            readback_channels={c: d for d, c in rb.items()},
            drive_channels=drive,
            fs=float(m["fs"]),
            exc_mode=cds_cfg.get("exc_mode", "stream"),
            t_ramp=float(m.get("t_ramp", 3.0)),
            warmup_s=float(cds_cfg.get("warmup_s", 0.0)),
            segment_duration=float(m.get("segment_duration", 0.0)) or None,
            n_segments=int(m.get("n_segments", 8)),
            read_chunk_s=float(cds_cfg.get("read_chunk_s", 1.0)),
            passive_read_retries=int(cds_cfg.get("passive_read_retries", 3)),
        )

    def _preflight(self) -> None:
        """The pre-flight channel probe (S4.3.5), folded into one classify()
        call -- zero extra hardware cost, still pre-injection. Failing here
        costs nothing; failing after an injection costs an injection."""
        channels = sorted(self._all_channels)
        info = self.transport.classify(channels)
        for ch in channels:
            if ch not in info or not info[ch].exists:
                raise TransportUnavailable(f"channel not found during pre-flight: {ch!r}")
            ci = info[ch]
            self._rates[ch] = ci.rate
            if ci.rate % self.fs != 0:
                raise TransportUnavailable(
                    f"{ch!r}: hardware rate {ci.rate} Hz is not an integer multiple of "
                    f"measurement.fs {self.fs} Hz"
                )
            if self.segment_duration is not None:
                if abs(self.segment_duration * ci.rate - round(self.segment_duration * ci.rate)) > 1e-6:
                    raise TransportUnavailable(
                        f"{ch!r}: segment_duration * {ci.rate} Hz is not an integer sample count"
                    )
                if abs(self.segment_duration * self.fs - round(self.segment_duration * self.fs)) > 1e-6:
                    raise TransportUnavailable(
                        f"segment_duration * fs ({self.fs} Hz) is not an integer sample count"
                    )
            if not ci.live:
                warnings.warn(f"{ch!r}: not live at pre-flight (non-finite, or zero variance)",
                              stacklevel=3)
            if ch in self.exc_channels and not ci.injectable:
                raise TransportUnavailable(
                    f"{ch!r} is not structurally injectable (a slow read-only EPICS "
                    "record, not a fast test point)"
                )
        # X/Y rate mismatch costs the exact decimation-filter cancellation
        # (spec S7) -- warn and proceed, never refuse (S4.3.1).
        for exc_ch, dof in self.exc_channels.items():
            drive_ch = self.drive_channels.get(dof)
            if drive_ch and self._rates.get(drive_ch) not in (None, self._rates.get(exc_ch)):
                warnings.warn(
                    f"{dof}: drive-monitor {drive_ch!r} rate {self._rates[drive_ch]} Hz != "
                    f"excitation {exc_ch!r} rate {self._rates[exc_ch]} Hz -- the decimation "
                    "filters differ and no longer cancel in Ybar/Xbar (spec S4.3.1, S7)",
                    stacklevel=3,
                )

    # -- inject: stage, never start (#13/#31) ---------------------------------
    def inject(self, channel: str, timeseries: np.ndarray, fs: float) -> None:
        if channel in self._live:
            self.transport.abort(self._live.pop(channel))
        self._generation += 1
        self._cache.clear()

        rate = self._rates.get(channel)
        if rate is None:
            raise TransportUnavailable(f"{channel!r} was not classified at pre-flight")
        ts = np.asarray(timeseries, dtype=float)
        # One period, then tile -- never resample the tiled array (#10's own
        # lesson). Resampling the FULL periodic array with the FFT-based
        # sig.resample is exactly equivalent to resampling one period and
        # re-tiling (verified in tests/test_periodic_measurement.py), so no
        # period-boundary bookkeeping is needed here either.
        resampled = sig.resample(ts, int(round(ts.size * rate / fs)))

        if self.exc_mode == "loop":
            # [loop]: do NOT call _soft_start_stop (S2.2) -- the ramp is the
            # transport's own linear gain ramp, applied at start() time.
            staged_array = resampled
        else:
            # [stream]: the one-shot four-segment envelope (S2.3) -- this
            # taper IS _soft_start_stop's job done a different way, so that
            # method is equally never called here, not merely skipped.
            staged_array = self._segmented_envelope(resampled, rate)

        # Pre-injection amplitude check (#4), on the EXACT samples about to be
        # handed to the transport -- opt-in (twin/rtsfreerun never call this).
        # #32 item 5 folds in here too: a non-finite drive is rejected before
        # it can reach append()/ArbitraryLoop, not after.
        if not np.all(np.isfinite(staged_array)):
            raise SafetyAbort(f"{channel!r}: non-finite samples in the drive, refusing to inject")
        if self.safety_limits is not None:
            staged_array = self._check_drive_limits(staged_array, self.safety_limits)
        self._staged[channel] = {"array": staged_array, "rate": rate}

        # The single-use approval token (#17, spec S5): prompts HERE, on the
        # exact samples staged above; read()'s first call of this generation
        # is what CONSUMES it (Rule 2 -- ask again, every time a NEW inject()
        # actually stages something new). Re-reads of an already-running
        # injection never re-prompt: nothing new is actuated then.
        peak = float(np.max(np.abs(staged_array))) if staged_array.size else 0.0
        rms = float(np.sqrt(np.mean(staged_array ** 2))) if staged_array.size else 0.0
        crest = peak / rms if rms > 0 else float("inf")
        prompt = (
            f"about to inject on {channel!r}: rate={rate:.6g} Hz, "
            f"duration={staged_array.size / rate:.3f} s, peak={peak:.6g}, rms={rms:.6g}, "
            f"crest={crest:.3f}, start_gps~={self.transport.now_gps():.3f}. "
            "type 'yes' to approve this injection: "
        )
        self._approval[channel] = bool(self.authorizer(prompt))

    def _segmented_envelope(self, main_period: np.ndarray, rate: float) -> np.ndarray:
        """ramp-on(t_ramp, cosine) / settle(warmup_s, flat) / main(segment_duration
        * n_segments, flat) / ramp-off(t_ramp, cosine) -- spec S2.3. A single
        ``tukey(N, alpha)`` window over the WHOLE 4-segment span already IS
        exactly this shape (rising cosine over the first ``alpha*N/2``
        samples, flat 1.0 in the middle, falling cosine over the last
        ``alpha*N/2``) for ``alpha = 2*n_ramp/N`` -- the same alpha formula
        ``ChannelBackend._soft_start_stop`` uses, just applied to a
        differently-shaped array (this one is never fed through that method;
        this taper IS its job, done via the transport instead, S2.2)."""
        if self.segment_duration is None:
            raise TransportUnavailable("cds.exc_mode: stream needs measurement.segment_duration")
        n_main = int(round(self.segment_duration * self.n_segments * rate))
        n_settle = int(round(self.warmup_s * rate))
        n_ramp = int(round(self.t_ramp * rate))
        n_total = n_ramp + n_settle + n_main + n_ramp
        full_signal = np.resize(main_period, n_total)
        alpha = min(1.0, 2.0 * n_ramp / max(n_total, 1))
        return full_signal * sig.windows.tukey(n_total, alpha)

    # -- read: assemble/start on the first call of a generation (#14/#15) ---
    def read(self, channels: list[str], duration: float) -> dict[str, np.ndarray]:
        if abs(duration - round(duration)) > 1e-9:
            raise TransportUnavailable(f"duration {duration} does not round to an integer second")
        duration = float(round(duration))

        newly_staged = [ch for ch in self._staged if ch not in self._live]
        if newly_staged:
            self._start_staged(newly_staged)

        key = (self._generation, duration)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._fetch_verdict(sorted(self._all_channels), duration,
                                         excited=bool(self._live))
            self._cache[key] = cached

        out: dict[str, np.ndarray] = {}
        for ch in channels:
            if ch not in cached:
                raise TransportUnavailable(f"{ch!r} was not part of the pre-flight channel set")
            out[ch] = cached[ch]
        return out

    def _start_staged(self, channels: list[str]) -> None:
        """The first read() of a generation: open every staged stream and
        begin append() for all of them in one tight loop (stream mode), or
        start(ramptime=0)-trigger every staged loop (loop mode) -- S9.3's
        stash-then-assemble pattern, so simultaneous mode is tractable.
        NOT implemented here: S9.3's skew-minimising "start all unramped,
        then gain-ramp separately" loop-mode refinement -- that exact detail
        is still an open, partly hardware-only question (spec S9 item 3);
        this uses backend_rtcds.py's proven per-channel ramptime instead."""
        started: list[str] = []
        try:
            for ch in channels:
                # Refuse to start any staged injection whose token is missing
                # or already consumed (#17). A gate that constructs the AWG
                # object and then declines is still a gate that reached the
                # AWG API -- check BEFORE transport.start()/open(), not after.
                if not self._approval.pop(ch, False):
                    raise SafetyAbort(
                        f"{ch!r}: injection not approved (denied, or inject() was never "
                        "called with an approving authorizer) -- refusing to start"
                    )
                staged = self._staged.pop(ch)
                if self.exc_mode == "loop":
                    handle = self.transport.start(
                        ch, staged["array"], staged["rate"], self.transport.now_gps(),
                        ramptime=self.t_ramp)
                else:
                    handle = self.transport.open(ch, staged["rate"])
                    self.transport.append(handle, staged["array"])
                self._live[ch] = handle
                started.append(ch)
        except Exception:
            self._stop_all()
            raise
        # settle once (#15): ramp_s + warmup_s, discarded, never cached.
        # Both modes need this -- [loop]'s ramp is the transport's OWN gain
        # ramp over the untapered array, so the plant sees a tapered drive
        # here too, just via a different mechanism than [stream]'s baked-in
        # envelope; either way the transient it causes must not enter the
        # analysed record.
        lead_s = self.t_ramp + self.warmup_s
        if lead_s > 0:
            self._fetch_verdict(sorted(self._all_channels), lead_s, excited=True, cache=False)

    # -- the read path (#14, record verdict S4.3.6) ---------------------------
    def _fetch_verdict(self, channels: list[str], duration: float, *, excited: bool,
                       cache: bool = True) -> dict[str, np.ndarray]:
        attempts = 1 if excited else 1 + self.passive_read_retries
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                caps = self.transport.fetch(channels, duration)
                verdict: dict[str, np.ndarray] = {}
                for ch in channels:
                    cap = caps[ch]
                    data = np.asarray(cap.data, dtype=float)
                    expected_n = int(round(duration * cap.sample_rate))
                    if data.size != expected_n:
                        raise DataIntegrityError(
                            f"{ch}: expected {expected_n} samples, got {data.size}")
                    if not np.all(np.isfinite(data)):
                        raise DataIntegrityError(f"{ch}: non-finite samples in record")
                    verdict[ch] = self._decimate(data, cap.sample_rate, ch)
                return verdict
            except _ABORT_TIER:
                # Transport/hardware-state faults abort the campaign through
                # the EXISTING teardown path (S4.3.6) -- this backend's own
                # idempotent _stop_all, then re-raise so loop.py's abort-tier
                # handling calls watchdog.abort() (the same path SafetyAbort
                # already uses; no second abort mechanism).
                self._stop_all()
                raise
            except _REJECT_TIER as exc:
                last_exc = exc
                if excited:
                    # Excited records never auto-retry: re-taking one means
                    # re-injecting, a new actuation needing a fresh approval
                    # token (S4.3.7/S1 Rule 2). Reject at zero weight, never
                    # a small one -- that is #6's lesson, restated here.
                    raise
                # Passive (no excitation): bounded, logged retry (S4.3.2/S4.3.7).
                continue
        raise last_exc

    def _decimate(self, data: np.ndarray, rate: float, channel: str) -> np.ndarray:
        """Hardware-rate -> fs, applied ONCE to the assembled contiguous
        record, never per block: per-block decimation would reintroduce
        #10's measured edge failure at every chunk boundary. A genuine
        anti-aliasing decimation (resample_poly), not the FFT-exact resample
        `inject()` uses for the periodic design array (S7: the decimation
        filter must actually be LTI to cancel in Ybar/Xbar)."""
        if np.isclose(rate, self.fs):
            return data
        frac = Fraction(self.fs / rate).limit_denominator(100000)
        return sig.resample_poly(data, frac.numerator, frac.denominator)

    # -- ramp_down / lifecycle (#16) ------------------------------------------
    def ramp_down(self, channel: str, secs: float) -> None:
        handle = self._live.get(channel)
        if handle is None:
            return  # idempotent: already stopped, or never started
        if self.exc_mode == "loop":
            self.transport.stop(handle, ramptime=secs)
        else:
            self.transport.set_gain(handle, 0.0, ramptime=secs)
        del self._live[channel]

    def _stop_all(self) -> None:
        """Idempotent hard stop: every transport fault, KeyboardInterrupt,
        atexit and SIGINT/SIGTERM tear down through this ONE routine (#16) --
        not just the historical KeyboardInterrupt-only case."""
        while self._live:
            channel, handle = self._live.popitem()
            try:
                self.transport.abort(handle)
            except Exception as exc:                       # noqa: BLE001
                print(f"\n*** WARNING: could not abort the excitation on {channel!r}: "
                      f"{exc!r}\n*** Check for a running excitation before doing anything else.\n")
        self._staged.clear()

    def _register_lifecycle(self) -> None:
        # #19: a mandatory out-of-band STOP -- printed prominently at startup
        # rather than left implicit, since this is the operator's actual
        # stop path whenever the dashboard is not running (--no-dashboard,
        # or headless because the extra is not installed).
        print("CDS backend ready. Out-of-band STOP: Ctrl-C (SIGINT) immediately hard-stops "
              "every live excitation (or the dashboard's STOP button, which ramps down "
              "gracefully via ramp_down/Watchdog.abort, if running).")
        atexit.register(self._stop_all)
        for sig_num in (signal.SIGINT, signal.SIGTERM):
            try:
                prev = signal.getsignal(sig_num)
            except (ValueError, OSError):
                continue                                    # not the main thread; atexit still covers it

            def _handler(signum, frame, _prev=prev):
                self._stop_all()
                if callable(_prev):
                    _prev(signum, frame)

            try:
                signal.signal(sig_num, _handler)
            except (ValueError, OSError):
                pass

    # -- safe-state handoff (#16, mandatory) -----------------------------------
    def snapshot_state(self, channels: list[str]) -> dict:
        """Live-excitation state only. Filter-module/gain/offset state is
        **not** captured -- that needs ``ezca``/pyepics and operator sign-off
        on what may be written (Component 2). Do not fake it."""
        return {"generation": self._generation,
               "staged": {ch: dict(v) for ch, v in self._staged.items()},
               "live": set(self._live)}

    def restore_state(self, snapshot: dict) -> None:
        """The "hand control back to the damping loops" step on hardware
        needs ``ezca``/pyepics (absent from ``pyproject.toml``) plus operator
        sign-off (Component 2) -- so what this actually restores is only
        what :meth:`snapshot_state` actually captured: any live excitation is
        stopped. It does NOT re-inject the snapshot's staged drives."""
        self._stop_all()
