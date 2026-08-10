"""RTSfreerun backend: drive a LIGO digital-twin model as a sysID backend.

This is an **adapter**, not a copy of the ``rtsfreerun`` package: it adapts a
*compiled* ``rtsfreerun`` model (an advligorts CDS ``.mdl`` built into an
importable Python module — see ``GIT/digital_twin/``) to the
:class:`~system_ident.backends.base.ChannelBackend` API. (Distinct from
``backends/twin.py``'s in-process analytic ``TwinBackend``.) So the existing
:class:`~system_ident.loop.SysIDLoop` identifies the twin's drive→sensor plant
*under the twin's own realistic seismic + readout noise*.

The FRF excitation is always the deterministic Pintelon–Schoukens **multisine**
injected via :meth:`inject`; the Gaussian generators below are only the realistic
*background* (seismic / readout) injected on separate disturbance channels — never
the measurement drive.

The model object exposes ``mdl.sample_rate``,
``mdl.run(cycles, excitations=[ch], excitation_data=(n,k))`` and
``fetch = mdl.fetch_later(t0, t1, names); ...; buffers = fetch()`` (``buf.data``
per probe).  Channels are CDS ports: ``<NAME>_EXC`` (inject), ``<NAME>_OUT``
(probe).  The model module is **lazy-imported** so importing this package never
requires the twin; the realistic-noise generators are self-contained (the same
IFFT-of-coloured-Gaussian recipe as ``digital_twin/twin/src/twin/noise.py``), so
the adapter and its tests run on a plain SciPy stack.
"""

from __future__ import annotations

import importlib
from fractions import Fraction

import numpy as np
import scipy.signal as sig

from .base import ChannelBackend

# ── realistic noise (self-contained; mirrors digital_twin twin.noise) ─────────
_SEISMIC_PRESETS = {
    "ligo-india": dict(peak=3e-6, f0=0.15, q=3.0, hf_corner=1.0, hf_at_corner=1e-7, floor=1e-11),
    "lho-like":   dict(peak=1e-6, f0=0.15, q=3.0, hf_corner=1.0, hf_at_corner=1e-7, floor=1e-11),
    "llo-like":   dict(peak=5e-6, f0=0.15, q=3.0, hf_corner=1.0, hf_at_corner=1e-7, floor=1e-11),
}


def _seismic_asd(f: np.ndarray, preset: str) -> np.ndarray:
    """Lorentzian microseism peak + 1/f² rolloff + NLNM floor [m/√Hz]."""
    p = _SEISMIC_PRESETS[preset]
    f = np.maximum(np.asarray(f, float), 1e-6)
    lorentz = p["peak"] / np.sqrt(1 + p["q"] ** 2 * (f / p["f0"] - p["f0"] / f) ** 2)
    rolloff = np.where(f > p["hf_corner"], p["hf_at_corner"] * (p["hf_corner"] / f) ** 2, 0.0)
    return np.sqrt(lorentz ** 2 + rolloff ** 2 + p["floor"] ** 2)


def _flat_knee_asd(f: np.ndarray, floor: float, knee_hz: float) -> np.ndarray:
    """Flat readout floor with a 1/f knee (BOSEM/OSEM style) [units/√Hz]."""
    f = np.maximum(np.asarray(f, float), 1e-6)
    return floor * np.sqrt(1 + (knee_hz / f) ** 2)


def _colored_noise(asd_fn, n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """A real time series of length ``n`` with one-sided ASD ``asd_fn(f)``."""
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    scale = asd_fn(freqs) * np.sqrt(n * fs / 2)
    white = (rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)) / np.sqrt(2)
    white[0] = white[0].real
    if n % 2 == 0:
        white[-1] = white[-1].real
    return np.fft.irfft(scale * white, n=n)


class RTSfreerunBackend(ChannelBackend):
    """Drive a compiled rtsfreerun twin model through the channel API.

    Parameters
    ----------
    model:
        Name of the built rtsfreerun module to lazy-import (e.g. ``"x1hsts"``),
        or ``None`` if ``mdl`` is supplied directly (tests).
    exc_channels, readback_channels:
        Maps of model channel name -> DoF (as in ``TwinBackend``). Excitation
        channels are the model ``_EXC`` ports the loop injects into; reading one
        returns the stashed drive (it is not a probe), which satisfies the
        watchdog's drive-peak check. The FRF input X comes from a drive-monitor
        probe named in ``channels.drive``; the response Y from ``readback``.
    fs:
        sysID sample rate; defaults to ``mdl.sample_rate`` (keep them equal to
        avoid resampling and preserve the periodic-multisine structure).
    noise:
        Background realistic noise: list of ``{"channel", "kind", "params"}`` where
        ``kind`` is ``"seismic"`` / ``"bosem"`` / ``"osem"`` / ``"flat"``.
    warmup_s:
        Seconds to run (discarded) before recording, to settle slow loop modes.
    mdl:
        An already-instantiated model object (for tests / a pre-built instance).
    """

    def __init__(self, model=None, *, exc_channels, readback_channels=None, fs=None,
                 noise=None, warmup_s=0.0, saturate=None, seed=None, mdl=None,
                 scenario=None, plant_inputs=None, ramp_s=3.0):
        if mdl is None:
            if model is None:
                raise ValueError("RTSfreerunBackend needs a model name or an mdl instance")
            pkg = importlib.import_module(model)
            mdl = getattr(pkg, model)()
        if scenario is not None:
            # Realise the scenario plant on the bare model (foton ZPK → cdsFilt
            # modules), exactly as the twin orchestrator does. A dict is applied
            # as-is; a path is loaded first.
            from .rtsfreerun_oracle import apply_scenario_init, load_scenario
            scen = scenario if isinstance(scenario, dict) else load_scenario(scenario)
            apply_scenario_init(mdl, scen)
        self._mdl = mdl
        self.model = model
        self.fs_model = float(mdl.sample_rate)
        self.fs = float(fs) if fs is not None else self.fs_model
        self.exc_channels = dict(exc_channels)              # {channel: dof}
        self.readback_channels = dict(readback_channels or {})
        self.noise = list(noise or [])
        self.warmup_s = float(warmup_s)
        self.saturate = None if saturate is None else float(saturate)
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._drives: dict[str, np.ndarray] = {}
        # Virtual "plant input" channels: the true input to a plant inside a closed
        # loop = injected drive + the controller's feedback. Reading one reconstructs
        # ``drive(exc) + coeff·Σ feedback_probes`` (at the model rate, then decimated)
        # so the reference-based FRF (X = plant input) cancels the loop and recovers
        # the *open-loop* plant. ``coeff`` (``feedback_coeff``, default ``+1``) carries
        # the summing-junction sign — e.g. ``-1`` for a ``"+-"`` drive/feedback sum.
        # {name: {"exc": ch, "feedback": [ch,…], "feedback_coeff": ±1}}.
        self.plant_inputs = {k: dict(v) for k, v in (plant_inputs or {}).items()}
        # Actuator-safe injection: ramp every sysID drive up over ``ramp_s`` s and down
        # over ``ramp_s`` s (Tukey) across the assembled warmup+record. The ramp-up lands
        # in the discarded warmup and the ramp-down in the last (dropped) period, so the
        # leakage-free FRF is unaffected. ``ramp_s=0`` disables it.
        self.ramp_s = float(ramp_s)

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> "RTSfreerunBackend":
        """Build from a run config's ``channels`` + ``rtsfreerun`` sections."""
        ch = config["channels"]
        exc = {chan: dof for dof, chan in ch["excitation"].items()}
        rb = {chan: dof for dof, chan in ch["readback"].items()}
        rts = config.get("rtsfreerun", {})
        kwargs.setdefault("ramp_s", float(config.get("measurement", {}).get("t_ramp", 3.0)))
        return cls(rts.get("model"), exc_channels=exc, readback_channels=rb,
                   noise=rts.get("noise", []), warmup_s=float(rts.get("warmup_s", 0.0)),
                   saturate=rts.get("saturate"), scenario=rts.get("scenario"), **kwargs)

    # -- channel API ---------------------------------------------------------
    def inject(self, channel: str, timeseries: np.ndarray, fs: float) -> None:
        if channel not in self.exc_channels:
            raise KeyError(f"unknown excitation channel {channel!r}")
        ts = np.asarray(timeseries, dtype=float)
        if not np.isclose(fs, self.fs_model):
            # ``resample_poly`` is a FIR filter and sees ``ts`` as a finite,
            # zero-padded-at-the-ends array -- but ``ts`` is an exactly periodic
            # multisine tiling, and the FIR's edge transient corrupts precisely the
            # periods nearest each end (measured: 6.9% on the first period, 53% on
            # the last, at the shipped 256/16384 ratio). ``sig.resample`` is
            # FFT-based and needs no such padding assumption: an exactly-periodic
            # array's DFT is already sparse on the harmonics its own period repeats
            # at, so resampling the FULL tiled array in the frequency domain is
            # exactly equivalent to resampling one period and re-tiling it -- no
            # period-boundary bookkeeping required -- and measured deviation from
            # the interior period drops from 5.3e-1 to ~1e-16 (#10).
            ts = sig.resample(ts, int(round(len(ts) * self.fs_model / fs)))
        if self.saturate is not None:
            ts = np.clip(ts, -self.saturate, self.saturate)
        self._drives[channel] = ts

    def read(self, channels: list[str], duration: float) -> dict[str, np.ndarray]:
        n_rec = int(round(duration * self.fs_model))
        n_warm = int(round(self.warmup_s * self.fs_model))
        # Actuator-safe injection lives OUTSIDE the measured record: the drive ramps up
        # over ``ramp_s`` s BEFORE the warmup settle (so ``warmup_s`` is still a
        # full-amplitude settling time) and ramps down over ``ramp_s`` s in a trailing
        # tail that is not fetched. The measured record stays full-amplitude flat, so the
        # FRF recovers the plant to within tolerance — NOT bit-identical to the un-ramped
        # case: the ramp-up shifts the multisine phase seen during settling, leaving a
        # small high-Q transient residual that ``warmup_s`` must be large enough to absorb
        # (≈2× the slowest mode's ringdown; the 6-DOF HSTS demo uses 32 s for its ~0.67 Hz
        # modes). Budget the warmup accordingly when adding a high-Q model.
        n_ramp = int(round(self.ramp_s * self.fs_model)) if self.ramp_s > 0.0 else 0
        n_lead = n_ramp + n_warm                            # ramp-up + full-amplitude settle
        n_total = n_lead + n_rec + n_ramp

        # Assemble excitation columns over lead+record+tail (drives tiled then ramped;
        # noise/disturbance channels are never ramped). The drive is phase-aligned so the
        # measured record begins exactly at the injected drive's period start — otherwise
        # the lead (not a whole number of periods) cyclically shifts the multisine and the
        # decimation edges turn that shift into FRF bias. Cache the ramped drives so the
        # drive-monitor and plant-input readbacks reuse the same waveform.
        ramped: dict[str, np.ndarray] = {
            chan: self._soft_start_stop(self._fit_periodic(drive, n_total))
            for chan, drive in self._drives.items()
        }
        names: list[str] = []
        cols: list[np.ndarray] = []
        for chan in self._drives:
            names.append(chan)
            cols.append(ramped[chan])
        for ns in self.noise:
            names.append(ns["channel"])
            cols.append(self._gen_noise(ns, n_total))
        exc_data = np.column_stack(cols) if cols else None

        # Expand any requested virtual plant-input channel into the model probes it
        # needs (its feedback monitors), so they get fetched alongside the rest.
        feedback_needed: list[str] = []
        for c in channels:
            if c in self.plant_inputs:
                feedback_needed += self.plant_inputs[c]["feedback"]
        probe_names = [c for c in dict.fromkeys(list(channels) + feedback_needed)
                       if c not in self.exc_channels and c not in self.plant_inputs]
        if n_lead > 0:                                       # ramp-up + settle (discarded)
            self._mdl.run(cycles=n_lead, excitations=names or None,
                          excitation_data=None if exc_data is None else exc_data[:n_lead])
        fetch = self._mdl.fetch_later(0, duration, probe_names) if probe_names else None
        self._mdl.run(cycles=n_rec, excitations=names or None,
                      excitation_data=None if exc_data is None else exc_data[n_lead:n_lead + n_rec])
        if n_ramp > 0:                                       # ramp the actuator down (not fetched)
            self._mdl.run(cycles=n_ramp, excitations=names or None,
                          excitation_data=None if exc_data is None else exc_data[n_lead + n_rec:n_total])

        fetched: dict[str, np.ndarray] = {}
        if fetch is not None:
            for name, buf in zip(probe_names, fetch()):
                fetched[name] = np.asarray(buf.data, dtype=float)

        out: dict[str, np.ndarray] = {}
        for c in channels:
            if c in self.plant_inputs:                      # virtual: drive ± feedback
                spec = self.plant_inputs[c]
                coeff = float(spec.get("feedback_coeff", 1.0))
                x = self._drive_record(ramped, spec["exc"], n_total, n_lead, n_rec)
                for fb in spec["feedback"]:
                    x = x + coeff * fetched.get(fb, np.zeros(n_rec))
                out[c] = self._to_sysid_rate(x, n_rec)
            elif c in self.exc_channels:                    # drive monitor (stashed)
                d = self._drive_record(ramped, c, n_total, n_lead, n_rec)
                out[c] = self._to_sysid_rate(d, n_rec)
            else:                                           # model probe
                out[c] = self._to_sysid_rate(fetched.get(c, np.zeros(n_rec)), n_rec)
        return out

    def ramp_down(self, channel: str, secs: float) -> None:
        drive = self._drives.get(channel)
        if drive is None or len(drive) == 0:
            return
        n = min(int(round(secs * self.fs_model)), len(drive))
        out = np.zeros_like(drive)
        if n > 0:
            out[:n] = drive[:n] * 0.5 * (1 + np.cos(np.pi * np.arange(n) / n))
        self._drives[channel] = out

    # -- safe-state handoff --------------------------------------------------
    def snapshot_state(self, channels: list[str]) -> dict:
        return {"drives": {c: d.copy() for c, d in self._drives.items()}}

    def restore_state(self, snapshot: dict) -> None:
        self._drives = {c: d.copy() for c, d in snapshot["drives"].items()}

    # -- internals -----------------------------------------------------------
    def _gen_noise(self, ns: dict, n: int) -> np.ndarray:
        kind = ns["kind"]
        params = ns.get("params", {})
        if kind == "seismic":
            preset = params.get("preset", "ligo-india")
            asd = lambda f: _seismic_asd(f, preset)  # noqa: E731
        elif kind in ("bosem", "osem", "flat"):
            default_floor = {"bosem": 1e-10, "osem": 3e-10, "flat": 1.0}[kind]
            floor = float(params.get("floor", default_floor))
            knee = float(params.get("knee_hz", 1.0))
            asd = lambda f: _flat_knee_asd(f, floor, knee)  # noqa: E731
        else:
            raise ValueError(f"unknown rtsfreerun noise kind {kind!r}")
        return _colored_noise(asd, n, self.fs_model, self._rng)

    def _to_sysid_rate(self, arr: np.ndarray, n_rec: int) -> np.ndarray:
        """Resample a model-rate (length ``n_rec``) array to the sysID rate.

        Returns exactly ``round(n_rec * fs/fs_model)`` samples — for an integer
        decimation (e.g. 16384 → 256) that preserves the periodic-multisine
        block structure the leakage-free FRF needs.
        """
        arr = np.asarray(arr, dtype=float)
        if np.isclose(self.fs, self.fs_model):
            return self._fit_length(arr, n_rec)
        n_out = int(round(n_rec * self.fs / self.fs_model))
        frac = Fraction(self.fs / self.fs_model).limit_denominator(100000)
        res = sig.resample_poly(arr, frac.numerator, frac.denominator)
        return self._fit_length(res, n_out)

    def _soft_start_stop(self, col: np.ndarray) -> np.ndarray:
        """The base on/off ramp over the assembled drive at the model rate."""
        return ChannelBackend._soft_start_stop(self, col, self.fs_model)

    def _drive_record(self, ramped: dict, chan, n_total: int, n_lead: int,
                      n_rec: int) -> np.ndarray:
        """The measured-record slice (full-amplitude flat) of the cached ramped drive —
        between the leading ramp-up+settle and the trailing ramp-down tail."""
        col = ramped.get(chan)
        if col is None:
            col = self._soft_start_stop(self._fit_periodic(self._drives.get(chan), n_total))
        return col[n_lead:n_lead + n_rec]

    @staticmethod
    def _fit_periodic(x, n: int) -> np.ndarray:
        """Tile ``x`` periodically (or zero-fill) to length ``n``."""
        if x is None or len(x) == 0:
            return np.zeros(n)
        return np.resize(np.asarray(x, dtype=float), n)

    @staticmethod
    def _fit_length(x, n: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) >= n:
            return x[:n]
        out = np.zeros(n)
        out[: len(x)] = x
        return out
