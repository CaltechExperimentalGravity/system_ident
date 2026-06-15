"""Digital-twin backend: same channel API, backed by the suspension plant.

Applies each per-DoF plant transfer function to the injected excitation to
synthesise the readback, plus additive sensor noise, so the entire loop /
safety / dashboard stack runs in simulation with no CDS libraries present.

The continuous-time ``TFModel`` is discretised with the bilinear transform at
the backend sample rate (matching ``sysIDlib.par_dict_to_sos``), then the drive
is filtered through it. ``read`` with no prior injection returns sensor noise
only — i.e. the quiet-time readback whose PSD is the ``Pyy`` the Fisher and
excitation-design code consume.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import scipy.signal as sig

from ..plant import SuspensionPlant
from .base import ChannelBackend


class TwinBackend(ChannelBackend):
    """Simulation backend sharing the real channel API.

    Parameters
    ----------
    plant:
        The suspension plant (per-DoF transfer functions + sample rate).
    exc_channels, readback_channels:
        Maps of channel name -> DoF for the excitation and readback channels.
    fs:
        Backend sample rate [Hz]; defaults to ``plant.fs``.
    sensor_asd:
        Flat (white) sensor-noise amplitude spectral density added to every
        readback, in readback units / sqrt(Hz). ``0`` means a noiseless twin.
    seed:
        Seed / ``Generator`` for the sensor noise.
    """

    def __init__(
        self,
        plant: SuspensionPlant,
        exc_channels: dict[str, str],
        readback_channels: dict[str, str],
        fs: float | None = None,
        sensor_asd: float = 0.0,
        disturbance_asd: float = 0.0,
        seed: int | np.random.Generator | None = None,
        controllers: dict[str, tuple] | None = None,
        injection_point: str | dict[str, str] = "after_controller",
        drive_channels: dict[str, str] | None = None,
        error_channels: dict[str, str] | None = None,
        response_delay_samples: int = 0,
        saturate: float | None = None,
        coupling: dict | None = None,
    ) -> None:
        self.plant = plant
        self.exc_channels = dict(exc_channels)
        self.readback_channels = dict(readback_channels)
        # Closed-loop digital test points (per DoF): the after-controller drive
        # ``u`` (the plant input -> FRF input X) and the before-controller error
        # ``e`` (a diagnostic). Both map channel name -> DoF, like the others.
        self.drive_channels = dict(drive_channels) if drive_channels else {}
        self.error_channels = dict(error_channels) if error_channels else {}
        self.fs = float(fs if fs is not None else plant.fs)
        self.sensor_asd = float(sensor_asd)
        self.disturbance_asd = float(disturbance_asd)
        self._rng = (
            seed if isinstance(seed, np.random.Generator)
            else np.random.default_rng(seed)
        )
        self._drives: dict[str, np.ndarray] = {}
        # Discretise each plant TF once (bilinear at the backend rate).
        self._ba = {
            dof: sig.bilinear(tf.num, tf.den, self.fs)
            for dof, tf in plant.transfer_functions.items()
        }
        # Optional realism knobs (all off by default -> byte-identical to before):
        #  * controllers   : per-DoF continuous feedback C(s)=(num,den); engaging a
        #    loop so the closed-loop reference-based FRF can be exercised on the twin.
        #  * response_delay_samples : differential transport delay on the response
        #    path (a real DAC->analog->ADC chain), a linear phase a rational fit sees.
        #  * saturate       : hard actuator clip, so the multisine crest factor matters.
        self.controllers = dict(controllers) if controllers else None
        # Per-DoF excitation injection point relative to the digital controller:
        #  * "after_controller"  (default): EXC sums into the drive u (= r − C·y),
        #    the actuator-side injection the twin has always modelled;
        #  * "before_controller": EXC enters ahead of C (u = C·(EXC − y)), so EXC
        #    propagates through the loop filter. Either way the reference-based FRF
        #    mean(Y)/mean(X) with X = u, Y = y recovers the open-loop plant G.
        if isinstance(injection_point, str):
            self.injection_point = {d: injection_point for d in (self.controllers or {})}
        else:
            self.injection_point = dict(injection_point)
        for d, ip in self.injection_point.items():
            if ip not in ("after_controller", "before_controller"):
                raise ValueError(
                    f"injection_point for {d!r} must be 'after_controller' or "
                    f"'before_controller', got {ip!r}"
                )
        self.response_delay_samples = int(response_delay_samples)
        self.saturate = None if saturate is None else float(saturate)
        self._cl = self._build_closed_loop() if self.controllers else {}
        # MIMO cross-coupling: off-diagonal plant terms H_{i<-j} so a drive on
        # input ``j`` shows up on output ``i``. Keys are (output_dof, input_dof);
        # the diagonal stays in ``plant.transfer_functions``. Default None -> pure
        # SISO (byte-identical to before).
        if coupling and self.controllers:
            raise NotImplementedError(
                "twin MIMO coupling is not supported together with closed-loop "
                "controllers (open-loop measurement only)"
            )
        self._ba_cross = {
            (out_dof, in_dof): sig.bilinear(tf.num, tf.den, self.fs)
            for (out_dof, in_dof), tf in (coupling or {}).items()
        }

    @classmethod
    def from_config(
        cls, config: dict, plant: SuspensionPlant, **kwargs
    ) -> "TwinBackend":
        """Build from a run config's ``channels`` section (``{dof: channel}``).

        Optional ``channels.drive`` / ``channels.error`` sections name the
        after-controller drive (``u``) and before-controller error (``e``)
        readback channels used for closed-loop identification.
        """
        ch = config["channels"]
        exc = {chan: dof for dof, chan in ch["excitation"].items()}
        rb = {chan: dof for dof, chan in ch["readback"].items()}
        drive = {chan: dof for dof, chan in ch.get("drive", {}).items()}
        error = {chan: dof for dof, chan in ch.get("error", {}).items()}
        return cls(plant, exc, rb, drive_channels=drive, error_channels=error, **kwargs)

    # -- channel API ---------------------------------------------------------
    def inject(self, channel: str, timeseries: np.ndarray, fs: float) -> None:
        if channel not in self.exc_channels:
            raise KeyError(f"unknown excitation channel {channel!r}")
        ts = np.asarray(timeseries, dtype=float)
        if not np.isclose(fs, self.fs):
            frac = Fraction(self.fs / fs).limit_denominator(1000)
            ts = sig.resample_poly(ts, frac.numerator, frac.denominator)
        if self.saturate is not None:
            ts = np.clip(ts, -self.saturate, self.saturate)
        self._drives[self.exc_channels[channel]] = ts

    def read(self, channels: list[str], duration: float) -> dict[str, np.ndarray]:
        n = int(round(duration * self.fs))
        out: dict[str, np.ndarray] = {}
        # In closed-loop mode the drive monitor (u) and the response (y) come from
        # one consistent loop solution with shared noise draws, so cache per DoF.
        pair_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        def _pair(dof: str) -> tuple[np.ndarray, np.ndarray]:
            if dof not in pair_cache:
                pair_cache[dof] = self._simulate_closed(dof, n)
            return pair_cache[dof]

        for ch in channels:
            if ch in self.readback_channels:
                dof = self.readback_channels[ch]
                if self.controllers and dof in self.controllers:
                    out[ch] = _pair(dof)[1]            # response y_meas (Y)
                else:
                    out[ch] = self._simulate(dof, n)
            elif ch in self.drive_channels:
                dof = self.drive_channels[ch]
                if self.controllers and dof in self.controllers:
                    out[ch] = _pair(dof)[0]            # after-controller drive u (X)
                else:
                    out[ch] = self._fit_length(self._drives.get(dof), n)
            elif ch in self.error_channels:
                dof = self.error_channels[ch]
                if self.controllers and dof in self.controllers:
                    u, y = _pair(dof)
                    # before-controller error e = (injected ref if before-C) − y_meas
                    a = (self._fit_length(self._drives.get(dof), n)
                         if self.injection_point.get(dof) == "before_controller"
                         else 0.0)
                    out[ch] = a - y
                else:
                    out[ch] = -self._simulate(dof, n)
            elif ch in self.exc_channels:
                dof = self.exc_channels[ch]
                if self.controllers and dof in self.controllers:
                    out[ch] = _pair(dof)[0]            # drive monitor u (twin convenience)
                else:
                    # monitoring the drive itself
                    drive = self._drives.get(dof)
                    out[ch] = self._fit_length(drive, n)
            else:
                raise KeyError(f"unknown channel {ch!r}")
        return out

    def ramp_down(self, channel: str, secs: float) -> None:
        if channel not in self.exc_channels:
            raise KeyError(f"unknown excitation channel {channel!r}")
        dof = self.exc_channels[channel]
        drive = self._drives.get(dof)
        if drive is None or len(drive) == 0:
            return
        n_ramp = min(int(round(secs * self.fs)), len(drive))
        ramped = np.zeros_like(drive)
        if n_ramp > 0:
            # half-cosine taper from full amplitude to zero, then silence
            taper = 0.5 * (1 + np.cos(np.pi * np.arange(n_ramp) / n_ramp))
            ramped[:n_ramp] = drive[:n_ramp] * taper
        self._drives[dof] = ramped

    # -- safe-state handoff support -----------------------------------------
    def snapshot_state(self, channels: list[str]) -> dict:
        """Capture the current per-DoF drive state for later restore."""
        return {"drives": {dof: d.copy() for dof, d in self._drives.items()}}

    def restore_state(self, snapshot: dict) -> None:
        """Restore the drive state captured by :meth:`snapshot_state`."""
        self._drives = {dof: d.copy() for dof, d in snapshot["drives"].items()}

    # -- internals -----------------------------------------------------------
    def _simulate(self, dof: str, n: int) -> np.ndarray:
        b, a = self._ba[dof]
        drive = self._drives.get(dof)
        u = self._fit_length(drive, n) if drive is not None else np.zeros(n)
        w = self._disturbance(n)
        resp = sig.lfilter(b, a, u + w)
        # MIMO cross-coupling: add every other input's contribution to this output
        for (out_dof, in_dof), (bc, ac) in self._ba_cross.items():
            if out_dof != dof:
                continue
            dj = self._drives.get(in_dof)
            if dj is not None:
                resp = resp + sig.lfilter(bc, ac, self._fit_length(dj, n))
        return self._delay(resp) + self._sensor_noise(n)

    def _build_closed_loop(self) -> dict:
        """Per-DoF discrete closed-loop filters mapping (excitation, disturbance,
        sensor noise) to the measured response ``y`` and the after-controller
        drive ``u``.

        With plant ``G = Gn/Gd``, controller ``C = Cn/Cd`` and ``D = Gd*Cd + Gn*Cn``,
        every map shares denominator ``D``.  The six numerators depend on where the
        excitation ``r`` is injected:

        **after_controller** (``u = r − C y``; the actuator-side injection):

            y = (Gn*Cd/D)(r + w) + (Gd*Cd/D) n
            u = (Gd*Cd/D) r − (Cn*Gn/D) w − (Cn*Gd/D) n

        **before_controller** (``u = C(r − y) + w``; EXC enters ahead of C):

            y = (Gn*Cn/D) r + (Gn*Cd/D) w + (Gd*Cd/D) n
            u = (Gd*Cn/D) r + (Gd*Cd/D) w − (Gd*Cn/D) n

        In both cases the reference-based FRF ``mean(Y)/mean(X)`` with ``X = u``,
        ``Y = y`` projects onto the deterministic ``r`` and recovers the open-loop
        ``G`` (``Y_r / U_r = Gn/Gd``).  Each numerator is discretised against ``D``
        (bilinear at ``fs``).
        """
        cl: dict[str, dict] = {}
        for dof, tf in self.plant.transfer_functions.items():
            if dof not in self.controllers:
                continue
            Gn, Gd = np.atleast_1d(tf.num), np.atleast_1d(tf.den)
            Cn, Cd = (np.atleast_1d(np.asarray(c, dtype=float))
                      for c in self.controllers[dof])
            D = np.polyadd(np.polymul(Gd, Cd), np.polymul(Gn, Cn))
            if self.injection_point.get(dof) == "before_controller":
                nums = {
                    "y_r": np.polymul(Gn, Cn),      # r -> y
                    "y_w": np.polymul(Gn, Cd),      # w -> y
                    "y_n": np.polymul(Gd, Cd),      # n -> y
                    "u_r": np.polymul(Gd, Cn),      # r -> u
                    "u_w": np.polymul(Gd, Cd),      # w -> u
                    "u_n": -np.polymul(Gd, Cn),     # n -> u
                }
            else:  # after_controller (default)
                nums = {
                    "y_r": np.polymul(Gn, Cd),      # r -> y
                    "y_w": np.polymul(Gn, Cd),      # w -> y
                    "y_n": np.polymul(Gd, Cd),      # n -> y
                    "u_r": np.polymul(Gd, Cd),      # r -> u
                    "u_w": -np.polymul(Cn, Gn),     # w -> u
                    "u_n": -np.polymul(Cn, Gd),     # n -> u
                }
            cl[dof] = {k: sig.bilinear(num, D, self.fs) for k, num in nums.items()}
        return cl

    def _simulate_closed(self, dof: str, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Closed-loop (u, y_meas) for ``n`` samples with shared noise draws."""
        f = self._cl[dof]
        drive = self._drives.get(dof)
        r = self._fit_length(drive, n) if drive is not None else np.zeros(n)
        w = self._disturbance(n)
        nz = self._sensor_noise(n)
        y = (sig.lfilter(*f["y_r"], r)
             + sig.lfilter(*f["y_w"], w)
             + sig.lfilter(*f["y_n"], nz))
        u = (sig.lfilter(*f["u_r"], r)
             + sig.lfilter(*f["u_w"], w)
             + sig.lfilter(*f["u_n"], nz))
        return u, self._delay(y)

    def _delay(self, x: np.ndarray) -> np.ndarray:
        """Apply the response-path transport delay (integer samples)."""
        d = self.response_delay_samples
        if d <= 0:
            return x
        out = np.zeros_like(x)
        out[d:] = x[:-d]
        return out

    def _sensor_noise(self, n: int) -> np.ndarray:
        if self.sensor_asd == 0.0:
            return np.zeros(n)
        # one-sided ASD A -> discrete white-noise std A*sqrt(fs/2)
        std = self.sensor_asd * np.sqrt(self.fs / 2.0)
        return self._rng.standard_normal(n) * std

    def _disturbance(self, n: int) -> np.ndarray:
        if self.disturbance_asd == 0.0:
            return np.zeros(n)
        # input-referred white noise: one-sided ASD A -> std A*sqrt(fs/2)
        std = self.disturbance_asd * np.sqrt(self.fs / 2.0)
        return self._rng.standard_normal(n) * std

    @staticmethod
    def _fit_length(x: np.ndarray | None, n: int) -> np.ndarray:
        if x is None:
            return np.zeros(n)
        if len(x) >= n:
            return np.asarray(x[:n], dtype=float)
        out = np.zeros(n)
        out[: len(x)] = x
        return out
