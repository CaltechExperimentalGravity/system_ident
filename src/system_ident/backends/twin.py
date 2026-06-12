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
        seed: int | np.random.Generator | None = None,
    ) -> None:
        self.plant = plant
        self.exc_channels = dict(exc_channels)
        self.readback_channels = dict(readback_channels)
        self.fs = float(fs if fs is not None else plant.fs)
        self.sensor_asd = float(sensor_asd)
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

    @classmethod
    def from_config(
        cls, config: dict, plant: SuspensionPlant, **kwargs
    ) -> "TwinBackend":
        """Build from a run config's ``channels`` section (``{dof: channel}``)."""
        ch = config["channels"]
        exc = {chan: dof for dof, chan in ch["excitation"].items()}
        rb = {chan: dof for dof, chan in ch["readback"].items()}
        return cls(plant, exc, rb, **kwargs)

    # -- channel API ---------------------------------------------------------
    def inject(self, channel: str, timeseries: np.ndarray, fs: float) -> None:
        if channel not in self.exc_channels:
            raise KeyError(f"unknown excitation channel {channel!r}")
        ts = np.asarray(timeseries, dtype=float)
        if not np.isclose(fs, self.fs):
            frac = Fraction(self.fs / fs).limit_denominator(1000)
            ts = sig.resample_poly(ts, frac.numerator, frac.denominator)
        self._drives[self.exc_channels[channel]] = ts

    def read(self, channels: list[str], duration: float) -> dict[str, np.ndarray]:
        n = int(round(duration * self.fs))
        out: dict[str, np.ndarray] = {}
        for ch in channels:
            if ch in self.readback_channels:
                out[ch] = self._simulate(self.readback_channels[ch], n)
            elif ch in self.exc_channels:
                # monitoring the drive itself
                drive = self._drives.get(self.exc_channels[ch])
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
        drive = self._drives.get(dof)
        if drive is None:
            resp = np.zeros(n)
        else:
            b, a = self._ba[dof]
            resp = self._fit_length(sig.lfilter(b, a, drive), n)
        return resp + self._sensor_noise(n)

    def _sensor_noise(self, n: int) -> np.ndarray:
        if self.sensor_asd == 0.0:
            return np.zeros(n)
        # one-sided ASD A -> discrete white-noise std A*sqrt(fs/2)
        std = self.sensor_asd * np.sqrt(self.fs / 2.0)
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
