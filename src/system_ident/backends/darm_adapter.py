"""Digital-twin backend for the DARM loop: same channel API, backed by DARMLoop.

Injecting a multisine on Pcal or any actuation stage and reading the DARM error
synthesises the closed-loop response (frequency domain) under the loop's process
disturbance + sensing noise, so the existing P&S loop / FRF run unchanged.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import scipy.signal as sig

from ..darm import DARMLoop
from .base import ChannelBackend


class DARMBackend(ChannelBackend):
    def __init__(self, loop: DARMLoop, exc_channels: dict, derr_channel: str,
                 fs: float | None = None, seed=None, ramp_s: float = 3.0) -> None:
        self.loop = loop
        self.exc_channels = dict(exc_channels)           # channel -> port
        self.derr_channel = derr_channel
        self.fs = float(fs if fs is not None else loop.fs)
        self.ramp_s = float(ramp_s)
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._drives: dict[str, np.ndarray] = {}          # port -> ramped drive

    @classmethod
    def from_config(cls, config: dict, loop: DARMLoop, **kwargs) -> "DARMBackend":
        ch = config["channels"]
        exc = {chan: port for port, chan in ch["excitation"].items()}
        derr = ch["readback"]["DARM"]
        kwargs.setdefault("ramp_s", float(config.get("measurement", {}).get("t_ramp", 3.0)))
        return cls(loop, exc, derr, fs=float(config["measurement"]["fs"]), **kwargs)

    def inject(self, channel: str, timeseries: np.ndarray, fs: float) -> None:
        if channel not in self.exc_channels:
            raise KeyError(f"unknown excitation channel {channel!r}")
        ts = np.asarray(timeseries, dtype=float)
        if not np.isclose(fs, self.fs):
            frac = Fraction(self.fs / fs).limit_denominator(1000)
            ts = sig.resample_poly(ts, frac.numerator, frac.denominator)
        self._drives[self.exc_channels[channel]] = self._soft_start_stop(ts, self.fs)

    def read(self, channels: list[str], duration: float) -> dict[str, np.ndarray]:
        n = int(round(duration * self.fs))
        out: dict[str, np.ndarray] = {}
        derr = None
        for ch in channels:
            if ch == self.derr_channel:
                if derr is None:
                    derr = self.loop.simulate(self._drives, n, self._rng)
                out[ch] = derr
            elif ch in self.exc_channels:
                out[ch] = self._fit_length(self._drives.get(self.exc_channels[ch]), n)
            else:
                raise KeyError(f"unknown channel {ch!r}")
        return out

    def ramp_down(self, channel: str, secs: float) -> None:
        if channel not in self.exc_channels:
            raise KeyError(f"unknown excitation channel {channel!r}")
        port = self.exc_channels[channel]
        drive = self._drives.get(port)
        if drive is None or len(drive) == 0:
            return
        n_ramp = min(int(round(secs * self.fs)), len(drive))
        ramped = np.zeros_like(drive)
        if n_ramp > 0:
            taper = 0.5 * (1 + np.cos(np.pi * np.arange(n_ramp) / n_ramp))
            ramped[:n_ramp] = drive[:n_ramp] * taper
        self._drives[port] = ramped

    @staticmethod
    def _fit_length(x, n):
        if x is None:
            return np.zeros(n)
        if len(x) >= n:
            return np.asarray(x[:n], dtype=float)
        out = np.zeros(n)
        out[: len(x)] = x
        return out
