"""MIMO coupled+closed-loop twin backend (ChannelBackend).

Injects a multisine at each actuator, reads the drive monitors + sensors through the
live diagonal loops. Simulates CONSISTENTLY: u_drive = forced_response(Sd, U); then
y_sens = forced_response(Gd, u_drive) + sensor noise — so Y = Gd·X and the matrix
recovery is exact off-resonance.
"""
from __future__ import annotations
from fractions import Fraction
import numpy as np
import scipy.signal as sig
import control
from .base import ChannelBackend


class MIMOTwinBackend(ChannelBackend):
    def __init__(self, loop, exc_channels, drive_channels, sens_channels, *,
                 sensor_asd=0.0, process_asd=0.0, seed=None, ramp_s=3.0):
        self.loop = loop
        self.fs = float(loop.fs)
        self.exc = dict(exc_channels)      # name -> actuator index
        self.drv = dict(drive_channels)    # name -> actuator index (monitor)
        self.sen = dict(sens_channels)     # name -> sensor index
        self.sensor_asd = float(sensor_asd)
        self.process_asd = float(process_asd)
        self.ramp_s = float(ramp_s)
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._drives = {}                  # actuator index -> ramped drive
        self._cache = None                 # (n, u_drive, y_sens)

    def inject(self, channel, timeseries, fs):
        if channel not in self.exc:
            raise KeyError(channel)
        ts = np.asarray(timeseries, float)
        if not np.isclose(fs, self.fs):
            fr = Fraction(self.fs / fs).limit_denominator(1000)
            ts = sig.resample_poly(ts, fr.numerator, fr.denominator)
        self._drives[self.exc[channel]] = self._soft_start_stop(ts, self.fs)
        self._cache = None

    def _simulate(self, n):
        if self._cache is not None and self._cache[0] == n:
            return self._cache[1], self._cache[2]
        T = np.arange(n) / self.fs
        U = np.zeros((self.loop.n_act, n))
        for idx, d in self._drives.items():
            m = min(len(d), n)
            U[idx, :m] = d[:m]
        if self.process_asd:
            U = U + self._rng.standard_normal(U.shape) * self.process_asd * np.sqrt(self.fs / 2)
        u_drive = control.forced_response(self.loop.Sd, T, U).outputs.reshape(self.loop.n_act, n)
        y_sens = control.forced_response(self.loop.Gd, T, u_drive).outputs.reshape(self.loop.n_sens, n)
        if self.sensor_asd:
            y_sens = y_sens + self._rng.standard_normal(y_sens.shape) * self.sensor_asd * np.sqrt(self.fs / 2)
        self._cache = (n, u_drive, y_sens)
        return u_drive, y_sens

    def read(self, channels, duration):
        n = int(round(duration * self.fs))
        u_drive, y_sens = self._simulate(n)
        out = {}
        for ch in channels:
            if ch in self.sen:
                out[ch] = y_sens[self.sen[ch]]
            elif ch in self.drv:
                out[ch] = u_drive[self.drv[ch]]
            elif ch in self.exc:
                d = self._drives.get(self.exc[ch])
                out[ch] = (np.zeros(n) if d is None else np.r_[d, np.zeros(n)][:n])
            else:
                raise KeyError(ch)
        return out

    def ramp_down(self, channel, secs):
        if channel not in self.exc:
            raise KeyError(channel)
        d = self._drives.get(self.exc[channel])
        if d is None or not len(d):
            return
        nr = min(int(round(secs * self.fs)), len(d))
        out = np.zeros_like(d)
        if nr > 0:
            out[:nr] = d[:nr] * 0.5 * (1 + np.cos(np.pi * np.arange(nr) / nr))
        self._drives[self.exc[channel]] = out
        self._cache = None
