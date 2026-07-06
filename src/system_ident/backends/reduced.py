"""Frequency-domain backend over a ReducedStateSpacePlant (ChannelBackend).

Drives the P&S pipeline like MIMOTwinBackend, but synthesizes the steady-state periodic
response in the frequency domain — Ŷ = G·X̂ on the rFFT grid — so it needs only the plant
FRF (numpy), no `control`/`slycot`. Valid because the injected multisine is periodic over
the record: its steady-state response is exactly G·X on the excited lines.
"""
from __future__ import annotations
from fractions import Fraction
import numpy as np
import scipy.signal as sig
from .base import ChannelBackend


class ReducedPlantBackend(ChannelBackend):
    def __init__(self, plant, exc_channels, sens_channels, *,
                 fs, sensor_asd=0.0, seed=None, ramp_s=3.0):
        self.plant = plant
        self.fs = float(fs)
        self.exc = {ch: plant.inputs.index(lbl) for ch, lbl in exc_channels.items()}
        self.sen = {ch: plant.outputs.index(lbl) for ch, lbl in sens_channels.items()}
        self.sensor_asd = float(sensor_asd)
        self.ramp_s = float(ramp_s)
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._drives = {}     # input index -> ramped drive
        self._cache = None    # (n, y_sens (n_out, n))

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
            return self._cache[1]
        n_in, n_out = self.plant.B.shape[1], self.plant.C.shape[0]
        X = np.zeros((n_in, n))
        for idx, d in self._drives.items():
            m = min(len(d), n)
            X[idx, :m] = d[:m]
        Xhat = np.fft.rfft(X, axis=1)                      # (n_in, nf)
        fa = np.fft.rfftfreq(n, 1 / self.fs)
        # evaluate G only where some input has power (multisine is sparse)
        active = np.where(np.any(np.abs(Xhat) > 0, axis=0))[0]
        Yhat = np.zeros((n_out, Xhat.shape[1]), complex)
        if len(active):
            G = self.plant.eval(fa[active])                # (F, n_out, n_in)
            # per active bin b: Yhat[:, b] = G[b] @ Xhat[:, b]
            Yhat[:, active] = np.einsum("foi,if->of", G, Xhat[:, active])
        y_sens = np.fft.irfft(Yhat, n=n, axis=1)
        if self.sensor_asd:
            y_sens = y_sens + self._rng.standard_normal(y_sens.shape) * self.sensor_asd * np.sqrt(self.fs / 2)
        self._cache = (n, y_sens)
        return y_sens

    def read(self, channels, duration):
        n = int(round(duration * self.fs))
        y_sens = self._simulate(n)
        out = {}
        for ch in channels:
            if ch in self.sen:
                out[ch] = y_sens[self.sen[ch]]
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
