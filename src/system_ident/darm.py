"""A representative closed-loop DARM twin and the P&S recovery on it.

The DARM loop is  d_err = C·(x_free + x_pc + Σ κ_i N_i c_i)/(1+G),  G = C·A·D,
with sensing C (cavity pole + delay), three-stage actuation A, and a derived
servo D.  Because the sensing delay makes a rational time-domain loop
intractable, the twin synthesises the closed-loop response in the frequency
domain (exact for the periodic P&S multisine; the suspension resonances sit
below the measurement band, so the in-band dynamics are smooth).

All numbers are *representative of an Advanced-LIGO DARM loop, not a specific
interferometer state* — a single coupled-cavity pole + delay for C, three
pendulum-stage actuators, and a UGF≈50 Hz open-loop gain shaped for stability.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import TFModel


def sensing_model(freq, g_c: float, f_cc: float, tau: float) -> np.ndarray:
    """Optical sensing response C(f) = g_c/(1+i f/f_cc)·exp(-i 2π f τ) [ct/m]."""
    f = np.asarray(freq, dtype=float)
    return g_c / (1.0 + 1j * f / f_cc) * np.exp(-2j * np.pi * f * tau)


def _pendulum_stage(f_pend: float, q: float, gain: float) -> TFModel:
    """One quad actuation stage: a pendulum force→displacement TF [m/ct].

    In the DARM band (well above f_pend) this is the ~1/f² actuator rolloff; the
    resonance itself sits below the measurement band.
    """
    return TFModel.from_resonances([(f_pend, q)], gain)


@dataclass
class DARMLoop:
    """Representative closed-loop DARM twin (single loop, three actuation stages)."""

    fs: float = 4096.0
    fmin: float = 10.0
    fmax: float = 1500.0
    # sensing
    g_c: float = 1.0e6          # optical gain [ct/m]
    f_cc: float = 360.0         # coupled-cavity pole [Hz]
    tau: float = 77.0e-6        # light-travel / processing delay [s]
    # actuation: name -> (stage TFModel, kappa strength)
    stages: dict = field(default_factory=dict)
    # open-loop-gain shape (used to derive the servo D = G/(A·C))
    f_ugf: float = 50.0         # unity-gain frequency [Hz]
    f_hi: float = 400.0         # high-frequency control rolloff pole [Hz]
    # disturbance / sensing noise ASDs (set on the twin used for simulation)
    disturbance_asd: float = 0.0   # process (length) disturbance, [m/√Hz] referred to x_free
    sensor_asd: float = 0.0        # readout noise on d_err, [ct/√Hz]

    @classmethod
    def default(cls) -> "DARMLoop":
        stages = {
            "UIM": (_pendulum_stage(0.43, 300.0, 4.0e-7), 1.00),
            "PUM": (_pendulum_stage(1.00, 200.0, 8.0e-8), 0.40),
            "TST": (_pendulum_stage(3.40, 100.0, 1.2e-8), 0.08),
        }
        return cls(stages=stages)

    @property
    def ports(self) -> list[str]:
        return ["PCAL", "UIM", "PUM", "TST"]

    # -- elements ----------------------------------------------------------
    def C(self, freq) -> np.ndarray:
        return sensing_model(freq, self.g_c, self.f_cc, self.tau)

    def stage(self, name: str, freq) -> np.ndarray:
        tf, kappa = self.stages[name]
        return kappa * tf.eval(freq)

    def A(self, freq) -> np.ndarray:
        return sum(self.stage(n, freq) for n in self.stages)

    def _ol_shape(self, freq) -> np.ndarray:
        """The *designed* open-loop gain G(f): integrator to UGF, a control
        rolloff pole, and the sensing transport delay — shaped for a stable loop
        with healthy phase margin.  D is then derived so G = A·D·C exactly."""
        f = np.asarray(freq, dtype=float)
        return (self.f_ugf / (1j * f)) / (1.0 + 1j * f / self.f_hi) \
            * np.exp(-2j * np.pi * f * self.tau)

    def G(self, freq) -> np.ndarray:
        return self._ol_shape(freq)

    def D(self, freq) -> np.ndarray:
        """Representative digital servo, derived from the designed G: D = G/(A·C)."""
        return self.G(freq) / (self.A(freq) * self.C(freq))

    def R(self, freq) -> np.ndarray:
        """The calibration deliverable: counts→displacement response (1+G)/C."""
        return (1.0 + self.G(freq)) / self.C(freq)

    # -- closed-loop FRFs per injection point ------------------------------
    def frf_pcal(self, freq) -> np.ndarray:
        """d_err/x_pc = C/(1+G)  (Pcal displacement → DARM error)."""
        return self.C(freq) / (1.0 + self.G(freq))

    def frf_stage(self, name: str, freq) -> np.ndarray:
        """d_err/c_i = C·κ_i·N_i/(1+G)  (stage drive counts → DARM error)."""
        return self.C(freq) * self.stage(name, freq) / (1.0 + self.G(freq))

    def disturbance_to_derr(self, freq) -> np.ndarray:
        """x_free enters at the test mass like x_pc: C/(1+G)."""
        return self.frf_pcal(freq)

    def sensing_to_derr(self, freq) -> np.ndarray:
        """Readout noise n adds at d_err and is loop-suppressed: 1/(1+G)."""
        return 1.0 / (1.0 + self.G(freq))

    # -- simulation -----------------------------------------------------------
    def _white(self, asd: float, n: int, rng) -> np.ndarray:
        if asd == 0.0:
            return np.zeros(n)
        # one-sided ASD A -> discrete white-noise std A·sqrt(fs/2)
        return rng.standard_normal(n) * asd * np.sqrt(self.fs / 2.0)

    def simulate(self, drives: dict, n: int, rng) -> np.ndarray:
        """Synthesise d_err[n] for injected ``drives`` under process disturbance +
        sensing noise, by frequency-domain closed-loop filtering.

        Deterministic drives are periodic (P&S multisine), so rfft·H·irfft is the
        exact periodic steady-state response; the stochastic disturbance/sensing
        noise are coloured by their closed-loop transfer functions.
        """
        n = int(n)
        f = np.fft.rfftfreq(n, d=1.0 / self.fs)
        Y = np.zeros(len(f), dtype=complex)
        for port, x in drives.items():
            x = np.asarray(x, dtype=float)
            xf = np.zeros(n)
            xf[: min(len(x), n)] = x[: n]
            H = self.frf_pcal(f) if port == "PCAL" else self.frf_stage(port, f)
            H = np.where(np.isfinite(H), H, 0.0)
            Y += np.fft.rfft(xf) * H
        # process disturbance x_free -> d_err  (C/(1+G))
        if self.disturbance_asd:
            w = self._white(self.disturbance_asd, n, rng)
            Hd = np.where(np.isfinite(self.disturbance_to_derr(f)),
                          self.disturbance_to_derr(f), 0.0)
            Y += np.fft.rfft(w) * Hd
        # readout/sensing noise n -> d_err  (1/(1+G))
        if self.sensor_asd:
            v = self._white(self.sensor_asd, n, rng)
            Hs = np.where(np.isfinite(self.sensing_to_derr(f)),
                          self.sensing_to_derr(f), 0.0)
            Y += np.fft.rfft(v) * Hs
        return np.fft.irfft(Y, n)
