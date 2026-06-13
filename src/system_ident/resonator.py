"""Physical resonator model: estimate in ``(f0, Q, gain)`` instead of polynomial
coefficients.

A ``ResonatorModel`` is a product of second-order resonances

    H(s) = gain / prod_i (s^2 + (w_i / Q_i) s + w_i^2),   w_i = 2*pi*f0_i

matching ``TFModel.from_resonances`` exactly (``num = [gain]``), so the same
config ``gain`` means the same thing here and in the twin. Each mode is a
physically meaningful ``(f0, Q)`` pair. Estimating in these parameters (rather
than the expanded num/den
coefficients) is far better conditioned for mechanical resonances and, crucially,
gives a gradient ``dH/df0`` that *directly* moves a resonance in frequency — so a
local Gauss-Newton/MAP step can relocate a peak that sits away from the prior,
which coefficient-space fitting cannot.

The parameter vector is ``theta = [f0_0..f0_{m-1}, Q_0..Q_{m-1}, gain]`` (all
free, no gauge). ``eval`` / ``jacobian`` / ``with_params`` give the numeric
surface the Fisher and Bayesian machinery need; ``to_tf`` converts to a
:class:`~system_ident.model.TFModel` for discretisation / Foton export.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .model import TFModel


@dataclass
class ResonatorModel:
    """Product-of-resonances transfer function parameterised by ``(f0, Q, gain)``."""

    f0: np.ndarray  # resonance frequencies [Hz], shape (m,)
    Q: np.ndarray   # quality factors, shape (m,)
    gain: float     # numerator constant (see module docstring)
    log: bool = False  # if True, the estimation params are (log f0, log Q, log|gain|)

    def __post_init__(self) -> None:
        self.f0 = np.atleast_1d(np.asarray(self.f0, dtype=float))
        self.Q = np.atleast_1d(np.asarray(self.Q, dtype=float))
        self.gain = float(self.gain)
        # sign of gain is carried separately so log|gain| can be a free param
        self._gain_sign = -1.0 if self.gain < 0 else 1.0
        if self.f0.shape != self.Q.shape:
            raise ValueError("f0 and Q must have the same length")

    @property
    def n_modes(self) -> int:
        return self.f0.size

    # -- parameter vector ----------------------------------------------------
    @property
    def params(self) -> np.ndarray:
        """Estimation parameter vector ``[f0.., Q.., gain]``.

        In ``log`` mode this is ``[log f0.., log Q.., log|gain|]`` — strictly
        positive, decade-spanning physical quantities are far better conditioned
        for a Gauss-Newton/MAP step in log-space (and a "fractional uncertainty"
        becomes literally the log-sigma). ``eval`` is always physical.
        """
        if self.log:
            return np.concatenate([np.log(self.f0), np.log(self.Q), [np.log(abs(self.gain))]])
        return np.concatenate([self.f0, self.Q, [self.gain]])

    def with_params(self, theta: np.ndarray) -> "ResonatorModel":
        """Rebuild from a parameter vector laid out like :attr:`params`."""
        theta = np.asarray(theta, dtype=float)
        m = self.n_modes
        if self.log:
            return ResonatorModel(
                f0=np.exp(theta[:m]), Q=np.exp(theta[m:2 * m]),
                gain=self._gain_sign * np.exp(theta[2 * m]), log=True,
            )
        return ResonatorModel(f0=theta[:m], Q=theta[m:2 * m], gain=theta[2 * m])

    @classmethod
    def from_resonances(
        cls, resonances: Sequence[tuple[float, float]], gain: float, log: bool = False
    ) -> "ResonatorModel":
        res = np.asarray(resonances, dtype=float).reshape(-1, 2)
        return cls(f0=res[:, 0], Q=res[:, 1], gain=gain, log=log)

    # -- numeric surface -----------------------------------------------------
    def eval(self, freq: np.ndarray) -> np.ndarray:
        """Complex frequency response on ``freq`` [Hz]."""
        s = 2j * np.pi * np.asarray(freq, dtype=float)
        w = 2.0 * np.pi * self.f0
        # num = gain (constant), den = prod of resonance factors -- matches
        # TFModel.from_resonances so ResonatorModel and the twin agree.
        H = np.full(s.shape, self.gain, dtype=complex)
        for wi, Qi in zip(w, self.Q):
            H = H / (s ** 2 + (wi / Qi) * s + wi ** 2)
        return H

    def jacobian(self, freq: np.ndarray, dpar: float = 1e-6) -> np.ndarray:
        """``dH/dtheta`` (complex, shape ``(n_par, n_bin)``) by central differences.

        ``dpar`` is a *relative* step; each parameter is perturbed by
        ``dpar * max(|theta_i|, 1)`` so the step scales with the (positive,
        order-of-magnitude-varying) physical parameters.
        """
        theta = self.params
        n_par = theta.size
        freq = np.asarray(freq, dtype=float)
        J = np.zeros((n_par, freq.size), dtype=complex)
        for i in range(n_par):
            h = dpar * max(abs(theta[i]), 1.0)
            tp = theta.copy(); tp[i] += h
            tm = theta.copy(); tm[i] -= h
            J[i] = (self.with_params(tp).eval(freq) - self.with_params(tm).eval(freq)) / (2.0 * h)
        return J

    # -- conversion ----------------------------------------------------------
    def to_tf(self) -> TFModel:
        """Expanded :class:`~system_ident.model.TFModel` (same response)."""
        return TFModel.from_resonances(list(zip(self.f0, self.Q)), self.gain)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        modes = ", ".join(f"({f:.4g}Hz,Q{q:.4g})" for f, q in zip(self.f0, self.Q))
        return f"ResonatorModel[{modes}] gain={self.gain:.4g}"


def resonator_from_tf(tf, log: bool = False) -> "ResonatorModel":
    """Extract a :class:`ResonatorModel` from a :class:`~system_ident.model.TFModel`.

    Reads ``(f0, Q)`` from each underdamped conjugate pole pair (roots of the
    denominator) and fits the gain by least squares so the responses match over a
    representative band. Used at the hybrid loop's locate-then-refine handoff
    (broadband_ls produces a TFModel; the Bayesian refinement wants a
    ResonatorModel).
    """
    den = np.asarray(tf.den, dtype=float)
    poles = np.roots(den)
    pairs = poles[poles.imag > 1e-9]            # one representative per conjugate pair
    if pairs.size == 0:
        raise ValueError("TFModel has no underdamped resonances to extract")
    w = np.abs(pairs)
    f0 = w / (2.0 * np.pi)
    Q = w / (2.0 * np.abs(pairs.real))
    # least-squares gain so the ResonatorModel response matches tf over a grid
    fmax = float(np.max(f0)) * 4.0 + 1.0
    grid = np.linspace(max(1e-3, float(np.min(f0)) * 0.1), fmax, 512)
    base = ResonatorModel(f0=f0, Q=Q, gain=1.0).eval(grid)
    target = tf.eval(grid)
    gain = float(np.real(np.vdot(base, target) / np.vdot(base, base)))
    return ResonatorModel(f0=f0, Q=Q, gain=gain, log=log)
