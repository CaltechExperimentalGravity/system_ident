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
    # Least-squares gain so the ResonatorModel response matches tf over a grid.
    # Fit on the MAGNITUDE: a noisy broadband_ls/invfreqs lock can land marginally
    # unstable (a right-half-plane pole pair) whose phase is opposite to the stable
    # resonator we build here (Q uses |Re pole| -> always stable). A complex inner
    # product then cancels at the resonance and collapses the gain toward zero. The
    # magnitude fit is phase-insensitive, always positive, and preserves |H|.
    fmax = float(np.max(f0)) * 4.0 + 1.0
    grid = np.linspace(max(1e-3, float(np.min(f0)) * 0.1), fmax, 512)
    base = np.abs(ResonatorModel(f0=f0, Q=Q, gain=1.0).eval(grid))
    target = np.abs(tf.eval(grid))
    gain = float(np.dot(base, target) / np.dot(base, base))
    return ResonatorModel(f0=f0, Q=Q, gain=gain, log=log)


def resonator_from_spectrum(freq, mag, f0_guess=None, min_bins=4.0, log=False):
    """Robust single-resonance estimate from a ``|H|`` spectrum via the half-power
    (-3 dB) bandwidth — the classic, fit-free way to read Q off a resonance.

    ``f0`` = (parabolically interpolated) peak frequency, ``Q = f0 / df_3dB`` where
    ``df_3dB`` is the full width where ``|H|`` falls to ``|H_peak| / sqrt(2)``, and
    ``gain = |H_peak| * w0**2 / Q`` (inverting ``|H(w0)| = gain * Q / w0**2`` for the
    single-mode :class:`ResonatorModel`).

    Why not least-squares? Fitting a resonator to the complex TF is fragile for
    sharp peaks: the spectral window distorts the peak *shape* and the SNR-weighted
    fit chases that distorted shoulder detail, biasing Q badly (it can diverge).
    The half-power width is set by the bins *around the peak* and is essentially
    unbiased once the peak is resolved. It does require resolution: the bandwidth
    ``f0/Q`` must span at least ``min_bins`` frequency bins (use a long enough Welch
    segment, ``T_seg >~ min_bins * Q / f0``), else the -3 dB points are not
    measurable and this raises ``ValueError``.

    Parameters
    ----------
    freq : (n,) ascending frequency grid [Hz].
    mag  : (n,) magnitude response ``|H|`` on ``freq``.
    f0_guess : optional prior peak frequency [Hz]; the peak is sought within
        ``[0.5, 1.5] * f0_guess`` so a noisy off-resonance bin cannot masquerade
        as the resonance.
    min_bins : minimum bins the -3 dB bandwidth must span to be trusted.
    log : passed through to the returned :class:`ResonatorModel`.

    Returns
    -------
    ResonatorModel  with a single ``(f0, Q, gain)`` mode.
    """
    fr = np.asarray(freq, dtype=float)
    m = np.asarray(mag, dtype=float)
    if fr.size < 3 or m.size != fr.size:
        raise ValueError("freq/mag must be matching arrays of length >= 3")
    df = fr[1] - fr[0]

    if f0_guess is not None:
        sel = np.where((fr > 0.5 * f0_guess) & (fr < 1.5 * f0_guess))[0]
        if sel.size == 0:
            raise ValueError("f0_guess lies outside the frequency grid")
        ipk = int(sel[np.argmax(m[sel])])
    else:
        ipk = int(np.argmax(m))
    if not 0 < ipk < len(m) - 1:
        raise ValueError("resonance peak is at the band edge — widen the band")

    # sub-bin peak by a parabola through the log-magnitude (Gaussian-ish peak)
    a, b, c = np.log(m[ipk - 1]), np.log(m[ipk]), np.log(m[ipk + 1])
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    f0 = fr[ipk] + delta * df
    peak = float(m[ipk])
    half = peak / np.sqrt(2.0)

    def _crossing(indices, step):
        for i in indices:
            if m[i] < half:               # interpolate between i and the bin toward the peak
                f1, f2, m1, m2 = fr[i], fr[i + step], m[i], m[i + step]
                return f1 + (half - m1) * (f2 - f1) / (m2 - m1)
        return np.nan

    f_lo = _crossing(range(ipk, 0, -1), +1)        # walk down in freq, step back up toward peak
    f_hi = _crossing(range(ipk, len(m) - 1), -1)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        raise ValueError("half-power points not found within the band (peak unresolved or band too narrow)")
    bw = f_hi - f_lo
    if bw < min_bins * df:
        raise ValueError(
            f"resonance under-resolved: -3dB bandwidth {bw:.4g} Hz spans "
            f"< {min_bins} bins (df={df:.4g} Hz). Use a longer Welch segment."
        )
    Q = f0 / bw
    w0 = 2.0 * np.pi * f0
    gain = peak * w0 ** 2 / Q
    return ResonatorModel(f0=np.array([f0]), Q=np.array([Q]), gain=gain, log=log)
