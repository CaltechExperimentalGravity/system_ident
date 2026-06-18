"""Canonical transfer-function model.

``TFModel`` is the one parameter representation shared across the package, so an
estimator, an input designer, and the Fisher calculation all
speak the same language. It mirrors the ``{num, den}`` dict convention used by
``sys_id_dev/sysIDlib.py`` (``unpack_par_dict`` / ``pack_par_to_dict``) so the
existing engine can be wrapped without translation layers.

The numeric surface (``eval``, ``jacobian``) and the resonance/zpk constructors
are the ported, modernised equivalents of ``sysIDlib``'s ``par_dict_to_TF_vect``
and ``get_res_g_pole_pair`` — validated against that engine in
``tests/test_step2_validation.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.signal as sig


def resonance_pole_pair(f0: float, Q: float) -> tuple[float, float]:
    """Complex-pole (real, imag) parts for a resonance at ``f0`` [Hz], quality ``Q``.

    Port of ``sysIDlib.get_res_g_pole_pair``: the pole is ``a ± 1j*b`` with
    ``a = -w0/(2Q)`` and ``b = sqrt(w0**2 - a**2)`` for ``w0 = 2*pi*f0``.
    """
    w0 = 2.0 * np.pi * f0
    a = -w0 / (2.0 * Q)
    b = np.sqrt(w0**2 - a**2)
    return a, b


def pole_pair_f0_Q(a: float, b: float) -> tuple[float, float]:
    """Inverse of :func:`resonance_pole_pair` (port of ``sysIDlib.get_res_f0_Q``)."""
    w0 = np.sqrt(a**2 + b**2)
    f0 = w0 / (2.0 * np.pi)
    Q = -w0 / (2.0 * a)
    return f0, Q


@dataclass
class TFModel:
    """A SISO transfer function H(s) = num(s) / den(s), highest power first.

    Parameters
    ----------
    num, den:
        Real polynomial coefficients (numerator, denominator), highest power
        first — matching ``scipy.signal.freqs`` and the ``sysIDlib`` ``{num,
        den}`` convention.
    """

    num: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    den: np.ndarray = field(default_factory=lambda: np.array([1.0]))

    def __post_init__(self) -> None:
        self.num = np.atleast_1d(np.asarray(self.num, dtype=float))
        self.den = np.atleast_1d(np.asarray(self.den, dtype=float))

    # -- canonical {num, den} dict interop (sysIDlib convention) -------------
    @classmethod
    def from_dict(cls, par_dict: dict) -> "TFModel":
        """Build from a ``{"num": [...], "den": [...]}`` dict."""
        return cls(num=par_dict["num"], den=par_dict["den"])

    def to_dict(self) -> dict:
        """Return the ``{"num", "den"}`` dict the sysIDlib engine consumes."""
        return {"num": self.num.copy(), "den": self.den.copy()}

    # -- zpk / resonance construction ---------------------------------------
    @classmethod
    def from_zpk(cls, zeros, poles, gain: float) -> "TFModel":
        """Build from zeros/poles/gain (continuous-time), via ``scipy.signal.zpk2tf``."""
        num, den = sig.zpk2tf(np.asarray(zeros), np.asarray(poles), gain)
        return cls(num=num, den=den)

    @classmethod
    def from_resonances(
        cls,
        resonances: Sequence[tuple[float, float]],
        gain: float,
        zeros: Sequence[complex] = (),
    ) -> "TFModel":
        """Build a resonant plant TF from ``(f0, Q)`` pairs and an overall ``gain``.

        Each resonance contributes a conjugate pole pair via
        :func:`resonance_pole_pair`. This reproduces the way the
        ``double_pend_demo`` constructs its plant.
        """
        poles: list[complex] = []
        for f0, Q in resonances:
            a, b = resonance_pole_pair(f0, Q)
            poles.extend((a + 1j * b, a - 1j * b))
        return cls.from_zpk(zeros, poles, gain)

    # -- parameter-vector interop (sysIDlib unpack/pack convention) ---------
    @property
    def params(self) -> np.ndarray:
        """Stacked ``[num, den]`` parameter vector (``sysIDlib.unpack_par_dict``)."""
        return np.concatenate([self.num, self.den])

    @property
    def n_num(self) -> int:
        return len(self.num)

    def with_params(self, theta: np.ndarray) -> "TFModel":
        """Rebuild from a flat ``[num, den]`` parameter vector (protocol complement of :attr:`params`)."""
        theta = np.asarray(theta, dtype=float)
        n_num = self.n_num
        return TFModel(num=theta[:n_num], den=theta[n_num:])

    def to_tf(self) -> "TFModel":
        """Identity — returns ``self`` (completes the four-method protocol shared with ``ResonatorModel``)."""
        return self

    # -- numeric surface -----------------------------------------------------
    def eval(self, freq: np.ndarray) -> np.ndarray:
        """Evaluate the complex response ``H(2j*pi*freq)`` over ``freq`` [Hz].

        Port of ``sysIDlib.par_dict_to_TF_vect`` (``scipy.signal.freqs`` on the
        analog ``num``/``den`` at angular frequency ``2*pi*freq``).
        """
        freq = np.asarray(freq, dtype=float)
        _, H = sig.freqs(self.num, self.den, worN=2.0 * np.pi * freq)
        return H

    def jacobian(
        self,
        freq: np.ndarray,
        dpar: float | np.ndarray = 1e-8,
        logflag: np.ndarray | None = None,
    ) -> np.ndarray:
        """Numerical ``dH/d(params)`` over ``freq``; shape ``(n_par, len(freq))``.

        Central differences on the stacked ``[num, den]`` parameter vector,
        matching the derivative ``sysIDlib.get_Fisher_from_psd`` forms
        internally. Where ``logflag[i]`` is truthy the column is scaled by the
        parameter value, giving the derivative w.r.t. a fractional change.
        """
        freq = np.asarray(freq, dtype=float)
        par = self.params.astype(float)
        n_par = len(par)
        dpar = np.broadcast_to(np.asarray(dpar, dtype=float), (n_par,))
        n_num = self.n_num

        dH = np.zeros((n_par, len(freq)), dtype=np.complex128)
        for i in range(n_par):
            up, lo = par.copy(), par.copy()
            up[i] += dpar[i]
            lo[i] -= dpar[i]
            Hu = TFModel(num=up[:n_num], den=up[n_num:]).eval(freq)
            Hl = TFModel(num=lo[:n_num], den=lo[n_num:]).eval(freq)
            dH[i, :] = (Hu - Hl) / (2.0 * dpar[i])
            if logflag is not None and logflag[i]:
                dH[i, :] *= par[i]
        return dH

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TFModel(num={self.num.tolist()}, den={self.den.tolist()})"
