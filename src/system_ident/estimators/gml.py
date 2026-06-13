"""Gaussian maximum-likelihood estimator (Pintelon-Schoukens parametric fit).

The P&S optimal estimator: minimise ``sum |H_meas - H(theta)|^2 / H_err^2`` to
convergence by Gauss-Newton / Levenberg-Marquardt, which is asymptotically
unbiased and attains the Cramer-Rao bound when ``H_err`` is the true per-bin
noise.  Two model parameterisations are supported through the shared
four-method protocol:

* :class:`~system_ident.resonator.ResonatorModel` (physical ``f0, Q, gain``) —
  fit directly; ``dH/df0`` relocates a peak and the problem is well conditioned
  and naturally multi-mode.
* :class:`~system_ident.model.TFModel` (``num/den`` coefficients) — a
  Sanathanan-Koerner linearisation (re-weighted ``invfreqs``, updating the
  denominator each iterate instead of freezing it at the prior, which is the
  plain-``invfreqs`` bias) gives a good starting point, then a Gauss-Newton /
  LM polish on the exact ML objective finishes the fit.
"""

from __future__ import annotations

import numpy as np

from ..model import TFModel
from .base import Estimator
from .bayesian import ml_fit
from .invfreqs import invfreqs


def _sanathanan_koerner(freq, H_meas, w, model, n_iter=10):
    """Sanathanan-Koerner iterations for a ``TFModel`` starting from ``model``.

    Each iterate solves the linear ``invfreqs`` problem with weight
    ``sqrt(w)/|A^(m-1)(jw)|`` so the minimised residual approximates the true
    ``sum w |H - B/A|^2``; updating ``A`` every step (rather than freezing it at
    the prior) removes the plain-``invfreqs`` bias.
    """
    freq = np.asarray(freq, dtype=float)
    jw = 2j * np.pi * freq
    n_num = model.n_num
    n_den = len(model.den)
    sqrt_w = np.sqrt(np.clip(w, 0.0, None))
    A = np.asarray(model.den, dtype=float)
    out = model
    for _ in range(n_iter):
        Aval = np.polyval(A, jw)
        wt = sqrt_w / np.maximum(np.abs(Aval), 1e-30)
        num, den = invfreqs(2.0 * np.pi * freq, H_meas, wt, n_den - 1)
        num = num[-n_num:]
        num = num / den[-n_den]
        den = den / den[-n_den]
        out = TFModel(num=num, den=den)
        A = out.den
    return out


class GMLEstimator(Estimator):
    """Maximum-likelihood plant fit (Gauss-Newton / Levenberg-Marquardt)."""

    def fit(
        self,
        freq: np.ndarray,
        H_meas: np.ndarray,
        H_err: np.ndarray,
        model: TFModel,
    ):
        freq = np.asarray(freq, dtype=float)
        H_meas = np.asarray(H_meas, dtype=complex)
        H_err = np.asarray(H_err, dtype=float)

        if hasattr(model, "f0"):
            # ResonatorModel: direct ML in physical (f0, Q, gain) parameters.
            fitted, _ = ml_fit(freq, H_meas, H_err, model)
            return fitted

        # TFModel: SK linearisation for a good start, then exact-objective polish.
        valid = np.isfinite(H_err) & (H_err > 0) & np.isfinite(H_meas)
        w = np.where(valid, 1.0 / np.where(valid, H_err, 1.0) ** 2, 0.0)
        start = _sanathanan_koerner(freq, H_meas, w, model)
        fitted, _ = ml_fit(freq, H_meas, H_err, start)
        # normalise the gauge (monic denominator) like the other estimators
        den0 = fitted.den[0]
        return TFModel(num=fitted.num / den0, den=fitted.den / den0)
