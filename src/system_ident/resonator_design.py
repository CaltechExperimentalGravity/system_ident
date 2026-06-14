"""Gauge-free Fisher information and optimal excitation for ResonatorModel.

Mirror of ``fisher.py`` / ``design/pintelon.py`` but operating on any model
that exposes the four-method protocol::

    .params            -> theta  (n_par,)
    .jacobian(freq)    -> J      (n_par, n_bin)  complex
    .eval(freq)        -> H      (n_bin,)
    .with_params(theta)-> model

No gauge row/column is removed (``ResonatorModel`` is fully identifiable).
The TFModel-gauge versions in ``fisher.py`` / ``design/pintelon.py`` remain
unchanged and are used by the loop's TFModel path.

Functions
---------
fisher_information(freq, model, Pxx, Pyy, T_tot) -> (n_par, n_par) ndarray
dispersion(freq, model, Pxx, Pyy)                -> (nu, gamma)
optimal_excitation(freq, model, Pyy, Px_tot, ...) -> Pxx ndarray
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid


# ---------------------------------------------------------------------------
# Fisher information (gauge-free)
# ---------------------------------------------------------------------------

def fisher_information(
    freq: np.ndarray,
    model,
    Pxx: np.ndarray,
    Pyy: np.ndarray,
    T_tot: float,
) -> np.ndarray:
    """Gauge-free Fisher information matrix for the model parameters.

    ``gamma[i,j] = 2 * Re{ trapezoid(conj(J_i) * J_j * Pxx/Pyy, freq) } * T_tot``

    where ``J = model.jacobian(freq)`` (shape ``(n_par, n_bin)``).  No row/column
    is dropped — ``ResonatorModel`` params are identifiable as supplied.

    Parameters
    ----------
    freq : (n_bin,) frequency grid [Hz].
    model : model with ``.jacobian(freq)`` returning ``(n_par, n_bin)`` complex.
    Pxx : (n_bin,) excitation PSD.
    Pyy : (n_bin,) quiet-time readout PSD (denominator).
    T_tot : total measurement time [s].

    Returns
    -------
    gamma : (n_par, n_par) Fisher matrix, symmetric positive (semi)definite.
    """
    freq = np.asarray(freq, dtype=float)
    Pxx = np.asarray(Pxx, dtype=float)
    Pyy = np.asarray(Pyy, dtype=float)

    J = model.jacobian(freq)      # (n_par, n_bin), complex
    n_par = J.shape[0]
    weight = Pxx / Pyy

    gamma = np.zeros((n_par, n_par))
    for i in range(n_par):
        for j in range(i, n_par):
            val = 2.0 * float(np.real(
                trapezoid(np.conj(J[i]) * J[j] * weight, freq)
            ))
            gamma[i, j] = val
            gamma[j, i] = val

    return gamma * T_tot


def fisher_information_per_freq(
    freq: np.ndarray,
    model,
    Pxx: np.ndarray,
    Pyy: np.ndarray,
    T_tot: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(gamma, gamma_vs_freq)`` for the dispersion function.

    ``gamma_vs_freq[i,j,k]`` is the per-frequency integrand at ``freq[k]``.
    """
    freq = np.asarray(freq, dtype=float)
    Pxx = np.asarray(Pxx, dtype=float)
    Pyy = np.asarray(Pyy, dtype=float)

    J = model.jacobian(freq)
    n_par, n_bin = J.shape
    weight = Pxx / Pyy

    dens = np.zeros((n_par, n_par, n_bin))
    for i in range(n_par):
        for j in range(i, n_par):
            d = 2.0 * np.real(np.conj(J[i]) * J[j]) * weight
            dens[i, j] = d
            dens[j, i] = d

    gamma = np.array([
        [trapezoid(dens[i, j], freq) for j in range(n_par)]
        for i in range(n_par)
    ]) * T_tot

    return gamma, dens * T_tot


# ---------------------------------------------------------------------------
# Dispersion function (gauge-free)
# ---------------------------------------------------------------------------

def dispersion(
    freq: np.ndarray,
    model,
    Pxx: np.ndarray,
    Pyy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauge-free dispersion function ``nu(freq)`` (Pintelon & Schoukens §5.4.2).

    ``nu[k] = trace(Sigma @ dens[:,:,k]) * Pxx_tot / Pxx[k]``

    where ``Sigma = gamma^{-1}`` is the parameter covariance and
    ``T_tot = 1/df`` (``n_avg=1``; the overall time scaling cancels in ``nu``).

    Returns ``(nu, gamma)``.
    """
    freq = np.asarray(freq, dtype=float)
    df = freq[1] - freq[0]
    gamma, dens = fisher_information_per_freq(
        freq, model, Pxx, Pyy, T_tot=1.0 / df
    )
    sigma = np.linalg.inv(gamma)
    Pxx_tot = float(np.sum(Pxx))
    Pxx = np.asarray(Pxx, dtype=float)

    nu = np.array([
        float(np.trace(sigma @ dens[:, :, k]) * Pxx_tot / max(Pxx[k], 1e-300))
        for k in range(len(freq))
    ])
    return nu, gamma


# ---------------------------------------------------------------------------
# Optimal excitation  (gauge-free, mirrors design/pintelon.py)
# ---------------------------------------------------------------------------

def optimal_excitation(
    freq: np.ndarray,
    model,
    Pyy: np.ndarray,
    Px_tot: float,
    Pxx: np.ndarray | None = None,
    n_iter: int = 3,
    rec_progress: bool = False,
):
    """Iteratively optimise the excitation PSD via the dispersion-function
    fixed point (gauge-free version for ResonatorModel).

    Mirrors ``design.pintelon.optimal_excitation`` but uses the gauge-free
    :func:`dispersion` function above.  Starting from a flat (or supplied)
    ``Pxx``, each iteration reweights toward high-``nu`` bins and renormalises
    to ``Px_tot``.

    With ``rec_progress=False`` returns the final ``Pxx``.
    With ``rec_progress=True`` returns ``(Pxx_rec, nu_rec, gamma_rec)``.
    """
    freq = np.asarray(freq, dtype=float)
    n_bin = len(freq)
    n_par = len(model.params)

    if Pxx is None:
        Pxx = np.ones(n_bin, dtype=float)
    Pxx = np.asarray(Pxx, dtype=float).copy()
    Pxx *= Px_tot / trapezoid(Pxx, freq)

    if rec_progress:
        Pxx_rec = np.zeros((n_iter, n_bin))
        nu_rec = np.zeros((n_iter, n_bin))
        gamma_rec = np.zeros((n_iter, n_par, n_par))

    for cnt in range(n_iter):
        nu, gamma = dispersion(freq, model, Pxx, Pyy)
        Pxx = Pxx * nu
        Pxx = np.clip(Pxx, 0.0, None)   # safety: nu can be tiny-negative due to numerics
        integral = trapezoid(Pxx, freq)
        if integral <= 0:
            Pxx = np.ones(n_bin, dtype=float)
        Pxx *= Px_tot / trapezoid(Pxx, freq)
        if rec_progress:
            Pxx_rec[cnt] = Pxx
            nu_rec[cnt] = nu
            gamma_rec[cnt] = gamma

    if rec_progress:
        return Pxx_rec, nu_rec, gamma_rec
    return Pxx


def prior_robust_excitation(
    freq: np.ndarray,
    model,
    Pyy: np.ndarray,
    Px_tot: float,
    prior_uncertainty: float,
    n_iter: int = 3,
    n_samples: int = 7,
):
    """Prior-uncertainty-aware excitation for a :class:`ResonatorModel`.

    With only a ``±prior_uncertainty`` (fractional) guess of the resonance
    frequency, concentrating the limited drive at the point estimate risks
    missing the true resonance. This averages the optimal excitation over
    plausible resonance frequencies ``f0·(1 ± prior_uncertainty)`` so the drive
    covers everywhere the resonance could be: efficient (not flat broadband) AND
    robust (the prior's spread sets the drive's spread). ``prior_uncertainty=0``
    reduces to the point-optimal drive; larger values broaden it.
    """
    from .resonator import ResonatorModel

    u = float(prior_uncertainty)
    if u <= 0.0:
        return optimal_excitation(freq, model, Pyy, Px_tot, n_iter=n_iter)

    freq = np.asarray(freq, dtype=float)
    scales = np.linspace(1.0 - u, 1.0 + u, n_samples)
    acc = np.zeros(len(freq), dtype=float)
    for sc in scales:
        m_k = ResonatorModel(
            f0=np.asarray(model.f0) * sc, Q=np.asarray(model.Q),
            gain=model.gain, log=getattr(model, "log", False),
        )
        acc += optimal_excitation(freq, m_k, Pyy, Px_tot, n_iter=n_iter)
    integral = trapezoid(acc, freq)
    if integral > 0:
        acc *= Px_tot / integral
    return acc
