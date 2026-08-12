"""Fisher information / parameter covariance.

Ported (modernised for numpy 2.x / scipy 1.x) from
``sys_id_dev/sysIDlib.py::get_Fisher_from_psd``: turns (model, input PSD, output
PSD, measurement time) into a Fisher matrix and hence parameter uncertainties —
which drive both the optimal-excitation design and the dashboard's convergence
display and "uncertainty below target" DONE criterion.

The math follows LIGO-G2101503 (p19). The denominator's leading coefficient is
held fixed at 1 (the standard TF gauge), so its row/column is removed from the
returned matrices; the remaining parameter order is ``[num, den[1:]]``.

Validated bit-for-bit against the legacy engine in
``tests/test_step2_validation.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import trapezoid

from .model import TFModel

# Never let Pyy hit exactly 0: weight = Pxx / Pyy below goes to inf at a
# zero bin (a disconnected/dead channel gives Welch a flat, exactly-zero
# PSD), which poisons the Fisher matrix, its inverse, and the whole
# excitation-design fixed point downstream -- the "divide by 0" this floor
# exists to prevent. Absolute, not relative to Pyy's own peak: a genuinely
# all-zero Pyy has no peak to be relative to, which is exactly the case this
# needs to survive.
_PYY_FLOOR = 1e-30


def _default_logflag(n_par: int) -> np.ndarray:
    """Legacy default: fractional error on the last parameter, absolute on the rest."""
    logflag = np.zeros(n_par, dtype=bool)
    logflag[-1] = True
    return logflag


def fisher_matrix(
    freq: np.ndarray,
    model: TFModel,
    Pxx: np.ndarray,
    Pyy: np.ndarray,
    T_tot: float,
    dpar: float | np.ndarray = 1e-8,
    logflag: np.ndarray | None = None,
    return_per_freq: bool = False,
):
    """Fisher information matrix for the model parameters.

    Parameters
    ----------
    freq:
        Frequency grid [Hz].
    model:
        Current model whose parameter sensitivities set the information content.
    Pxx:
        PSD of the excitation over ``freq``.
    Pyy:
        Quiet-time PSD of the readout (no injection) over ``freq``.
    T_tot:
        Total measurement time [s].
    dpar, logflag:
        Finite-difference step and per-parameter fractional/absolute flags; see
        :meth:`TFModel.jacobian`. Defaults match ``sysIDlib`` (1e-8 step,
        fractional error on the trailing parameter only).
    return_per_freq:
        If true, also return the per-frequency information density
        ``gamma_vs_freq`` (shape ``(n, n, len(freq))``), used by the
        dispersion-function excitation design.

    Returns
    -------
    gamma or (gamma, gamma_vs_freq):
        The ``(n, n)`` Fisher matrix, with the fixed ``den[0]`` row/column
        removed (so ``n = len(num) + len(den) - 1``).
    """
    freq = np.asarray(freq, dtype=float)
    n_num = model.n_num
    par = model.params.astype(float)
    # Gauge: hold the leading denominator coefficient fixed at 1.
    par = par / par[n_num]
    n_par = len(par)

    if logflag is None:
        logflag = _default_logflag(n_par)

    gauged = TFModel(num=par[:n_num], den=par[n_num:])
    dH = gauged.jacobian(freq, dpar=dpar, logflag=logflag)
    # The fixed parameter contributes nothing to the information.
    dH[n_num, :] = 0.0

    Pyy = np.asarray(Pyy, dtype=float)
    if np.any(Pyy <= 0):
        warnings.warn(
            "Pyy (quiet-time noise PSD) has a zero or negative bin -- likely a "
            "disconnected/dead channel, or a genuinely zero-noise test signal. "
            "Flooring at a tiny value to keep the Fisher-information weighting "
            "well-posed; this makes excitation design behave as if there IS an "
            "extremely small noise floor, not literal zero -- on a REAL channel "
            "this usually means it isn't actually live, worth checking.",
            stacklevel=2,
        )
    weight = Pxx / np.maximum(Pyy, _PYY_FLOOR)

    gamma = np.zeros((n_par, n_par))
    for i in range(n_par):
        for j in range(i, n_par):
            gamma[i, j] = 2.0 * np.real(
                trapezoid(np.conj(dH[i]) * dH[j] * weight, freq)
            )
            gamma[j, i] = gamma[i, j]

    keep = [k for k in range(n_par) if k != n_num]
    gamma = gamma[np.ix_(keep, keep)] * T_tot

    if not return_per_freq:
        return gamma

    n_bin = len(freq)
    dens = np.zeros((n_par, n_par, n_bin))
    for i in range(n_par):
        for j in range(i, n_par):
            dens[i, j, :] = 2.0 * np.real(np.conj(dH[i]) * dH[j]) * weight
            dens[j, i, :] = dens[i, j, :]
    dens = dens[np.ix_(keep, keep)] * T_tot
    return gamma, dens


def parameter_covariance(
    freq: np.ndarray,
    model: TFModel,
    Pxx: np.ndarray,
    Pyy: np.ndarray,
    T_tot: float,
    dpar: float | np.ndarray = 1e-8,
    logflag: np.ndarray | None = None,
) -> np.ndarray:
    """Parameter covariance, the inverse of :func:`fisher_matrix`."""
    gamma = fisher_matrix(
        freq, model, Pxx, Pyy, T_tot, dpar=dpar, logflag=logflag
    )
    return np.linalg.inv(gamma)


def safe_inverse(m: np.ndarray) -> np.ndarray:
    """Inverse of a Fisher/covariance matrix, robust to rank deficiency.

    A full-rank matrix inverts exactly (preserves the sysIDlib bit-for-bit path).
    High-order plants with near-cancelling pole/zero pairs (e.g. realistic
    suspension cascades) give a rank-deficient Fisher; the Moore–Penrose
    pseudo-inverse is the standard handling, returning the minimum-norm result so
    the P&S optimal-excitation design and the CRB still go through.
    """
    try:
        out = np.linalg.inv(m)
        if not np.all(np.isfinite(out)):
            raise np.linalg.LinAlgError("non-finite inverse")
        return out
    except np.linalg.LinAlgError:
        pass
    m = np.nan_to_num(np.asarray(m, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return np.linalg.pinv(m)
    except np.linalg.LinAlgError:
        # pathological (SVD non-convergence): add a tiny diagonal ridge and retry
        eps = 1e-12 * (np.trace(m) / max(len(m), 1) + 1e-300)
        return np.linalg.pinv(m + eps * np.eye(len(m)))


def dispersion(
    freq: np.ndarray,
    model: TFModel,
    Pxx: np.ndarray,
    Pyy: np.ndarray,
    dpar: float | np.ndarray = 1e-8,
    logflag: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispersion function ``nu(freq)`` (Pintelon & Schoukens, Sec. 5.4.2).

    Port of ``sysIDlib.get_dispersion``. ``nu`` measures how much each frequency
    bin contributes to the total parameter information given the current
    excitation ``Pxx``; reallocating drive power toward high-``nu`` bins is the
    fixed point that maximises information for a fixed power budget.

    Returns ``(nu, gamma)`` where ``gamma`` is the Fisher matrix. As in the
    legacy code the Fisher here is evaluated at ``T_tot = 1/df`` (``n_avg=1``);
    any overall time scaling cancels in ``nu`` since it uses
    ``sigma = gamma^-1``.
    """
    freq = np.asarray(freq, dtype=float)
    df = freq[1] - freq[0]
    gamma, dens = fisher_matrix(
        freq, model, Pxx, Pyy, T_tot=1.0 / df, dpar=dpar, logflag=logflag,
        return_per_freq=True,
    )
    sigma = safe_inverse(gamma)
    Pxx_tot = np.sum(Pxx)
    nu = np.array([
        np.trace(sigma @ (dens[:, :, i] * Pxx_tot / Pxx[i]))
        for i in range(len(freq))
    ])
    return nu, gamma
