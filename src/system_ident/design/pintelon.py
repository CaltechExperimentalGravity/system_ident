"""Default input designer: Pintelon-Schoukens dispersion-function iteration.

Port of ``sys_id_dev/sysIDlib.py::get_opt_exc_Pxx`` (general SISO). Starting from
a flat excitation under a fixed power budget, each iteration evaluates the
dispersion function :func:`system_ident.fisher.dispersion` and reweights the drive
PSD toward the bins that carry the most parameter information, renormalising back
to the budget. Two or three iterations are usually enough; more makes the result
lean too heavily on the prior model.

Validated against the legacy engine on the ``double_pend_demo`` setup in
``tests/test_step3_validation.py``.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sig
from scipy.integrate import trapezoid

from ..fisher import dispersion
from ..model import TFModel
from .base import InputDesigner

# Never let a drive bin starve to ~0: the dispersion function divides by ``Pxx[i]``
# (it is analytically removable — the per-bin Fisher density ``∝ Pxx[i]`` — but goes
# 0/0 → NaN numerically at an exactly-zero bin, which then poisons the next Fisher
# matrix and crashes the design with "SVD did not converge"). Flooring every bin at a
# tiny fraction of the peak keeps the reweight well-posed and guarantees a sliver of
# excitation (hence identifiability) everywhere in band.
_EXC_FLOOR_FRAC = 1e-8


def optimal_excitation(
    freq: np.ndarray,
    model: TFModel,
    Pyy: np.ndarray,
    Px_tot: float,
    Pxx: np.ndarray | None = None,
    n_iter: int = 3,
    dpar: float | np.ndarray = 1e-8,
    logflag: np.ndarray | None = None,
    rec_progress: bool = False,
    floor_frac: float = _EXC_FLOOR_FRAC,
):
    """Iteratively optimise the excitation PSD over ``freq``.

    Parameters mirror ``sysIDlib.get_opt_exc_Pxx``. ``Pxx`` is the (optional)
    starting PSD — a flat spectrum by default. ``Px_tot`` is the total
    drive-power budget (``trapezoid(Pxx, freq)``), enforced after every step.

    With ``rec_progress=False`` returns the final ``Pxx``; with ``True`` returns
    ``(Pxx_rec, nu_rec, gamma_rec)`` recording every iteration — used for the
    dashboard's convergence view and for validation.

    ``floor_frac`` floors every bin at that fraction of the per-iteration peak. The
    default ``_EXC_FLOOR_FRAC`` (1e-8) is purely numerical (keeps the 1/Pxx dispersion
    division well-posed). A larger value (e.g. 0.05) makes a *meaningful* per-line floor
    so every multisine component carries usable power — an uncertainty-aware drive that
    can iterate, not a flat broadband/noise drive.
    """
    freq = np.asarray(freq, dtype=float)
    n_bin = len(freq)
    n_par = len(model.params)

    if Pxx is None:
        Pxx = np.ones(n_bin)
    Pxx = np.asarray(Pxx, dtype=float).copy()
    Pxx *= Px_tot / trapezoid(Pxx, freq)

    Pxx_rec = np.zeros((n_iter, n_bin))
    nu_rec = np.zeros((n_iter, n_bin))
    gamma_rec = np.zeros((n_iter, n_par - 1, n_par - 1))

    for cnt in range(n_iter):
        nu, gamma = dispersion(freq, model, Pxx, Pyy, dpar=dpar, logflag=logflag)
        Pxx = Pxx * nu
        peak = np.max(Pxx)
        if peak > 0:
            Pxx = np.maximum(Pxx, floor_frac * peak)        # no bin starves to ~0
        Pxx *= Px_tot / trapezoid(Pxx, freq)
        Pxx_rec[cnt] = Pxx
        nu_rec[cnt] = nu
        gamma_rec[cnt] = gamma

    if rec_progress:
        return Pxx_rec, nu_rec, gamma_rec
    return Pxx


def prior_robust_excitation(
    freq: np.ndarray,
    model: TFModel,
    Pyy: np.ndarray,
    Px_tot: float,
    prior_uncertainty: float,
    n_iter: int = 3,
    n_samples: int = 7,
    floor_frac: float = _EXC_FLOOR_FRAC,
    floor_energy_frac: float | None = None,
) -> np.ndarray:
    """Optimal excitation averaged over the prior's plausible resonance band.

    With only a ``±prior_uncertainty`` (fractional) handle on where the
    resonances sit, spending the whole budget at the point estimate risks
    missing them. This frequency-scales the model by ``sc`` over ``[1-u, 1+u]``
    (shifting every resonance by ``sc`` while preserving Q), averages the
    point-optimal design across those scaled models, and renormalises to the
    budget — so the drive covers everywhere a resonance could plausibly be:
    efficient (not flat broadband) yet robust to a far prior.
    ``prior_uncertainty=0`` reduces to the point-optimal drive.

    Floor (a meaningful per-line floor so every line carries iterable power, without a
    flat/noise drive):
    - ``floor_frac``: floor at that fraction of the FINAL design's peak. A *fixed*
      peak-fraction floors N·floor_frac·peak·df of energy → for large N (~2000 lines) the
      floor dominates the budget and the drive collapses to near-flat.
    - ``floor_energy_frac`` (preferred): sets the floor so its TOTAL energy is that fraction
      of ``Px_tot`` — a fixed small budget SHARE regardless of the line count, so the
      Fisher-shaped peaks keep the rest. Derives ``α = floor_energy_frac·Px_tot/(peak·B)``
      (B = band width), overriding ``floor_frac``.
    """
    u = float(prior_uncertainty)
    if u <= 0.0 and floor_energy_frac is None:
        return optimal_excitation(freq, model, Pyy, Px_tot, n_iter=n_iter,
                                  floor_frac=floor_frac)
    freq = np.asarray(freq, dtype=float)
    if u <= 0.0:
        acc = optimal_excitation(freq, model, Pyy, Px_tot, n_iter=n_iter)
    else:
        z, p, k = sig.tf2zpk(model.num, model.den)
        scales = np.linspace(max(1.0 - u, 0.05), 1.0 + u, n_samples)
        acc = np.zeros(len(freq))
        for sc in scales:
            scaled = TFModel.from_zpk(z * sc, p * sc, k)
            # inner designs keep the tiny NUMERICAL floor; the meaningful floor is applied
            # ONCE to the averaged design below (else it compounds through the average).
            acc += optimal_excitation(freq, scaled, Pyy, Px_tot, n_iter=n_iter)
        integral = trapezoid(acc, freq)
        if integral > 0:
            acc *= Px_tot / integral
    peak = float(np.max(acc))
    if peak > 0:
        ff = floor_frac
        if floor_energy_frac is not None:
            band = float(freq[-1] - freq[0]) or 1.0
            ff = min(float(floor_energy_frac) * Px_tot / (peak * band), 0.5)   # derived α
        if ff > 0:
            acc = np.maximum(acc, ff * peak)                # meaningful floor on every line
            acc *= Px_tot / trapezoid(acc, freq)            # renormalise once to budget
    return acc


class PintelonSchoukensDesigner(InputDesigner):
    """Iterative optimal excitation via the dispersion-function fixed point.

    With ``prior_uncertainty > 0`` the design is averaged over the prior's
    plausible resonance band (:func:`prior_robust_excitation`) — used for the
    loop's first pass, where the model still carries large error bars.
    """

    def design(
        self,
        freq: np.ndarray,
        model: TFModel,
        Pyy: np.ndarray,
        Px_tot: float,
        n_iter: int = 3,
        prior_uncertainty: float = 0.0,
        floor_frac: float = _EXC_FLOOR_FRAC,
    ) -> np.ndarray:
        if prior_uncertainty > 0.0:
            return prior_robust_excitation(
                freq, model, Pyy, Px_tot, prior_uncertainty, n_iter=n_iter,
                floor_frac=floor_frac,
            )
        return optimal_excitation(freq, model, Pyy, Px_tot, n_iter=n_iter,
                                  floor_frac=floor_frac)
