"""Default input designer: Pintelon-Schoukens dispersion-function iteration.

Port of ``sys_id_dev/sysIDlib.py::get_opt_exc_Pxx`` (general SISO). Starting from
a flat excitation under a fixed power budget, each iteration evaluates the
dispersion function :func:`ligo_sysid.fisher.dispersion` and reweights the drive
PSD toward the bins that carry the most parameter information, renormalising back
to the budget. Two or three iterations are usually enough; more makes the result
lean too heavily on the prior model.

Validated against the legacy engine on the ``double_pend_demo`` setup in
``tests/test_step3_validation.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from ..fisher import dispersion
from ..model import TFModel
from .base import InputDesigner


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
):
    """Iteratively optimise the excitation PSD over ``freq``.

    Parameters mirror ``sysIDlib.get_opt_exc_Pxx``. ``Pxx`` is the (optional)
    starting PSD — a flat spectrum by default. ``Px_tot`` is the total
    drive-power budget (``trapezoid(Pxx, freq)``), enforced after every step.

    With ``rec_progress=False`` returns the final ``Pxx``; with ``True`` returns
    ``(Pxx_rec, nu_rec, gamma_rec)`` recording every iteration — used for the
    dashboard's convergence view and for validation.
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
        Pxx *= Px_tot / trapezoid(Pxx, freq)
        Pxx_rec[cnt] = Pxx
        nu_rec[cnt] = nu
        gamma_rec[cnt] = gamma

    if rec_progress:
        return Pxx_rec, nu_rec, gamma_rec
    return Pxx


class PintelonSchoukensDesigner(InputDesigner):
    """Iterative optimal excitation via the dispersion-function fixed point."""

    def design(
        self,
        freq: np.ndarray,
        model: TFModel,
        Pyy: np.ndarray,
        Px_tot: float,
        n_iter: int = 3,
    ) -> np.ndarray:
        return optimal_excitation(freq, model, Pyy, Px_tot, n_iter=n_iter)
