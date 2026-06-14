"""Weighted inverse-frequency transfer-function fit.

Port of ``sys_id_dev/sysIDlib.py`` ``invfreqs`` (Semlyen & Deri). Given a
measured complex response with per-point weights and a denominator order, it
solves the linear, SVD-regularised inverse-frequency least-squares problem for
``(num, den)`` coefficients. Used by
:class:`~system_ident.estimators.gml.GMLEstimator` as the Sanathanan-Koerner
linearisation step.

Validated against the legacy engine in ``tests/test_step6_estimator.py``.
"""

from __future__ import annotations

import numpy as np


def invfreqs(w: np.ndarray, H: np.ndarray, wt: np.ndarray, nb: int):
    """Weighted inverse-frequency fit of ``H(w)`` to ``B(w)/A(w)``.

    Port of ``sysIDlib.invfreqs`` (Semlyen & Deri). Minimises
    ``sum( wt**2 * |B(w) - A(w) H(w)|**2 )`` via SVD with a magnitude/frequency
    rescaling for conditioning. ``w`` is angular frequency [rad/s]; ``nb`` is the
    denominator order (number of denominator coefficients minus one). Returns
    ``(num, den)`` in ``scipy`` (highest-power-first) convention.
    """
    pp = np.real(H)
    qq = np.imag(H)
    npt = len(pp)

    bb = np.zeros(2 * npt)
    bb[0 : 2 * npt : 2] = pp
    bb[1 : 2 * npt + 1 : 2] = qq

    AA = np.zeros((2 * npt, 2 * nb))
    for i in range(npt):
        w_pow = w[i] ** np.arange(nb + 1)
        _sign = 1
        for j in range(0, nb - 1, 2):
            AA[2 * i, j] = w_pow[j] * _sign
            AA[2 * i + 1, j + 1] = w_pow[j + 1] * _sign

            AA[2 * i, j + nb] = qq[i] * w_pow[j + 1] * _sign
            AA[2 * i, j + nb + 1] = pp[i] * w_pow[j + 2] * _sign
            AA[2 * i + 1, j + nb] = -pp[i] * w_pow[j + 1] * _sign
            AA[2 * i + 1, j + nb + 1] = qq[i] * w_pow[j + 2] * _sign
            _sign *= -1

    # For odd nb the main loop leaves the last numerator column (nb-1) and last
    # denominator column (2*nb-1) unfilled, producing zero singular values and NaN.
    # Fill them here; the sign follows the same alternating pattern.
    if nb % 2 == 1:
        _sign_last = (-1) ** ((nb - 1) // 2)
        for i in range(npt):
            w_pow = w[i] ** np.arange(nb + 1)
            AA[2 * i, nb - 1] = w_pow[nb - 1] * _sign_last
            AA[2 * i, 2 * nb - 1] = qq[i] * w_pow[nb] * _sign_last
            AA[2 * i + 1, 2 * nb - 1] = -pp[i] * w_pow[nb] * _sign_last

    # weighting (changes the residual being minimised)
    ww = np.zeros(2 * npt)
    ww[0 : 2 * npt : 2] = wt
    ww[1 : 2 * npt + 1 : 2] = wt
    ww = np.diag(ww)
    AA = ww @ AA
    bb = ww @ bb

    # rescaling for conditioning (preserves the residual)
    inv_w_max = 1.0 / np.max(w)
    inv_w_max_pow = inv_w_max ** np.arange(nb)
    inv_H_max = 1.0 / np.max(np.abs(H))
    DD = np.diag(np.hstack((inv_w_max_pow, inv_w_max_pow * inv_w_max * inv_H_max)))
    AA = AA @ DD

    UU, SS, VT = np.linalg.svd(AA)
    gg = (UU.T @ bb)[: 2 * nb]
    yy = gg / SS[: 2 * nb]
    xx = VT.T @ yy
    xx = DD @ xx  # undo rescaling

    num = xx[:nb]
    den = np.hstack((1, xx[nb:]))
    return num[::-1], den[::-1]
