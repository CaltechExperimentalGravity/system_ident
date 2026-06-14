"""Gauss-Newton / maximum-likelihood fitting primitives (model-agnostic).

Operates on any model that exposes the four-method protocol::

    .params        -> theta  (n_par,)  float array
    .jacobian(freq)-> J      (n_par, n_bin)  complex array
    .eval(freq)    -> H      (n_bin,)  complex array
    .with_params(theta) -> model  same type, new parameter values

No gauge is assumed; all parameters are taken as identifiable.  For
``ResonatorModel`` (f0, Q, gain) this is exactly true.

Functions
---------
gn_normal_equations(J, r, w)                  -> (I_mat, g)
ml_fit(freq, H_meas, H_err, model, ...)       -> (model_hat, cov)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Gauss-Newton normal equations  (shared by the ML fit)
# ---------------------------------------------------------------------------

def gn_normal_equations(J: np.ndarray, r: np.ndarray, w: np.ndarray) -> tuple:
    """Gauss-Newton information matrix and gradient for a weighted complex LS fit.

    For residual ``r_j = H_meas_j - H(theta)_j``, Jacobian ``J[i,j] = dH_j/dtheta_i``
    and per-bin real weights ``w_j``, returns

        I_mat[i,k] = Re sum_j w_j conj(J[i,j]) J[k,j]   (Gauss-Newton Hessian / Fisher)
        g[i]       = Re sum_j w_j conj(J[i,j]) r_j      (half-gradient toward the fit)

    so a Gauss-Newton step solves ``(I_mat + damping) d_theta = g``.  With
    ``w = 1/sigma^2`` the matrix is exactly the Fisher information, so its inverse
    at the optimum is the Cramer-Rao covariance.
    """
    n_par = J.shape[0]
    I_mat = np.zeros((n_par, n_par))
    g = np.zeros(n_par)
    for i in range(n_par):
        wJi = w * np.conj(J[i])
        for k in range(i, n_par):
            val = float(np.sum(np.real(wJi * J[k])))
            I_mat[i, k] = val
            I_mat[k, i] = val
        g[i] = float(np.sum(np.real(wJi * r)))
    return I_mat, g


# ---------------------------------------------------------------------------
# Maximum-likelihood fit  (model-agnostic, Gauss-Newton + Levenberg-Marquardt)
# ---------------------------------------------------------------------------

def ml_fit(
    freq: np.ndarray,
    H_meas: np.ndarray,
    H_err: np.ndarray,
    model,
    max_iter: int = 100,
    tol: float = 1e-12,
    lm_init: float = 1e-3,
    lm_grow: float = 4.0,
    lm_shrink: float = 0.5,
    lm_max: float = 1e12,
) -> tuple:
    """Gaussian maximum-likelihood fit by Gauss-Newton + Levenberg-Marquardt.

    Minimises ``C(theta) = sum_j |H_meas_j - H(theta)_j|^2 / H_err_j^2`` to
    convergence.  This is the Pintelon-Schoukens parametric estimator:
    asymptotically unbiased and efficient (it attains the Cramer-Rao bound) when
    the per-bin noise ``H_err`` is correct.

    Model-agnostic: works on any model with the four-method protocol
    (``.params``, ``.jacobian``, ``.eval``, ``.with_params``).

    Returns ``(model_hat, cov)`` where ``cov`` at the optimum is the Cramer-Rao
    covariance.  For circular-complex measurement noise the Fisher information is
    ``2 * sum w Re(conj(J_i) J_k)`` (the same factor 2 carried by
    :func:`system_ident.fisher.fisher_matrix`), so ``cov = inv(2 * I_mat)``.  Bins
    with non-finite/zero ``H_err`` get zero weight.
    """
    freq = np.asarray(freq, dtype=float)
    H_meas = np.asarray(H_meas, dtype=complex)
    H_err = np.asarray(H_err, dtype=float)
    valid = np.isfinite(H_err) & (H_err > 0) & np.isfinite(H_meas)
    w = np.where(valid, 1.0 / np.where(valid, H_err, 1.0) ** 2, 0.0)

    def cost(m) -> float:
        r = H_meas - m.eval(freq)
        return float(np.sum(w * np.abs(r) ** 2))

    c0 = cost(model)
    mu = lm_init
    for _ in range(max_iter):
        theta = np.asarray(model.params, dtype=float)
        J = model.jacobian(freq)
        r = H_meas - model.eval(freq)
        I_mat, g = gn_normal_equations(J, r, w)
        diag = np.clip(np.diag(I_mat), 1e-30, None)

        improved = False
        rel = np.inf
        for _bt in range(50):
            try:
                step = np.linalg.solve(I_mat + mu * np.diag(diag), g)
            except np.linalg.LinAlgError:
                mu *= lm_grow
                if mu > lm_max:
                    break
                continue
            cand = model.with_params(theta + step)
            c1 = cost(cand)
            if np.isfinite(c1) and c1 < c0:
                rel = (c0 - c1) / max(c0, 1e-300)
                model, c0 = cand, c1
                mu = max(mu * lm_shrink, 1e-12)
                improved = True
                break
            mu *= lm_grow
            if mu > lm_max:
                break
        if not improved or rel < tol:
            break

    # Cramer-Rao covariance from the Fisher (= GN Hessian) at the optimum.
    J = model.jacobian(freq)
    r = H_meas - model.eval(freq)
    I_fin, _ = gn_normal_equations(J, r, w)
    try:
        cov = np.linalg.inv(2.0 * I_fin)   # Fisher = 2*I_fin for circular-complex noise
    except np.linalg.LinAlgError:
        cov = np.full_like(I_fin, np.nan)
    return model, cov
