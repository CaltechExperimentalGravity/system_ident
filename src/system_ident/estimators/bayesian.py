"""Recursive Bayesian / MAP estimator — model-agnostic (gauge-free).

Operates on any model that exposes the four-method protocol::

    .params        -> theta  (n_par,)  float array
    .jacobian(freq)-> J      (n_par, n_bin)  complex array
    .eval(freq)    -> H      (n_bin,)  complex array
    .with_params(theta) -> model  same type, new parameter values

No gauge is assumed; all parameters are taken as identifiable.  For
``ResonatorModel`` (f0, Q, gain) this is exactly true.  The legacy gauge
helpers :func:`reduced_params` / :func:`model_from_reduced` are kept for
any caller that still needs coefficient-space ``TFModel`` operations, but
the three main entry-points (``prior_precision``, ``bayesian_update``,
``frac_uncertainty``) no longer call them.

Design: conservative small steps for the low-SNR regime
-------------------------------------------------------
The usual operating point is a *good prior* refined by *weak* (energy-limited,
low-SNR) measurements.  Each pass therefore takes ONE small, Levenberg-
Marquardt-damped, step-capped, backtracked Gauss-Newton step rather than
solving the per-batch MAP aggressively — the latter jumps to truth on the rare
informative measurement and diverges on weak ones.  Conservative steps make the
loop crawl gradually toward truth and essentially never diverge, while the
accumulated Fisher information shrinks the posterior covariance each pass.

Functions
---------
reduced_params(model)      -> (theta, n_num, n_den)   legacy gauge helper
model_from_reduced(...)    -> TFModel                  legacy gauge helper
prior_precision(...)       -> np.ndarray
bayesian_update(...)       -> (model_new, Lambda_new)
frac_uncertainty(...)      -> float
"""

from __future__ import annotations

import numpy as np

from ..model import TFModel


# ---------------------------------------------------------------------------
# Legacy gauge helpers  (kept for backward-compatibility; not used internally)
# ---------------------------------------------------------------------------

def reduced_params(model: TFModel) -> tuple[np.ndarray, int, int]:
    """Return the gauged reduced parameter vector plus shape integers.

    Mirrors ``fisher.fisher_matrix``: divide the full ``[num, den]`` vector by
    ``den[0]`` so that ``den[0] = 1``, then drop that fixed index ``n_num``.

    Returns
    -------
    theta : (n_par,) array  — ``[num[0..n_num-1], den[1..n_den-1]]``
    n_num : int             — length of the numerator polynomial
    n_den : int             — length of the denominator polynomial
    """
    n_num = model.n_num
    n_den = len(model.den)
    par = model.params.astype(float)
    par = par / par[n_num]          # gauge: den[0] → 1
    n_par_full = len(par)
    keep = [k for k in range(n_par_full) if k != n_num]
    theta = par[keep]
    return theta, n_num, n_den


def model_from_reduced(
    theta: np.ndarray,
    n_num: int,
    n_den: int,
) -> TFModel:
    """Reconstruct a ``TFModel`` from a reduced (gauged) parameter vector.

    Parameters
    ----------
    theta : (n_par,) array — ``[num[0..n_num-1], den[1..n_den-1]]``
    n_num : int
    n_den : int
    """
    num = theta[:n_num]
    den = np.concatenate([[1.0], theta[n_num:]])
    return TFModel(num=num, den=den)


# ---------------------------------------------------------------------------
# Prior precision  (model-agnostic)
# ---------------------------------------------------------------------------

def prior_precision(
    model0,
    prior_uncertainty: float,
    floor: float = 1e-3,
) -> np.ndarray:
    """Diagonal prior precision matrix Λ₀ = diag(1/σ₀²).

    Works on any model that exposes ``.params``.

    ``σ₀_i = prior_uncertainty · max(|θ₀_i|, floor · max|θ₀|)``

    The floor prevents a near-zero parameter from being frozen (it would
    otherwise have infinite precision and never move).
    """
    theta0 = np.asarray(model0.params, dtype=float)
    max_abs = float(np.max(np.abs(theta0)))
    sigma0 = prior_uncertainty * np.maximum(np.abs(theta0), floor * max_abs)
    return np.diag(1.0 / sigma0 ** 2)


# ---------------------------------------------------------------------------
# Gauss-Newton normal equations  (shared by the MAP step and the ML fit)
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
# MAP / Gauss-Newton update  (model-agnostic)
# ---------------------------------------------------------------------------

def _map_objective(theta, model_anchor, freq, H_meas, wt, Lambda, theta_anchor):
    """Regularised MAP objective.

    ``½ Σ_j wt_j |H_meas_j − H(θ)[j]|² + ½ (θ−θ_a)ᵀ Λ (θ−θ_a)``
    """
    resid = H_meas - model_anchor.with_params(theta).eval(freq)
    misfit = 0.5 * float(np.sum(wt * np.abs(resid) ** 2))
    d = theta - theta_anchor
    return misfit + 0.5 * float(d @ Lambda @ d)


def bayesian_update(
    freq: np.ndarray,
    model,
    H_meas: np.ndarray,
    H_err: np.ndarray,
    Lambda: np.ndarray,
    max_rel_step: float = 0.2,
    lm_init: float = 1e-3,
    lm_grow: float = 4.0,
    lm_max: float = 1e12,
    max_backtrack: int = 60,
) -> tuple:
    """One small, damped MAP step (Levenberg-Marquardt) for the low-SNR regime.

    Model-agnostic: works on any model implementing the four-method protocol
    (``.params``, ``.jacobian(freq)``, ``.eval(freq)``, ``.with_params(theta)``).
    The step is

      * **damped** — ``μ·diag(H)`` added to the GN Hessian for a robust
        direction even when the measurement is barely informative;
      * **capped** to a maximum relative parameter change ``max_rel_step`` per
        pass; and
      * **backtracked** — ``μ`` is grown until the regularised MAP objective
        does not increase.

    ``Lambda`` accumulates regardless of whether a step is taken, so the
    posterior covariance shrinks monotonically as measurements accumulate.

    Parameters
    ----------
    freq, model, H_meas, H_err, Lambda
        ``model`` is the current posterior mean (= linearisation anchor);
        ``Lambda`` is the incoming posterior precision.
    max_rel_step
        Cap on ``max_i |Δθ_i| / |θ_i|`` per call.
    lm_init, lm_grow, lm_max, max_backtrack
        Levenberg-Marquardt damping schedule.

    Returns
    -------
    model_new  : model after one damped step (same type as input).
    Lambda_new : ``Lambda + I`` (information accrues regardless of step).
    """
    freq = np.asarray(freq, dtype=float)
    H_meas = np.asarray(H_meas, dtype=complex)
    H_err = np.asarray(H_err, dtype=float)

    # Residual weighting: SNR (coherence) with a CONSTANT amplitude reference.
    #
    # The ML inverse-estimate-variance weight is 1/H_err**2 = 1/(|H|**2 rel_err**2)
    # (H_err = |H|*rel_err is the standard error of the cross-spectral estimate).
    # For a resonance the 1/|H|**2 factor makes the fit dominated by the small-|H|
    # off-resonance shoulders — the high-amplitude *peak* that actually pins f0/Q
    # gets ~1e4-1e5x LESS weight — so a concentrated drive drives Q upward without
    # bound (structurally, not from noise). Replacing the per-bin |H| with a single
    # amplitude reference H_ref keeps the weight in 1/H**2 units (consistent with
    # the prior precision Lambda) while weighting purely by SNR (1/rel_err**2), so
    # the resonance peak carries its due weight and the fit is unbiased.
    mag = np.abs(H_meas)
    valid = np.isfinite(H_err) & (H_err > 0) & (mag > 0)
    rel_err = np.where(valid, H_err / np.where(mag > 0, mag, 1.0), np.inf)
    H_ref = float(np.max(mag[valid])) if np.any(valid) else 1.0
    wt = np.where(valid, 1.0 / (rel_err * H_ref) ** 2, 0.0)

    theta_anchor = np.asarray(model.params, dtype=float).copy()

    # Jacobian and residual at the anchor.
    J = model.jacobian(freq)          # (n_par, n_bin), complex
    r = H_meas - model.eval(freq)

    # Gauss-Newton information 𝓘 and gradient g.
    I_mat, g = gn_normal_equations(J, r, wt)

    # Levenberg-Marquardt: damped + capped + backtracked step.
    H_mat = Lambda + I_mat
    diagH = np.clip(np.diag(H_mat), 1e-30, None)
    # floored scale so a near-zero parameter cannot dominate the relative cap
    scale = np.maximum(np.abs(theta_anchor), 1e-3 * np.max(np.abs(theta_anchor)))
    obj0 = _map_objective(theta_anchor, model, freq, H_meas, wt, Lambda, theta_anchor)

    theta_new = theta_anchor          # default: no movement if nothing helps
    mu = lm_init
    for _ in range(max_backtrack):
        step = np.linalg.solve(H_mat + mu * np.diag(diagH), g)
        rel = float(np.max(np.abs(step) / scale))
        if rel > max_rel_step:
            step = step * (max_rel_step / rel)   # cap (preserves direction)
        cand = theta_anchor + step
        if _map_objective(cand, model, freq, H_meas, wt, Lambda, theta_anchor) <= obj0:
            theta_new = cand
            break
        mu *= lm_grow
        if mu > lm_max:
            break

    Lambda_new = Lambda + I_mat
    return model.with_params(theta_new), Lambda_new


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
    convergence — the same per-bin objective ``bayesian_update`` takes one capped
    step on, here iterated with no prior (``Lambda = 0``).  This is the Pintelon-
    Schoukens parametric estimator: asymptotically unbiased and efficient (it
    attains the Cramer-Rao bound) when the per-bin noise ``H_err`` is correct.

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


# ---------------------------------------------------------------------------
# Fractional uncertainty  (model-agnostic)
# ---------------------------------------------------------------------------

def frac_uncertainty(model, Lambda: np.ndarray) -> float:
    """Maximum fractional posterior standard deviation.

    ``max_i( sqrt(Σ_ii) / max(|θ_i|, 1) )``  where ``Σ = Λ⁻¹``.

    Zero-valued parameters are protected by the ``max(·, 1)`` denominator.
    Works on any model that exposes ``.params``.
    """
    Sigma = np.linalg.inv(Lambda)
    theta = np.asarray(model.params, dtype=float)
    sigma_diag = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
    denom = np.where(np.abs(theta) > 0, np.abs(theta), 1.0)
    return float(np.max(sigma_diag / denom))
