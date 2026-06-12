"""Recursive Bayesian / MAP estimator.

Implements a MAP / Gauss-Newton update that refines both the model mean and the
posterior precision matrix Λ.  The parameterisation mirrors ``fisher.py``
exactly: the leading denominator coefficient is held at 1 (gauge), so the
reduced parameter vector is ``θ = [num[0..n_num-1], den[1..n_den-1]]``.

Implementation note on inner iterations
----------------------------------------
The MAP objective ``½ Σ_f wt|r_f|² + ½ θᵀΛθ`` is nonlinear in the TF
coefficients (H is a rational function).  A single linearisation step (pure
EKF / extended Kalman) diverges when the starting model is far from truth
because the Jacobian is evaluated at the wrong resonance location, causing
the relative-error weights (wt = 1/H_err²) to be dominated by the prior
model's resonance rather than the true resonance.  To ensure the MAP estimate
actually converges to the posterior mode for each measurement batch, the
update re-linearises iteratively until the parameter shift is negligible
(``max |Δθ/θ| < tol``).  This is the iterated-GN / Levenberg-free MAP solver;
the plan's "one GN step per pass" refers to one *measurement batch* per pass
in the loop (Task 2), not to the number of inner linearisations.  Lambda_new
is formed once at the converged linearisation point (not accumulated over inner
steps), so it is the correct Gauss-Newton approximation to the posterior
precision at the MAP estimate.

Functions
---------
reduced_params(model)         -> (theta, n_num, n_den)
model_from_reduced(...)       -> TFModel
prior_precision(...)          -> np.ndarray
bayesian_update(...)          -> (model_new, Lambda_new)
frac_uncertainty(...)         -> float
"""

from __future__ import annotations

import numpy as np

from ..model import TFModel


# ---------------------------------------------------------------------------
# Gauge helpers
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
# Prior precision
# ---------------------------------------------------------------------------

def prior_precision(
    model0: TFModel,
    prior_uncertainty: float,
    floor: float = 1e-3,
) -> np.ndarray:
    """Diagonal prior precision matrix Λ₀ = diag(1/σ₀²).

    ``σ₀_i = prior_uncertainty · max(|θ₀_i|, floor · max|θ₀|)``

    The floor prevents a zero-valued prior coefficient from being frozen (it
    would otherwise have infinite precision and never move).
    """
    theta0, _, _ = reduced_params(model0)
    max_abs = float(np.max(np.abs(theta0)))
    sigma0 = prior_uncertainty * np.maximum(np.abs(theta0), floor * max_abs)
    return np.diag(1.0 / sigma0 ** 2)


# ---------------------------------------------------------------------------
# MAP / Gauss-Newton update
# ---------------------------------------------------------------------------

def bayesian_update(
    freq: np.ndarray,
    model: TFModel,
    H_meas: np.ndarray,
    H_err: np.ndarray,
    Lambda: np.ndarray,
    dpar: float | np.ndarray = 1e-8,
    max_inner: int = 50,
    tol: float = 1e-8,
) -> tuple[TFModel, np.ndarray]:
    """MAP Gauss-Newton update, iterated to convergence.

    Parameters
    ----------
    freq      : (n_bin,) frequency grid [Hz]
    model     : current mean model (linearisation point)
    H_meas    : (n_bin,) complex measured frequency response
    H_err     : (n_bin,) per-bin noise std-dev (zero/non-finite → zero weight)
    Lambda    : (n_par, n_par) current posterior precision matrix
    dpar      : finite-difference step for the Jacobian (matches ``fisher.py``)
    max_inner : maximum number of GN re-linearisation steps per call
    tol       : convergence criterion ``max |Δθ_i / θ_i| < tol``

    Returns
    -------
    model_new  : updated ``TFModel`` (MAP estimate given Lambda and H_meas)
    Lambda_new : posterior precision at the converged MAP point
                 (= Lambda + 𝓘 evaluated at the converged θ)
    """
    freq = np.asarray(freq, dtype=float)
    H_meas = np.asarray(H_meas, dtype=complex)
    H_err = np.asarray(H_err, dtype=float)

    # ---- per-bin weights (computed once, independent of linearisation) -----
    valid = np.isfinite(H_err) & (H_err > 0)
    wt = np.where(valid, 1.0 / H_err ** 2, 0.0)   # (n_bin,)

    # ---- extract shape from model ------------------------------------------
    n_num = model.n_num
    n_den = len(model.den)

    # ---- initialise linearisation point from current model -----------------
    par0 = model.params.astype(float)
    par0 = par0 / par0[n_num]               # gauge: den[0] = 1
    n_par_full = len(par0)
    keep = [k for k in range(n_par_full) if k != n_num]
    n_par_red = len(keep)

    theta = par0[keep].copy()               # start from prior mean

    # ---- iterated GN: re-linearise at each inner step ----------------------
    I_mat = np.zeros((n_par_red, n_par_red))
    b_vec = np.zeros(n_par_red)

    for _ in range(max_inner):
        # Reconstruct gauged model at current theta
        gauged = model_from_reduced(theta, n_num, n_den)
        # Compute Jacobian (mirrors fisher.fisher_matrix exactly)
        par_cur = gauged.params.astype(float)          # den[0] already 1
        logflag = np.zeros(n_par_full, dtype=bool)
        dH = gauged.jacobian(freq, dpar=dpar, logflag=logflag)
        dH[n_num, :] = 0.0                              # fixed row
        J = dH[keep, :]                                 # reduced Jacobian

        # Residual at current linearisation point
        r = H_meas - gauged.eval(freq)

        # Information matrix and gradient
        I_mat[:] = 0.0
        b_vec[:] = 0.0
        for i in range(n_par_red):
            wJi = wt * np.conj(J[i])
            for j in range(i, n_par_red):
                val = float(np.sum(np.real(wJi * J[j])))
                I_mat[i, j] = val
                I_mat[j, i] = val
            b_vec[i] = float(np.sum(np.real(wJi * r)))

        # GN step: solve (Lambda + I) Δθ = b
        Lambda_inner = Lambda + I_mat
        dtheta = np.linalg.solve(Lambda_inner, b_vec)
        theta = theta + dtheta

        # Convergence check: max relative step
        rel_step = np.max(np.abs(dtheta) / (np.abs(theta) + 1e-30))
        if rel_step < tol:
            break

    # ---- posterior precision at the converged MAP point -------------------
    Lambda_new = Lambda + I_mat      # I_mat is from the last (converged) linearisation

    model_new = model_from_reduced(theta, n_num, n_den)
    return model_new, Lambda_new


# ---------------------------------------------------------------------------
# Fractional uncertainty
# ---------------------------------------------------------------------------

def frac_uncertainty(model: TFModel, Lambda: np.ndarray) -> float:
    """Maximum gauge-relative posterior standard deviation.

    ``max_i( sqrt(Σ_ii) / max(|θ_i|, 1) )``  where ``Σ = Λ⁻¹``.

    Zero-valued parameters are protected by dividing by 1 instead of 0.
    """
    Sigma = np.linalg.inv(Lambda)
    theta, _, _ = reduced_params(model)
    sigma_diag = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
    denom = np.where(np.abs(theta) > 0, np.abs(theta), 1.0)
    return float(np.max(sigma_diag / denom))
