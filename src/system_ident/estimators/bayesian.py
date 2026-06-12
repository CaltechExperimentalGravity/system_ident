"""Recursive Bayesian / MAP estimator.

Implements a MAP / Gauss-Newton update that refines both the model mean and the
posterior precision matrix Λ.  The parameterisation mirrors ``fisher.py``
exactly: the leading denominator coefficient is held at 1 (gauge), so the
reduced parameter vector is ``θ = [num[0..n_num-1], den[1..n_den-1]]``.

Design: conservative small steps for the low-SNR regime
-------------------------------------------------------
The usual operating point is a *good prior* refined by *weak* (energy-limited,
low-SNR) measurements. Each pass therefore takes ONE small, Levenberg-Marquardt-
damped, step-capped, backtracked Gauss-Newton step (see :func:`bayesian_update`)
rather than solving the per-batch MAP aggressively — the latter jumps to truth on
the rare informative measurement and diverges on weak ones. Conservative steps
make the loop crawl gradually toward truth and essentially never diverge, while
the accumulated Fisher information shrinks the posterior covariance each pass.

Scope / known limitation: this refines a prior that is already in the right
basin (its resonance peaks overlap the true ones). It does NOT relocate a
resonance that sits far from the prior — local coefficient-space fitting has no
gradient to slide a non-overlapping peak across a gap. For a far prior, run a
broadband sweep first (``broadband_ls`` loop mode) to get into the basin, then
refine here. (A physical ``(f0, Q, gain)`` parameterisation would lift this
limitation and is the natural next step.)

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

def _map_objective(theta, n_num, n_den, freq, H_meas, wt, Lambda, theta_anchor):
    """Regularised MAP objective ``½ Σ wt|H_meas − G(θ)|² + ½ (θ−θ_a)ᵀ Λ (θ−θ_a)``."""
    resid = H_meas - model_from_reduced(theta, n_num, n_den).eval(freq)
    misfit = 0.5 * float(np.sum(wt * np.abs(resid) ** 2))
    d = theta - theta_anchor
    return misfit + 0.5 * float(d @ Lambda @ d)


def bayesian_update(
    freq: np.ndarray,
    model: TFModel,
    H_meas: np.ndarray,
    H_err: np.ndarray,
    Lambda: np.ndarray,
    dpar: float | np.ndarray = 1e-8,
    max_rel_step: float = 0.2,
    lm_init: float = 1e-3,
    lm_grow: float = 4.0,
    lm_max: float = 1e12,
    max_backtrack: int = 60,
) -> tuple[TFModel, np.ndarray]:
    """One small, damped MAP step (Levenberg-Marquardt) for the low-SNR regime.

    The usual operating point is a *good prior* refined by *weak* (energy-limited,
    low-SNR) measurements, so each pass should nudge the model conservatively
    rather than solve the per-batch MAP aggressively (which jumps on the rare good
    measurement and diverges on weak ones). This takes a single Gauss-Newton step
    that is

      * **damped** — ``μ·diag(H)`` added to the GN Hessian ``H = Λ + 𝓘`` for a
        robust direction even when the measurement is barely informative;
      * **capped** to a maximum relative parameter change ``max_rel_step`` per
        pass (genuinely small steps); and
      * **backtracked** — ``μ`` is grown until the regularised MAP objective does
        not increase. The step direction ``(H+μ·diag)⁻¹ g`` is always a descent
        direction, so a small enough step always reduces the objective; if the
        schedule is exhausted the mean is left unchanged.

    The loop therefore crawls gradually toward truth over many passes and
    essentially never diverges. The measurement information ``𝓘`` (Fisher at the
    anchor) is always added to the posterior precision ``Λ``, so the reported
    uncertainty shrinks monotonically as measurements accumulate.

    Parameters
    ----------
    freq, model, H_meas, H_err, Lambda
        ``model`` is the current mean = incoming posterior mean = linearisation
        anchor; ``Lambda`` is the incoming posterior precision.
    max_rel_step
        Cap on ``max_i |Δθ_i| / |θ_i|`` per call — the conservativeness knob.
    lm_init, lm_grow, lm_max, max_backtrack
        Levenberg-Marquardt damping schedule.

    Returns
    -------
    model_new  : ``TFModel`` after one damped step.
    Lambda_new : ``Lambda + 𝓘`` (information accrues regardless of step size).
    """
    freq = np.asarray(freq, dtype=float)
    H_meas = np.asarray(H_meas, dtype=complex)
    H_err = np.asarray(H_err, dtype=float)

    valid = np.isfinite(H_err) & (H_err > 0)
    wt = np.where(valid, 1.0 / H_err ** 2, 0.0)

    n_num = model.n_num
    n_den = len(model.den)
    par0 = model.params.astype(float)
    par0 = par0 / par0[n_num]                    # gauge: den[0] = 1
    n_par_full = len(par0)
    keep = [k for k in range(n_par_full) if k != n_num]
    n_par_red = len(keep)
    theta_anchor = par0[keep].copy()             # incoming posterior mean (anchor)

    # Gauss-Newton information 𝓘 and gradient g at the anchor (mirrors fisher.py).
    gauged = model_from_reduced(theta_anchor, n_num, n_den)
    logflag = np.zeros(n_par_full, dtype=bool)
    dH = gauged.jacobian(freq, dpar=dpar, logflag=logflag)
    dH[n_num, :] = 0.0
    J = dH[keep, :]
    r = H_meas - gauged.eval(freq)
    I_mat = np.zeros((n_par_red, n_par_red))
    g = np.zeros(n_par_red)
    for i in range(n_par_red):
        wJi = wt * np.conj(J[i])
        for k in range(i, n_par_red):
            val = float(np.sum(np.real(wJi * J[k])))
            I_mat[i, k] = val
            I_mat[k, i] = val
        g[i] = float(np.sum(np.real(wJi * r)))

    # Levenberg-Marquardt damped + capped + backtracked step.
    H = Lambda + I_mat
    diagH = np.clip(np.diag(H), 1e-30, None)
    # floored scale so a single near-zero coefficient cannot dominate the cap
    scale = np.maximum(np.abs(theta_anchor), 1e-3 * np.max(np.abs(theta_anchor)))
    obj0 = _map_objective(theta_anchor, n_num, n_den, freq, H_meas, wt, Lambda, theta_anchor)

    theta_new = theta_anchor                     # default: don't move if nothing helps
    mu = lm_init
    for _ in range(max_backtrack):
        step = np.linalg.solve(H + mu * np.diag(diagH), g)
        rel = float(np.max(np.abs(step) / scale))
        if rel > max_rel_step:
            step = step * (max_rel_step / rel)   # cap (preserves direction)
        cand = theta_anchor + step
        if _map_objective(cand, n_num, n_den, freq, H_meas, wt, Lambda, theta_anchor) <= obj0:
            theta_new = cand
            break
        mu *= lm_grow
        if mu > lm_max:
            break

    Lambda_new = Lambda + I_mat
    return model_from_reduced(theta_new, n_num, n_den), Lambda_new


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
