"""Rank-1 modal sample-ML fit + Cramer-Rao bound (P&S step-2 joint MIMO fit).

Pipeline (verified on the real coupled plant, exact pole recovery from 10%/50%/arbitrary
prior error):

  peak_pick_modes(G, freq)         data-driven init: resonance peaks of the recovered FRF
  init_residues(model, ab, exps)   linear residue LS + rank-1 SVD -> mode shapes
  MIMOModalEstimator(model).fit()  IQML / iteratively-reweighted Levenberg-Marquardt SML
  parameter_covariance / modal_uncertainty / frf_band   the CRB and its propagation

The fit minimizes the P&S sample-ML equation-error cost (12-15/16/17) on the raw input/output
spectra (never Y*X^-1), with the weighting frozen per iteration (IQML, P&S section 9.12.2)
for speed and robustness. `exps` is a list of (Ybar, Ubar, Cz) per driven actuator (the robust
method, n_exp = n_act): Ybar (F,n_sens), Ubar (F,n_act) sample-mean spectra, Cz
(F,n_sens+n_act,n_sens+n_act) the stacked sample covariance of the mean.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from .mimo_loop import recover_open_loop, off_resonance_mask


# --------------------------------------------------------------------------- init
def peak_pick_modes(G, freq, n_modes, *, default_Q=20.0):
    """Initial modes [(f0,Q),...] from the resonance peaks of an FRF tensor G.

    G is the nonparametric (open-loop) FRF, shape (F, n_sens, n_act) -- e.g. step-1's
    recovered ``Y*X^-1``. Power = sum_ij |G_ij|^2; the n_modes strongest peaks seed f0.
    """
    G = np.asarray(G)
    power = (np.abs(G) ** 2).sum(axis=(1, 2))
    pk, _ = find_peaks(power)
    if len(pk) < n_modes:                      # fall back to strongest bins
        pk = np.argsort(power)[-n_modes:]
    top = sorted(pk[np.argsort(power[pk])[-n_modes:]])
    return [(float(freq[i]), float(default_Q)) for i in sorted(top)]


def init_residues(model, ab, exps, freq):
    """Mode shapes (phi, psi) from a linear residue LS given fixed poles, then rank-1 SVD.

    With poles fixed, G_ij = sum_k R_k,ij / D_k(sn) is linear in the full residue matrices
    R_k; solve per output from Ybar = G @ Ubar, then take each R_k's leading singular triple.
    Returns phi (n_modes, n_sens), psi (n_modes, n_act).
    """
    sn = model._sn(freq)
    D = [sn * sn + b * sn + a for (a, b) in ab]
    basis = [1.0 / d for d in D]
    M, ns, na = model.n_modes, model.n_sens, model.n_act
    Rk = np.zeros((M, ns, na))
    for i in range(ns):
        rows, rhs = [], []
        for (Y, U, _Cz) in exps:
            for bn in range(len(sn)):
                row = np.zeros(M * na, complex)
                for k in range(M):
                    for j in range(na):
                        row[k * na + j] = basis[k][bn] * U[bn, j]
                rows.append(row)
                rhs.append(Y[bn, i])
        A = np.array(rows)
        r = np.array(rhs)
        sol, *_ = np.linalg.lstsq(np.vstack([A.real, A.imag]),
                                  np.concatenate([r.real, r.imag]), rcond=None)
        Rk[:, i, :] = sol.reshape(M, na)
    phi = np.zeros((M, ns))
    psi = np.zeros((M, na))
    for k in range(M):
        u, s, vt = np.linalg.svd(Rk[k])
        phi[k] = u[:, 0] * np.sqrt(s[0])
        psi[k] = vt[0] * np.sqrt(s[0])
    return phi, psi


def initial_theta(model, exps, freq, G_nonparametric, *, prior_modes=None):
    """Full data-driven starting vector: peak-pick poles + rank-1 residue init."""
    modes = peak_pick_modes(G_nonparametric, freq, model.n_modes)
    if prior_modes is not None and len(modes) < model.n_modes:    # prior fills gaps
        modes = (modes + list(prior_modes))[:model.n_modes]
    ab = model.ab_from_modes(modes)
    phi, psi = init_residues(model, ab, exps, freq)
    return model.pack(ab, phi, psi)


# --------------------------------------------------------------------------- estimator
class FitResult:
    def __init__(self, theta, jac, cost, n_iter):
        self.theta = theta
        self.jac = jac          # whitened Jacobian at the solution (for the CRB)
        self.cost = cost
        self.n_iter = n_iter


class MIMOModalEstimator:
    """IQML / iteratively-reweighted Levenberg-Marquardt sample-ML fit (P&S 9.12.2, 12.3)."""

    def __init__(self, model):
        self.model = model

    def _assemble(self, theta, exps, freq):
        """Whitened (J, e) over all experiments/bins and the SML cost (weighting frozen)."""
        m = self.model
        G = m.eval(theta, freq)
        dG = m.jacobian(theta, freq)
        Jr, er, cost = [], [], 0.0
        for (Yb, Ub, Cz) in exps:
            for k in range(len(freq)):
                Gk = G[k]
                e = Yb[k] - Gk @ Ub[k]
                P = np.concatenate([np.eye(m.n_sens), -Gk], axis=1)
                Ceps = P @ Cz[k] @ P.conj().T
                w, V = np.linalg.eigh(Ceps)
                w = np.clip(w.real, 1e-18, None)
                Wh = (V * (1.0 / np.sqrt(w))) @ V.conj().T          # Ceps^-1/2
                cost += float((e.conj() @ np.linalg.solve(Ceps, e)).real)
                de = -np.einsum('ijp,j->ip', dG[k], Ub[k])
                Jr.append(Wh @ de)
                er.append(Wh @ e)
        return np.vstack(Jr), np.concatenate(er), cost

    def fit(self, exps, freq, theta0, *, max_iter=200, tol=1e-9,
            pole_prior_hz=None, prior_weight=0.0):
        """Fit the rank-1 modal model by IQML Levenberg-Marquardt.

        Optional frequency anchoring: with ``pole_prior_hz`` (one design f0 per mode)
        and ``prior_weight > 0``, a soft penalty ``prior_weight * sum_k (f0_k/prior_k - 1)^2``
        is added so a weakly-determined mode cannot drift to a non-physical frequency
        (legitimate where strong design priors exist, e.g. LIGO suspensions). The penalty
        only regularizes the fit; ``FitResult.jac`` (the CRB) carries the DATA information
        alone, so a poorly-measured anchored mode still gets honestly-large CRB bars.
        """
        m = self.model
        theta = np.asarray(theta0, float).copy()
        use_prior = pole_prior_hz is not None and prior_weight > 0.0
        if use_prior:
            fp = np.asarray(pole_prior_hz, float)
            if len(fp) != m.n_modes:
                raise ValueError(f"pole_prior_hz needs {m.n_modes} entries, got {len(fp)}")
            wpr = np.sqrt(float(prior_weight))

        def _prior_terms(th):
            # f0_k = sqrt(a_k)*s_ref/(2pi), a_k = th[k*per]; residual_k = wpr*(f0_k/fp_k - 1)
            Jp = np.zeros((m.n_modes, m.n_theta))
            ep = np.zeros(m.n_modes)
            for k in range(m.n_modes):
                ak = max(float(th[k * m.per]), 1e-12)
                f0 = np.sqrt(ak) * m.s_ref / (2 * np.pi)
                ep[k] = wpr * (f0 / fp[k] - 1.0)
                Jp[k, k * m.per] = wpr * (m.s_ref / (2 * np.pi * fp[k])) / (2 * np.sqrt(ak))
            return Jp, ep

        def _prior_cost(th):
            if not use_prior:
                return 0.0
            _, ep = _prior_terms(th)
            return float(ep @ ep)

        J, e, cost = self._assemble(theta, exps, freq)
        cost += _prior_cost(theta)
        mu = 1e-3
        n_done = 0
        dth = np.zeros_like(theta)
        for it in range(max_iter):
            Jre = np.vstack([J.real, J.imag])
            ere = np.concatenate([e.real, e.imag])
            if use_prior:
                Jp, ep = _prior_terms(theta)
                Jre = np.vstack([Jre, Jp])
                ere = np.concatenate([ere, ep])
            sc = np.linalg.norm(Jre, axis=0)
            sc[sc == 0] = 1.0
            Js = Jre / sc
            A = Js.T @ Js
            g = Js.T @ ere
            accepted = False
            for _ in range(40):                         # backtrack on mu until cost drops
                try:
                    d = np.linalg.solve(A + mu * np.eye(A.shape[0]), -g)
                except np.linalg.LinAlgError:
                    mu *= 10.0
                    continue
                dth = d / sc
                tnew = theta + dth
                Jn, en, cn = self._assemble(tnew, exps, freq)
                cn += _prior_cost(tnew)
                if cn < cost:
                    theta, J, e, cost = tnew, Jn, en, cn
                    mu = max(mu * 0.5, 1e-12)
                    accepted = True
                    break
                mu *= 4.0
                if mu > 1e14:
                    break
            n_done = it + 1
            if not accepted or np.linalg.norm(dth) < tol:
                break
        return FitResult(theta, J, cost, n_done)


# --------------------------------------------------------------------------- CRB
def parameter_covariance(fit_result, dof, n_sens):
    """CRB parameter covariance (2 Re(J^H J))^-1 with the SML inflation lambda_2 (P&S 12-30)."""
    d = dof - n_sens
    if d < 2:
        raise ValueError(
            f"dof - n_sens = {d} < 2: the SML inflation lambda_2 is undefined/negative. "
            f"Need dof >= n_sens + 2 (spec recommends dof >= n_sens + 8 for a trustworthy CRB)."
        )
    J = fit_result.jac
    fisher = 2.0 * (J.conj().T @ J).real
    cov = np.linalg.pinv(fisher, rcond=1e-10)     # drops the per-mode gauge directions
    lam = dof * dof / ((d + 1) * (d - 1))         # SML inflation lambda_2 (P&S 12-30)
    return cov * lam


def modal_uncertainty(model, theta, Ctheta):
    """Per-mode {f0, Q, f0_std, Q_std} by propagating Ctheta to the pole roots."""
    base = model.poles(theta)
    out = []
    h = 1e-6
    for idx, (f0, Q) in enumerate(base):
        gf = np.zeros(model.n_theta)
        gq = np.zeros(model.n_theta)
        for p in range(model.n_theta):
            dt = np.zeros(model.n_theta)
            dt[p] = h
            pj = model.poles(theta + dt)
            mj = model.poles(theta - dt)
            if len(pj) == len(base) == len(mj):
                gf[p] = (pj[idx][0] - mj[idx][0]) / (2 * h)
                gq[p] = (pj[idx][1] - mj[idx][1]) / (2 * h)
        out.append({"f0": f0, "Q": Q,
                    "f0_std": float(np.sqrt(max(gf @ Ctheta @ gf, 0.0))),
                    "Q_std": float(np.sqrt(max(gq @ Ctheta @ gq, 0.0)))})
    return out


def frf_band(model, theta, Ctheta, freq):
    """Per-bin FRF standard deviation sqrt(diag((dG/dtheta) Ctheta (dG/dtheta)^H)) (P&S 11-2)."""
    dG = model.jacobian(theta, freq)
    var = np.einsum('fijp,pq,fijq->fij', dG, Ctheta, dG.conj()).real
    return np.sqrt(np.clip(var, 0.0, None))


def validate_fit(model, theta, exps, freq, dof, modes_hz=None):
    """Fit validation: off-resonance FRF agreement + cost vs. expected (P&S 12-19).

    Returns dict with frf_rel_median_offres (median |G_fit-G_inv|/|G_inv| off-res),
    cost, cost_expected, cost_ratio.
    """
    n_sens, n_act = model.n_sens, model.n_act
    G_fit = model.eval(theta, freq)
    Xmat = np.stack([exps[l][1] for l in range(n_act)], axis=-1)   # (F,n_act,n_exp)
    Ymat = np.stack([exps[l][0] for l in range(n_act)], axis=-1)   # (F,n_sens,n_exp)
    G_inv = recover_open_loop(Xmat, Ymat)                          # nonparametric overlay
    keep = (np.ones(len(freq), bool) if modes_hz is None
            else off_resonance_mask(freq, modes_hz, frac=0.08))
    rel = np.abs(G_fit - G_inv) / np.maximum(np.abs(G_inv), 1e-30)
    frf_rel_median_offres = float(np.median(rel[keep]))
    cost = MIMOModalEstimator(model)._assemble(theta, exps, freq)[2]
    # cost sums over n_act experiments AND len(freq) bins (P&S 12-19 generalized)
    cost_expected = dof / (dof - n_sens) * n_sens * len(freq) * n_act
    return {"frf_rel_median_offres": frf_rel_median_offres,
            "cost": float(cost), "cost_expected": float(cost_expected),
            "cost_ratio": float(cost / cost_expected)}
