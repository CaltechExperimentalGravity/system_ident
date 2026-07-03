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


def _estimate_Q(power, freq, i, *, qmin=3.0, qmax=300.0):
    """Rough Q from the half-power width of the peak at bin ``i`` (local baseline removed)."""
    n = len(power)
    lo0 = max(0, i - 1); hi0 = min(n - 1, i + 1)
    # walk out until the power starts rising again -> local valley = baseline estimate
    lo = i
    while lo > 0 and power[lo - 1] < power[lo]:
        lo -= 1
    hi = i
    while hi < n - 1 and power[hi + 1] < power[hi]:
        hi += 1
    base = min(power[lo], power[hi])
    half = base + 0.5 * (power[i] - base)
    a = i
    while a > lo and power[a] > half:
        a -= 1
    b = i
    while b < hi and power[b] > half:
        b += 1
    fwhm = float(freq[b] - freq[a])
    if fwhm <= 0:
        fwhm = float(freq[hi0] - freq[lo0]) or 1.0
    return float(np.clip(freq[i] / fwhm, qmin, qmax))


def find_modes(G, freq, *, prominence_db=4.0, min_sep_frac=0.012, min_dist_bins=3,
               max_modes=24, default_Q=50.0):
    """Data-driven modes ``[(f0,Q),...]`` from an FRF tensor — NO oracle, NO fixed count.

    Robust to the two failure modes of a naive strongest-bins pick on fine-``df`` data:
    (a) piling several 'modes' onto adjacent bins of ONE sharp peak, and (b) over-sampling
    the high-frequency modes (which dominate the absolute power) while missing the weaker
    low modes. Both are fixed by working in **dB** (log power equalises the huge
    across-band dynamic range) and selecting by **prominence** — a real resonance stands
    tens of dB above its local baseline, a recovery sidelobe does not — with a minimum
    bin distance and a fractional-``min_sep_frac`` merge. Model ORDER is chosen from the
    data (the count of prominent peaks), not supplied. Sub-``min_sep_frac`` clusters (a
    spatial doublet) merge to one shared-pole seed — resolve those with
    ``fit_block_decoupled``. Q is estimated per peak from its half-power width.

    ``G``: nonparametric FRF ``(F, n_sens, n_act)`` (e.g. ``recover_open_loop`` output).
    """
    G = np.asarray(G)
    freq = np.asarray(freq, float)
    # Collect candidate peaks from EVERY diagonal channel |G_ii|^2 and UNION them: a mode
    # is prominent in its dominant DOF channel even where it is shallow in the summed power
    # (which the top-few modes dominate). dB + prominence rejects recovery sidelobes.
    cand = []                                             # (bin_index, prominence_dB)
    for i in range(min(G.shape[1], G.shape[2])):
        p = np.abs(G[:, i, i]) ** 2
        if p.max() <= 0:                                  # dead channel — no modes to find
            continue
        pdb = 10.0 * np.log10(p + p.max() * 1e-12)
        pk, props = find_peaks(pdb, prominence=prominence_db,
                               distance=max(1, int(min_dist_bins)))
        cand += list(zip(pk.tolist(), props["prominences"].tolist()))
    if not cand:
        return []
    cand.sort(key=lambda t: -t[1])                        # strongest-standing first
    accepted = []
    for idx, _ in cand:
        f0 = float(freq[idx])
        if any(abs(f0 - freq[j]) / f0 < min_sep_frac for j in accepted):
            continue                                      # merge sub-min_sep clusters
        accepted.append(int(idx))
        if len(accepted) >= max_modes:
            break
    power = (np.abs(G) ** 2).sum(axis=(1, 2)).astype(float)   # Q from summed-power half-width
    return sorted((float(freq[i]), _estimate_Q(power, freq, i, qmax=2 * default_Q))
                  for i in accepted)


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
def mimo_fisher_matrix(model, theta, exps, freq):
    """MIMO Fisher information ``2 Re(J^H J)`` for the modal parameters at ``theta``.

    ``J`` is the campaign-``Cz``-whitened Jacobian ``d(residual)/dtheta`` (P&S SML), the
    SAME object the estimator assembles. Unlike :func:`parameter_covariance` this is
    fit-INDEPENDENT — it takes ``(model, theta, exps)`` directly, so the feasibility gate
    can be evaluated on ANY model/scenario without a fit: the CRB at the oracle ``theta`` is
    the ideal bound, the CRB at a candidate model predicts its uncertainty, and it drives a
    DONE criterion for the iterative loop (:func:`modal_frac_uncertainty`).
    """
    J = MIMOModalEstimator(model)._assemble(theta, exps, freq)[0]
    return 2.0 * (J.conj().T @ J).real


def _cov_from_fisher(fisher, dof, n_sens):
    d = dof - n_sens
    if d < 2:
        raise ValueError(
            f"dof - n_sens = {d} < 2: the SML inflation lambda_2 is undefined/negative. "
            f"Need dof >= n_sens + 2 (spec recommends dof >= n_sens + 8 for a trustworthy CRB)."
        )
    cov = np.linalg.pinv(fisher, rcond=1e-10)     # drops the per-mode gauge directions
    lam = dof * dof / ((d + 1) * (d - 1))         # SML inflation lambda_2 (P&S 12-30)
    return cov * lam


def parameter_covariance(fit_result, dof, n_sens):
    """CRB parameter covariance (2 Re(J^H J))^-1 with the SML inflation lambda_2 (P&S 12-30),
    from a completed fit's whitened Jacobian ``fit_result.jac``."""
    fisher = 2.0 * (fit_result.jac.conj().T @ fit_result.jac).real
    return _cov_from_fisher(fisher, dof, n_sens)


def mimo_parameter_covariance(model, theta, exps, freq, *, dof, n_sens):
    """Fit-independent MIMO CRB covariance at ``theta`` from the campaign ``Cz`` — the same
    SML-inflated ``(2 Re J^H J)^-1`` as :func:`parameter_covariance` but computed directly
    from ``(model, theta, exps)`` (via :func:`mimo_fisher_matrix`), so you can predict the
    per-mode CRB for any model/scenario without running a fit."""
    return _cov_from_fisher(mimo_fisher_matrix(model, theta, exps, freq), dof, n_sens)


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


def modal_frac_uncertainty(model, theta, Ctheta):
    """Worst-case fractional per-mode uncertainty: ``max`` over modes of ``f0_std/f0`` and
    ``Q_std/Q``. A single scalar the feasibility gate / iterative loop compare to a target
    (the MIMO analog of ``loop.py::_frac_uncertainty``): stop when it drops below the goal.
    ``inf``-Q modes contribute only their f0 term. Returns 0.0 for an empty model.
    """
    fr = []
    for x in modal_uncertainty(model, theta, Ctheta):
        if x["f0"] > 0:
            fr.append(x["f0_std"] / abs(x["f0"]))
        if np.isfinite(x["Q"]) and x["Q"] > 0:
            fr.append(x["Q_std"] / x["Q"])
    return float(max(fr)) if fr else 0.0


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


def fit_block_decoupled(exps, freq, blocks, *, dof=None, max_iter=600, prior_weight=0.0):
    """Fit independent rank-1 modal models on decoupled DOF blocks, then combine.

    For a plant that block-diagonalizes into orthogonal DOF subspaces — e.g. a
    suspension whose dynamics split into the {L,P,V} and {T,R,Y} planes — two modes
    near-coincident in FREQUENCY but living in DIFFERENT blocks form a *spatial
    doublet*: they are separated by block membership, NOT by frequency resolution. A
    single shared-pole full-MIMO fit collapses such a pair (two near-equal poles with
    orthogonal residues make J^H J ill-conditioned); fitting each block alone — where
    only one of the pair appears — recovers both cleanly with no frequency
    super-resolution and no doublet-concentrated drive.

    Parameters
    ----------
    exps : list of full-MIMO experiments ``(Ybar(F,n_dof), Ubar(F,n_dof),
        Cz(F,2*n_dof,2*n_dof))``, one per driven DOF, indexed by DOF.
    freq : excited-line frequencies.
    blocks : list of ``{"sensors": [idx...], "actuators": [idx...], "modes":
        [(f0,Q),...]}`` — DOF indices into the full ordering plus the prior modes
        assigned to that block (from the suspension design / plane decomposition).
        Experiments ``exps[a]`` for ``a in actuators`` drive the block.
    dof : P&S effective dof (``n_periods - n_transient``) for the per-block CRB; if
        None, the CRB (``mu``) is skipped.
    max_iter : LM iterations per block fit.
    prior_weight : if > 0, softly anchor each block's poles to their seed frequencies
        (the block's ``modes``) — stabilises the fragile per-block fit against a pole
        drifting to a spurious near-critically-damped junk mode. The CRB (``FitResult.jac``)
        still carries DATA information alone, so anchored bars stay honest.

    Returns
    -------
    list of dicts, one per block: ``{"sensors", "actuators", "model", "fit",
    "modes": [(f0,Q),...], "mu": modal_uncertainty | None}``.
    """
    from .mimo_modal import Rank1ModalModel
    n_dof = exps[0][0].shape[1]
    F = len(freq)
    out = []
    for blk in blocks:
        si = list(blk["sensors"]); ai = list(blk["actuators"]); pm = sorted(blk["modes"])
        comp = si + [n_dof + a for a in ai]            # block rows/cols of the 2n Cz vector
        sub = [(exps[a][0][:, si], exps[a][1][:, ai],
                exps[a][2][np.ix_(np.arange(F), comp, comp)]) for a in ai]
        m = Rank1ModalModel(len(si), len(ai), n_modes=len(pm)).set_reference(freq)
        ab = m.ab_from_modes(pm)
        phi, psi = init_residues(m, ab, sub, freq)
        kw = ({"pole_prior_hz": [f for f, _ in pm], "prior_weight": prior_weight}
              if prior_weight > 0 else {})
        res = MIMOModalEstimator(m).fit(sub, freq, m.pack(ab, phi, psi), max_iter=max_iter, **kw)
        mu = None
        if dof is not None:
            Ct = parameter_covariance(res, dof=dof, n_sens=len(si))
            mu = modal_uncertainty(m, res.theta, Ct)
        out.append({"sensors": si, "actuators": ai, "model": m, "fit": res,
                    "modes": m.poles(res.theta), "mu": mu})
    return out
