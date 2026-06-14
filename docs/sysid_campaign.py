"""Multi-pass Pintelon–Schoukens refinement for the docs examples.

A thin, *transparent* driver that runs the same loop the library runs — design →
inject → leakage-free FRF → inverse-variance accumulate → ML refit → CRB — but
returns the full per-pass history so the examples can plot how the measurement,
the fit, and the parameter uncertainty evolve pass by pass.

It reuses the library internals verbatim (`SysIDLoop._estimate_tf_periodic`,
`SysIDLoop._accumulate`, `SysIDLoop._frac_uncertainty`, `fisher_matrix`,
`optimal_excitation` / `prior_robust_excitation`) so nothing here is a parallel
re-implementation of the method.
"""

from __future__ import annotations

import numpy as np

from system_ident.design.pintelon import optimal_excitation, prior_robust_excitation
from system_ident.estimators.gml import GMLEstimator
from system_ident.excitation import multisine_from_psd
from system_ident.fisher import fisher_matrix
from system_ident.loop import SysIDLoop
from system_ident.model import TFModel

from sysid_plots import f0_q, modes, dc_gain  # noqa: F401 (re-exported for examples)


def run_siso_passes(backend, exc_ch, rsp_ch, prior, *, fs, nperseg, n_periods,
                    band, freq, Pyy, px_total, n_passes=3, prior_uncertainty=0.5,
                    n_design_iter=6, flat_drive=False, seed=0):
    """Run `n_passes` of P&S refinement on one SISO channel pair.

    Pass 1 uses a prior-robust drive (spread over ``f0(1±u)``); later passes use
    the point-optimal drive from the current model. Set ``flat_drive=True`` for
    the SNR-limited (no-resonance) case where a flat in-band budget is used.

    Returns a list of per-pass dicts with keys:
        pass, Pxx, drive, response, H, H_err, coh, mask,
        H_acc, err_acc, model, cov, frac, peak, rms
    """
    rng = np.random.default_rng(seed)
    total = nperseg * n_periods / fs
    model = prior
    accum = {"w": np.zeros(len(freq)), "wH": np.zeros(len(freq), dtype=complex)}
    n_gauge = prior.num.size + prior.den.size - 1
    info = np.zeros((n_gauge, n_gauge))
    est = GMLEstimator()
    history = []

    for p in range(n_passes):
        if flat_drive:
            Pxx = np.full_like(freq, px_total / (freq[-1] - freq[0]))
        elif p == 0 and prior_uncertainty > 0:
            Pxx = prior_robust_excitation(
                freq, model, Pyy, px_total, prior_uncertainty, n_iter=n_design_iter)
        else:
            Pxx = optimal_excitation(freq, model, Pyy, px_total, n_iter=n_design_iter)

        drive = multisine_from_psd(Pxx, fs, nperseg, n_periods, freq, seed=rng)
        backend.inject(exc_ch, drive, fs)
        seg = backend.read([exc_ch, rsp_ch], total)
        backend.inject(exc_ch, np.zeros_like(drive), fs)  # clear for the next pass

        H, H_err, coh = SysIDLoop._estimate_tf_periodic(
            seg[exc_ch], seg[rsp_ch], fs, nperseg, band)
        H_acc, err_acc = SysIDLoop._accumulate(accum, H, H_err)
        model = est.fit(freq, H_acc, err_acc, model)

        info = info + fisher_matrix(freq, model, Pxx, Pyy, total)
        cov = np.linalg.inv(info)
        frac = SysIDLoop._frac_uncertainty(model, cov)

        history.append(dict(
            **{"pass": p + 1}, Pxx=Pxx, drive=seg[exc_ch], response=seg[rsp_ch],
            H=H, H_err=H_err, coh=coh, mask=np.isfinite(H_err),
            H_acc=H_acc, err_acc=err_acc, model=model, cov=cov, frac=frac,
            peak=float(np.max(np.abs(seg[exc_ch]))),
            rms=float(np.std(seg[rsp_ch])),
        ))
    return history


def param_sigmas(model, cov, targets=("f0", "Q", "gain")):
    """Propagate the CRB covariance to σ on physical params (f0, Q, gain, fc).

    Numerically differentiates each target metric w.r.t. the gauged parameter
    vector (same gauge as ``fisher_matrix``: leading den coeff fixed at 1, that
    column dropped), then σ² = diag(J · cov · Jᵀ).
    """
    par = model.params.astype(float)
    n_num = model.n_num
    par = par / par[n_num]
    keep = [k for k in range(len(par)) if k != n_num]

    def metric(vec, name):
        m = TFModel(num=vec[:n_num], den=vec[n_num:])
        if name == "gain":
            return dc_gain(m)
        if name == "fc":  # single real-pole corner frequency [Hz]
            poles = np.roots(np.asarray(m.den, float))
            return float(np.min(np.abs(poles)) / (2 * np.pi))
        # "f0"/"Q" = first mode; "f0_<i>"/"Q_<i>" = i-th mode (0-based, low-f first)
        kind, _, idx = name.partition("_")
        i = int(idx) if idx else 0
        ms = modes(m)
        i = min(i, len(ms) - 1)
        return ms[i][0] if kind == "f0" else ms[i][1]

    J = np.zeros((len(targets), len(keep)))
    for col, k in enumerate(keep):
        dp = 1e-6 * max(abs(par[k]), 1e-8)
        up, dn = par.copy(), par.copy()
        up[k] += dp
        dn[k] -= dp
        for row, t in enumerate(targets):
            J[row, col] = (metric(up, t) - metric(dn, t)) / (2 * dp)
    var = np.clip(np.diag(J @ cov @ J.T), 0.0, None)
    return {t: float(np.sqrt(var[i])) for i, t in enumerate(targets)}
