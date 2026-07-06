"""Presentation glue for the CLOSED-LOOP reduced-QUAD demo (example 12).

The companion to example 11 (open loop): the same real 59-state reduced aLIGO QUAD, now run
**through its damping loops**. The yaw chain is closed around velocity dampers with
**python-control** (time-domain state-space simulation, the natural tool — `MIMOTwinBackend` /
`CoupledLoop`), driven by a P&S multisine per actuator, and the leakage-free **reference-based
FRF cancels the controller** to return the open-loop plant — the flagship closed-loop method
(examples 05/09/10), here on a real reduced suspension that runs in CI and Binder.

NOT package API — the docs sibling of ``reduced_quad_demo`` / ``rank1_modal_demo``.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import control

_DOCS = Path(__file__).resolve().parent
for _p in (_DOCS, _DOCS.parent / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sysid_plots as sp  # noqa: E402
from system_ident.reduced_plant import ReducedStateSpacePlant  # noqa: E402
from system_ident.backends.mimo_twin import MIMOTwinBackend  # noqa: E402
from system_ident.mimo_loop import CoupledLoop, velocity_damper, recover_open_loop  # noqa: E402
from system_ident.mimo_plant import input_matrix, output_matrix  # noqa: E402
from system_ident.mimo_campaign import assemble_campaign  # noqa: E402
from system_ident.mimo_modal import Rank1ModalModel  # noqa: E402
from system_ident.mimo_fit import (  # noqa: E402
    find_modes, init_residues, MIMOModalEstimator, parameter_covariance, modal_uncertainty,
    frf_band)

STAGES = ("M0", "L1", "L2", "L3")
DOF = "Y"
N = 4
FS, NPERSEG, NPER = 64.0, 2048, 14
BAND = (0.4, 3.5)
DAMP_K, DAMP_FC = 1.0, 3.0        # velocity-damper gain and corner (per DOF)
SENSOR_ASD = 1e-9
N_LINES = 90


@lru_cache(maxsize=1)
def run():
    """Close the reduced-QUAD yaw chain around velocity dampers and identify it through the
    live loops. Cached; ~8 s, pure python (numpy/scipy/control)."""
    p = ReducedStateSpacePlant.load("quad")
    acts = [f"{s}.drive.{DOF}" for s in STAGES]
    sens = [f"{s}.disp.{DOF}" for s in STAGES]
    sub = p.subplant(sensors=sens, actuators=acts)

    G = control.ss(sub.A, sub.B, sub.C, sub.D)        # the reduced plant as a control StateSpace
    Min = input_matrix(N, N)
    Mout = output_matrix(G, N, N, basis="euler")
    loop = CoupledLoop(G, [velocity_damper(DAMP_K, DAMP_FC) for _ in range(N)], Min, Mout, fs=FS)
    stable = loop.is_stable()
    be = MIMOTwinBackend(loop, {f"E{j}": j for j in range(N)}, {f"D{j}": j for j in range(N)},
                         {f"S{i}": i for i in range(N)}, sensor_asd=SENSOR_ASD, seed=7)

    f = np.fft.rfftfreq(NPERSEG, 1 / FS)
    band = np.flatnonzero((f >= BAND[0]) & (f <= BAND[1]))
    lines = band[:: max(1, len(band) // N_LINES)]
    psd = np.zeros(len(f)); psd[lines] = 1.0
    exps, freq = assemble_campaign(
        be, [f"E{j}" for j in range(N)], [f"D{j}" for j in range(N)],
        [f"S{i}" for i in range(N)], f[lines], fs=FS, nperseg=NPERSEG, n_periods=NPER,
        drive_psd=psd, n_transient=1, seed=7)

    Xmat = np.stack([exps[l][1] for l in range(N)], -1)
    Ymat = np.stack([exps[l][0] for l in range(N)], -1)
    Gnp = recover_open_loop(Xmat, Ymat)               # reference-based: cancels the dampers
    Gtruth = np.transpose(loop.oracle(freq), (2, 0, 1))   # analytic open-loop plant (F,N,N)

    pp = find_modes(Gnp, freq)
    m = Rank1ModalModel(N, N, len(pp)).set_reference(freq)
    ab = m.ab_from_modes(pp)
    phi, psi = init_residues(m, ab, exps, freq)
    res = MIMOModalEstimator(m).fit(exps, freq, m.pack(ab, phi, psi))
    Gfit = m.eval(res.theta, freq)
    Ct = parameter_covariance(res, dof=NPER - 1, n_sens=N)
    band_frf = frf_band(m, res.theta, Ct, freq)
    mu = modal_uncertainty(m, res.theta, Ct)

    allm = p.modes()
    best: dict = {}
    for u in mu:
        tf, tq = min(allm, key=lambda t: abs(t[0] - u["f0"]))
        if tf not in best or abs(u["f0"] - tf) < abs(best[tf][2]["f0"] - tf):
            best[tf] = (tf, tq, u)
    matched = sorted(best.values(), key=lambda x: x[0])

    return SimpleNamespace(freq=freq, Gnp=Gnp, Gfit=Gfit, band=band_frf, Gtruth=Gtruth,
                           mu=mu, matched=matched, stable=stable, n_lines=len(lines))


def headline():
    d = run()
    df = np.median([abs(u["f0"] - tf) / tf * 100 for tf, tq, u in d.matched])
    return SimpleNamespace(n_modes=len(d.matched), df_med=df, stable=d.stable,
                           damp_k=DAMP_K, damp_fc=DAMP_FC, record_s=NPERSEG * NPER / FS)


def through_resonance_fig(elements=((0, 0), (0, 3)), *, height=560):
    """One diagonal and one off-diagonal element through the closed loops: nonparametric
    |Gnp| (the reference-based recovery), the rank-1 |Gfit| ± 1σ band, and the analytic
    open-loop truth. The controller is cancelled, not fit."""
    d = run()
    f = d.freq
    fig = go.Figure()
    mags = []
    for (i, j), (cfit, ctru) in zip(elements, [(sp.GOLD, sp.SKY), (sp.ROSE, sp.GREEN)]):
        gnp, gfit, gtru = np.abs(d.Gnp[:, i, j]), np.abs(d.Gfit[:, i, j]), np.abs(d.Gtruth[:, i, j])
        sig = d.band[:, i, j]
        mags += [gfit, gtru]
        tag = f"G[{i},{j}]"
        fig.add_scatter(x=np.r_[f, f[::-1]],
                        y=np.r_[gfit + sig, np.clip(gfit - sig, 1e-30, None)[::-1]],
                        fill="toself", fillcolor=sp._fade(cfit, 0.16), line=dict(width=0),
                        hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=f, y=gnp, mode="markers", name=f"{tag} · recovered (loops cancelled)",
                        marker=dict(color=sp._fade(cfit, 0.5), size=sp.MK_SMALL))
        fig.add_scatter(x=f, y=gfit, mode="lines", name=f"{tag} · rank-1 fit",
                        line=dict(color=cfit, width=2.6))
        fig.add_scatter(x=f, y=gtru, mode="lines", name=f"{tag} · open-loop truth",
                        line=dict(color=ctru, width=2.0, dash="dash"))
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=sp._logy_range(mags, decades=4), title_text="|G(f)|")
    fig.update_layout(title="Open-loop plant recovered through the damping loops — "
                            "controller cancelled, not fit")
    return sp.style(fig, height=height)


def modal_recovery_fig(*, height=520):
    """(fit − truth) f0 in %, per matched mode, with the Cramér–Rao ±1σ bars."""
    d = run()
    tf = np.array([m[0] for m in d.matched])
    f0 = np.array([m[2]["f0"] for m in d.matched])
    s = np.array([m[2]["f0_std"] for m in d.matched])
    fig = go.Figure()
    fig.add_hline(y=0.0, line_color="rgba(100,120,160,0.6)")
    fig.add_scatter(x=tf, y=(f0 - tf) / tf * 100, mode="markers", name="fit − truth",
                    marker=dict(color=sp.GOLD, size=sp.MK_BIG, line=dict(color="white", width=1.5)),
                    error_y=dict(type="data", array=s / tf * 100, visible=True,
                                 color=sp._fade(sp.GOLD, 0.5), thickness=2, width=8))
    fig.update_xaxes(title_text="mode frequency [Hz]")
    fig.update_yaxes(title_text="(fitted − truth) f₀  [%]")
    fig.update_layout(title="Yaw-mode recovery through the closed loops — fit − truth with the CRB")
    return sp.style(fig, height=height)


def modal_table():
    d = run()
    rows = [[f"yaw {k + 1}", f"{tf:.4f}", f"{tq:.0f}",
             f"{u['f0']:.4f} ± {u['f0_std']:.1e}", f"{u['Q']:.1f} ± {u['Q_std']:.1e}"]
            for k, (tf, tq, u) in enumerate(d.matched)]
    return sp.param_table(
        ["mode", "truth f₀ [Hz]", "truth Q", "fitted f₀ ± σ [Hz]", "fitted Q ± σ"],
        rows, caption="Reduced-QUAD yaw modes recovered through the closed damping loops")
