"""Presentation glue for the reduced-QUAD demo (example 11).

The Pintelon–Schoukens **joint-MIMO modal fit** run on a *real reduced aLIGO QUAD* — the
59-state modal-truncation model committed under ``system_ident/models/`` — instead of a lumped
teaching plant or the heavy compiled twin. We drive the four **yaw** stages (M0/L1/L2/L3 .Y)
of the reduced plant through the numpy-only ``ReducedPlantBackend``, recover the open-loop 4×4
FRF, and fit the shared modal poles + rank-1 residues; the plant's own eigen-modes are the
oracle. Everything here is numpy/scipy — it runs in CI and Binder, no twin required.

NOT package API — the docs sibling of ``rank1_modal_demo`` / ``sysid_plots``.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go

_DOCS = Path(__file__).resolve().parent
for _p in (_DOCS, _DOCS.parent / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sysid_plots as sp  # noqa: E402
from system_ident.reduced_plant import ReducedStateSpacePlant  # noqa: E402
from system_ident.backends.reduced import ReducedPlantBackend  # noqa: E402
from system_ident.mimo_campaign import assemble_campaign  # noqa: E402
from system_ident.mimo_loop import recover_open_loop  # noqa: E402
from system_ident.mimo_modal import Rank1ModalModel  # noqa: E402
from system_ident.mimo_fit import (  # noqa: E402
    find_modes, init_residues, MIMOModalEstimator, parameter_covariance, modal_uncertainty,
    frf_band)

STAGES = ("M0", "L1", "L2", "L3")
DOF = "Y"                    # the yaw chain: 4 well-separated rigid-body modes, cleanly decoupled
N = 4
FS, NPERSEG, NPER = 32.0, 4096, 14   # dof = NPER-1 = 13 >= n_sens+8; the backend synthesizes
BAND = (0.4, 3.5)                     # in the frequency domain, so a long record is still instant
SENSOR_ASD = 1e-9
N_LINES = 100


def _yaw_subplant():
    p = ReducedStateSpacePlant.load("quad")
    acts = [f"{s}.drive.{DOF}" for s in STAGES]
    sens = [f"{s}.disp.{DOF}" for s in STAGES]
    return p, p.subplant(sensors=sens, actuators=acts), acts, sens


@lru_cache(maxsize=1)
def run():
    """Drive the reduced-QUAD yaw chain once and return everything the figures need.

    Cached so all panels reuse a single deterministic run. Fast (~7 s) and pure numpy —
    the page is NOT frozen; it re-runs in CI and Binder.
    """
    p, sub, acts, sens = _yaw_subplant()
    be = ReducedPlantBackend(
        sub, {f"E{j}": acts[j] for j in range(N)}, {f"S{i}": sens[i] for i in range(N)},
        fs=FS, sensor_asd=SENSOR_ASD, seed=7)
    f = np.fft.rfftfreq(NPERSEG, 1 / FS)
    band = np.flatnonzero((f >= BAND[0]) & (f <= BAND[1]))
    lines = band[:: max(1, len(band) // N_LINES)]        # ~N_LINES sparse excited lines
    psd = np.zeros(len(f))
    psd[lines] = 1.0
    exps, freq = assemble_campaign(
        be, [f"E{j}" for j in range(N)], [f"E{j}" for j in range(N)],
        [f"S{i}" for i in range(N)], f[lines], fs=FS, nperseg=NPERSEG, n_periods=NPER,
        drive_psd=psd, n_transient=1, seed=7)

    Xmat = np.stack([exps[l][1] for l in range(N)], -1)
    Ymat = np.stack([exps[l][0] for l in range(N)], -1)
    Gnp = recover_open_loop(Xmat, Ymat)                  # nonparametric open-loop FRF
    Gtruth = sub.eval(freq)                              # the reduced plant's own FRF (analytic)

    m = Rank1ModalModel(N, N, 0)
    pp = find_modes(Gnp, freq)                           # data-driven init (no oracle)
    m = Rank1ModalModel(N, N, len(pp)).set_reference(freq)
    ab = m.ab_from_modes(pp)
    phi, psi = init_residues(m, ab, exps, freq)
    res = MIMOModalEstimator(m).fit(exps, freq, m.pack(ab, phi, psi))
    Gfit = m.eval(res.theta, freq)
    Ct = parameter_covariance(res, dof=NPER - 1, n_sens=N)
    band_frf = frf_band(m, res.theta, Ct, freq)
    mu = modal_uncertainty(m, res.theta, Ct)

    # oracle scoring: label each recovered mode with its nearest EXACT plant mode (the
    # reduced model's own eigen-(f0,Q)); dedupe if two recovered modes hit the same truth
    # (find_modes can over-split), keeping the closest — so `matched` is the clean set of
    # yaw modes the fit resolved, each against its ground truth.
    allm = p.modes()
    best: dict[float, tuple] = {}
    for u in mu:
        tf, tq = min(allm, key=lambda t: abs(t[0] - u["f0"]))
        if tf not in best or abs(u["f0"] - tf) < abs(best[tf][2]["f0"] - tf):
            best[tf] = (tf, tq, u)
    matched = sorted(best.values(), key=lambda x: x[0])
    truth = [(tf, tq) for tf, tq, _ in matched]

    power = (np.abs(Gnp) ** 2).sum(axis=(1, 2))
    return SimpleNamespace(freq=freq, Gnp=Gnp, Gfit=Gfit, band=band_frf, Gtruth=Gtruth,
                           mu=mu, truth=truth, matched=matched, power=power,
                           pick_f=np.array([q[0] for q in pp]), n_lines=len(lines))


def headline():
    d = run()
    df = np.median([abs(u["f0"] - tf) / tf * 100 for tf, tq, u in d.matched])
    dq = np.median([abs(u["Q"] - tq) / tq * 100 for tf, tq, u in d.matched if np.isfinite(tq)])
    return SimpleNamespace(n_modes=len(d.matched), df_med=df, q_med=dq, n_lines=d.n_lines,
                           record_s=NPERSEG * NPER / FS)


# ── figures (house style; data-driven ranges) ─────────────────────────────────
def through_resonance_fig(elements=((0, 0), (0, 3)), *, height=560):
    """One diagonal and one off-diagonal yaw element: nonparametric |Gnp| (noisy near the
    peaks), the rank-1 |Gfit| ± 1σ band, and the reduced plant's analytic |Gtruth|."""
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
        fig.add_scatter(x=f, y=gnp, mode="markers", name=f"{tag} · nonparametric",
                        marker=dict(color=sp._fade(cfit, 0.5), size=sp.MK_SMALL))
        fig.add_scatter(x=f, y=gfit, mode="lines", name=f"{tag} · rank-1 fit",
                        line=dict(color=cfit, width=2.6))
        fig.add_scatter(x=f, y=gtru, mode="lines", name=f"{tag} · reduced-plant truth",
                        line=dict(color=ctru, width=2.0, dash="dash"))
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=sp._logy_range(mags, decades=4), title_text="|G(f)|")
    fig.update_layout(title="Through-resonance recovery — rank-1 fit vs noisy per-bin inverse")
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
    fig.update_layout(title="Yaw-mode frequency recovery — fit − truth with the Cramér–Rao bound")
    return sp.style(fig, height=height)


def modal_table():
    """Mode | truth f0/Q (reduced model) | fitted f0 ± σ | fitted Q ± σ."""
    d = run()
    rows = []
    for k, (tf, tq, u) in enumerate(d.matched):
        rows.append([f"yaw {k + 1}", f"{tf:.4f}", f"{tq:.0f}",
                     f"{u['f0']:.4f} ± {u['f0_std']:.1e}",
                     f"{u['Q']:.1f} ± {u['Q_std']:.1e}"])
    return sp.param_table(
        ["mode", "truth f₀ [Hz]", "truth Q", "fitted f₀ ± σ [Hz]", "fitted Q ± σ"],
        rows, caption="Reduced-QUAD yaw modal recovery vs the plant's exact modes")
