# docs/rank1_modal_demo.py
"""Presentation-only glue for the joint-MIMO modal example page (09).

NOT package API — the docs sibling of ``sysid_plots``. It runs the coupled
closed-loop MIMO twin once (the ~60-90 s campaign), recovers the open-loop FRF
through the live damping loops, fits the rank-1 modal model, and builds the
page's plotly panels in the shared house style. The heavy campaign is cached
(``lru_cache``) so every figure reuses the single deterministic run. Every
figure is exported to SVG (Git LFS) by the page's render.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go

_DOCS = Path(__file__).resolve().parent
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))

import sysid_plots as sp  # noqa: E402
from system_ident.mimo_plant import mimo_suspension, input_matrix, output_matrix  # noqa: E402
from system_ident.mimo_loop import CoupledLoop, velocity_damper, recover_open_loop  # noqa: E402
from system_ident.backends.mimo_twin import MIMOTwinBackend  # noqa: E402
from system_ident.mimo_campaign import assemble_campaign  # noqa: E402
from system_ident.mimo_modal import Rank1ModalModel  # noqa: E402
from system_ident.mimo_fit import (  # noqa: E402
    peak_pick_modes, init_residues, MIMOModalEstimator,
    parameter_covariance, modal_uncertainty, frf_band,
)

# Truth (f0 [Hz], Q) of the six shared coupled modes; deterministic seeds throughout.
FS, NPERSEG, NPER = 128.0, 4096, 17     # dof = NPER - n_transient(3) = 14 = n_sens + 8
MODES = [(0.45, 20), (0.6, 25), (0.8, 18), (1.0, 30), (1.5, 35), (2.2, 28)]
N = 6


@lru_cache(maxsize=1)
def run():
    """Run the closed-loop MIMO campaign once and return everything the figures need.

    Heavy (~60-90 s); cached so all panels reuse a single deterministic run under
    ``freeze: true``.
    """
    plant = mimo_suspension(MODES, n_sens=N, n_act=N, coupling=0.15, gain=100.0, seed=0)
    loop = CoupledLoop(plant, [velocity_damper(1.0, 4.0) for _ in range(N)],
                       input_matrix(N, N), output_matrix(plant, N, N, basis="euler"), fs=FS)
    be = MIMOTwinBackend(loop, {f"E{j}": j for j in range(N)},
                         {f"D{j}": j for j in range(N)}, {f"S{i}": i for i in range(N)},
                         sensor_asd=1e-3, process_asd=1e-4, seed=7)
    f = np.fft.rfftfreq(NPERSEG, 1 / FS)
    lines = np.flatnonzero((f >= 0.3) & (f <= 2.6))
    psd = np.zeros(len(f))
    psd[lines] = 1.0
    exps, freq = assemble_campaign(
        be, [f"E{j}" for j in range(N)], [f"D{j}" for j in range(N)],
        [f"S{i}" for i in range(N)], f[lines], fs=FS, nperseg=NPERSEG, n_periods=NPER,
        drive_psd=psd, n_transient=3, seed=7)

    Xmat = np.stack([exps[l][1] for l in range(N)], -1)
    Ymat = np.stack([exps[l][0] for l in range(N)], -1)
    Gnp = recover_open_loop(Xmat, Ymat)                      # nonparametric open-loop FRF

    m = Rank1ModalModel(N, N, N).set_reference(freq)
    pp = peak_pick_modes(Gnp, freq, N)                       # data-driven init (no prior)
    ab = m.ab_from_modes(pp)
    phi, psi = init_residues(m, ab, exps, freq)
    res = MIMOModalEstimator(m).fit(exps, freq, m.pack(ab, phi, psi))
    Gfit = m.eval(res.theta, freq)
    Ct = parameter_covariance(res, dof=NPER - 3, n_sens=N)
    band = frf_band(m, res.theta, Ct, freq)
    mu = modal_uncertainty(m, res.theta, Ct)
    Gtruth = np.transpose(plant(2j * np.pi * freq), (2, 0, 1))   # analytic oracle (F,6,6)

    power = (np.abs(Gnp) ** 2).sum(axis=(1, 2))              # peak-pick power spectrum
    pick_f = np.array([p[0] for p in pp])
    return SimpleNamespace(freq=freq, Gnp=Gnp, Gfit=Gfit, band=band, Gtruth=Gtruth,
                           mu=mu, power=power, pick_f=pick_f, truth=MODES)


# ── figures (house style; data-driven ranges) ─────────────────────────────────
def through_resonance_fig(elements=((0, 0), (0, 3)), *, height=560):
    """For one diagonal and one off-diagonal element: nonparametric |Gnp| (noisy
    markers near resonance), the rank-1 modal |Gfit| ± 1σ band, and analytic |Gtruth|.

    The story: the parametric fit predicts cleanly THROUGH the resonances where the
    per-bin matrix inverse is noisy.
    """
    d = run()
    f = d.freq
    fig = go.Figure()
    mags = []
    palette = [(sp.GOLD, sp.SKY), (sp.ROSE, sp.GREEN)]
    for (i, j), (cfit, ctru) in zip(elements, palette):
        gnp = np.abs(d.Gnp[:, i, j])
        gfit = np.abs(d.Gfit[:, i, j])
        gtru = np.abs(d.Gtruth[:, i, j])
        sig = d.band[:, i, j]
        mags += [gfit, gtru]
        tag = f"G[{i},{j}]"
        # ±1σ band around the fit (shaded)
        fig.add_scatter(x=np.r_[f, f[::-1]],
                        y=np.r_[gfit + sig, (np.clip(gfit - sig, 1e-30, None))[::-1]],
                        fill="toself", fillcolor=sp._fade(cfit, 0.16),
                        line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=f, y=gnp, mode="markers", name=f"{tag} · nonparametric",
                        marker=dict(color=sp._fade(cfit, 0.5), size=sp.MK_SMALL))
        fig.add_scatter(x=f, y=gfit, mode="lines", name=f"{tag} · rank-1 fit",
                        line=dict(color=cfit, width=2.6))
        fig.add_scatter(x=f, y=gtru, mode="lines", name=f"{tag} · analytic truth",
                        line=dict(color=ctru, width=2.0, dash="dash"))
    yr = sp._logy_range(mags, decades=4)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="|G(f)|")
    fig.update_layout(title="Through-resonance recovery — rank-1 fit vs noisy per-bin inverse")
    return sp.style(fig, height=height)


def modal_recovery_fig(*, height=520):
    """Residual (fit − truth) f0 with the CRB ±1σ bars, per mode."""
    d = run()
    truth = np.array([t[0] for t in d.truth])
    f0 = np.array([u["f0"] for u in d.mu])
    s = np.array([u["f0_std"] for u in d.mu])
    resid_pct = (f0 - truth) / truth * 100.0
    s_pct = s / truth * 100.0
    fig = go.Figure()
    fig.add_hline(y=0.0, line_color="rgba(100,120,160,0.6)")
    fig.add_scatter(x=truth, y=resid_pct, mode="markers", name="fit − truth",
                    marker=dict(color=sp.GOLD, size=sp.MK_BIG,
                                line=dict(color="white", width=1.5)),
                    error_y=dict(type="data", array=s_pct, visible=True,
                                 color=sp._fade(sp.GOLD, 0.5), thickness=2, width=8))
    fig.update_xaxes(title_text="mode frequency [Hz]")
    fig.update_yaxes(title_text="(fitted − truth) f₀  [%]")
    fig.update_layout(title="Modal frequency recovery — fit − truth with the Cramér–Rao bound")
    return sp.style(fig, height=height)


def modal_table():
    """Mode | truth f0 | fitted f0 ± σ | fitted Q ± σ."""
    d = run()
    rows = []
    for k, (t0, tq) in enumerate(d.truth):
        u = d.mu[k]
        rows.append([f"mode {k + 1}", f"{t0:.3f}",
                     f"{u['f0']:.4f} ± {u['f0_std']:.4f}",
                     f"{u['Q']:.1f} ± {u['Q_std']:.1f}"])
    return sp.param_table(["mode", "truth f₀ [Hz]", "fitted f₀ ± σ [Hz]", "fitted Q ± σ"],
                          rows, caption="Joint-MIMO modal recovery through the live loops")


def peak_pick_fig(*, height=480):
    """The data-driven init: summed |Gnp|² power with the picked peaks marked."""
    d = run()
    f = d.freq
    fig = go.Figure()
    fig.add_scatter(x=f, y=d.power, mode="lines", name="Σ|Gnp(f)|²  (recovered FRF power)",
                    line=dict(color=sp.SKY, width=2.6))
    # mark each picked peak at its nearest power sample
    py = np.array([d.power[int(np.argmin(np.abs(f - pf)))] for pf in d.pick_f])
    fig.add_scatter(x=d.pick_f, y=py, mode="markers", name="peak-pick (init f₀)",
                    marker=dict(color=sp.GOLD, size=sp.MK_BIG, symbol="triangle-down",
                                line=dict(color="white", width=1.2)))
    yr = sp._logy_range([d.power], decades=5)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="Σᵢⱼ |Gnp(f)|²")
    fig.update_layout(title="Data-driven initialization — modes found from the recovered FRF")
    return sp.style(fig, height=height)
