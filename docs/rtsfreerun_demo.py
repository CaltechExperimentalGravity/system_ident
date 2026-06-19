"""Presentation-only glue for the RTSfreerun twin example page (07).

NOT package API — the docs sibling of ``sysid_plots`` / ``sysid_campaign``. It runs
the real-model rungs (the compiled ``x1hsts`` / ``x1hsts6dof`` twin) and builds the
page's plotly panels, importing the analytic oracle from the package and reusing the
house plot style. It only ever executes on a box where the models + twin archives are
present; the page is rendered with ``freeze: true`` so CI serves the frozen output and
never imports the twin.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_DOCS = Path(__file__).resolve().parent
_ROOT = _DOCS.parent
for _p in (_DOCS, _ROOT / "experiments" / "rtsfreerun"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sysid_plots as sp  # noqa: E402
from sysid_campaign import run_siso_passes  # noqa: E402
from system_ident.backends import rtsfreerun_oracle as orc  # noqa: E402
from system_ident.backends.rtsfreerun_adapter import RTSfreerunBackend  # noqa: E402

CONFIG = _ROOT / "src" / "system_ident" / "configs" / "rtsfreerun_hsts.yml"


# ── A2 — open-loop SISO recovery of the compiled x1hsts plant under twin noise ──
def a2_recovery(*, seed: int = 1, n_passes: int = 3) -> SimpleNamespace:
    import yaml
    import x1hsts

    raw = yaml.safe_load(open(CONFIG))
    scen = orc.load_scenario(raw["rtsfreerun"]["scenario"])
    oracle = orc.analytic_plant(scen)
    fs = float(raw["measurement"]["fs"])
    nper = int(round(raw["measurement"]["segment_duration"] * fs))
    nseg = int(raw["measurement"]["n_segments"])
    fa = np.fft.rfftfreq(nper, 1 / fs)
    band = (fa >= raw["measurement"]["freq_min"]) & (fa <= raw["measurement"]["freq_max"])
    freq = fa[band]
    exc = raw["channels"]["excitation"]["POS"]
    xmon = raw["channels"]["drive"]["POS"]
    rb = raw["channels"]["readback"]["POS"]

    mdl = x1hsts.x1hsts()
    orc.apply_scenario_init(mdl, scen)
    modules = sorted({op["fm"] for op in scen.get("init", []) if "fm" in op})
    mdl.fm_clear_history(*modules)

    be = RTSfreerunBackend(mdl=mdl, exc_channels={exc: "POS"}, readback_channels={rb: "POS"},
                           noise=raw["rtsfreerun"]["noise"], fs=fs,
                           warmup_s=raw["rtsfreerun"]["warmup_s"], seed=seed)
    prior = orc.prior_from_scenario(scen, perturb=0.08, rng=np.random.default_rng(7))
    hist = run_siso_passes(be, exc, rb, prior, x_ch=xmon, fs=fs, nperseg=nper, n_periods=nseg,
                           band=band, freq=freq, Pyy=np.ones_like(freq), px_total=1.0e6,
                           n_passes=n_passes, prior_uncertainty=0.6, seed=seed)
    ff = np.geomspace(raw["measurement"]["freq_min"], raw["measurement"]["freq_max"], 400)
    return SimpleNamespace(ff=ff, oracle=oracle, fit=hist[-1]["model"], freq=freq,
                           H_meas=hist[-1]["H_acc"], hist=hist,
                           fracs=[h["frac"] for h in hist],
                           peak=float(np.max(np.abs(hist[-1]["drive"]))))


# ── A3 + A4 — the real closed-loop 6-DOF composite ────────────────────────────
def a34(*, n_passes: int = 2) -> SimpleNamespace:
    import hsts6dof_loop as h6

    m = h6.HSTS6DOF()
    fs, nper, nseg = 256.0, 4096, 4
    fa = np.fft.rfftfreq(nper, 1 / fs)
    band = (fa >= 0.3) & (fa <= 8.0)
    freq = fa[band]
    kw = dict(fs=fs, nperseg=nper, n_periods=nseg, band=band, freq=freq,
              n_passes=n_passes, warmup_s=16.0, seed=0)
    H_open = m.measure_tensor(closed=False, **kw)
    H_closed = m.measure_tensor(closed=True, **kw)
    G = m.oracle_tensor(freq)
    return SimpleNamespace(model=m, dofs=m.dofs, freq=freq, ffreq=freq,
                           H_open=H_open, H_closed=H_closed, G=G,
                           rel_open=m.rel_err_tensor(H_open, freq),
                           rel_closed=m.rel_err_tensor(H_closed, freq))


class _SSDiag:
    """``eval``-able wrapper for one diagonal element of the analytic SS oracle, so
    the state-space truth can drive :func:`bode_overlay` like a ``TFModel``."""
    def __init__(self, model, dof):
        self.model, self.j = model, model.dofs.index(dof)

    def eval(self, ff):
        return self.model.oracle_tensor(np.asarray(ff, float))[:, self.j, self.j]


def a3_parametric(d, dof="L", *, n_passes=4):
    """The A2-style optimal-excitation parametric campaign on one DoF of the *closed*
    6-DOF loop — the depth complement to :func:`a34`'s all-DoF breadth table. Returns
    the recovered model, the SS-oracle truth, the raw FRF, and the CRB fractions.
    """
    fs, nper, nseg = 256.0, 4096, 4
    fa = np.fft.rfftfreq(nper, 1 / fs)
    band = (fa >= 0.3) & (fa <= 8.0)
    freq = fa[band]
    hist = d.model.parametric_recovery(dof, fs=fs, nperseg=nper, n_periods=nseg, band=band,
                                       freq=freq, n_passes=n_passes, warmup_s=16.0)
    return SimpleNamespace(dof=dof, ff=np.geomspace(0.3, 8, 400),
                           oracle=_SSDiag(d.model, dof), oracle_modes=d.model.oracle_prior(dof),
                           fit=hist[-1]["model"], freq=freq, H_meas=hist[-1]["H_acc"],
                           fracs=[h["frac"] for h in hist])


# ── figures (house style) ─────────────────────────────────────────────────────
def bode_overlay(ff, oracle, fit, *, freq=None, H_meas=None, title="", height=560):
    """Magnitude+phase Bode: analytic oracle vs the recovered model (+ raw FRF points)."""
    Ho, Hf = oracle.eval(ff), fit.eval(ff)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                        vertical_spacing=0.05,
                        subplot_titles=("magnitude  |G(f)|", "phase  ∠G(f)  [deg]"))
    if freq is not None and H_meas is not None:
        good = np.abs(H_meas) > 0
        fig.add_scatter(x=freq[good], y=np.abs(H_meas[good]), mode="markers", row=1, col=1,
                        name="measured FRF", marker=dict(color=sp.ROSE, size=5, opacity=0.55))
    fig.add_scatter(x=ff, y=np.abs(Ho), mode="lines", row=1, col=1, name="analytic oracle",
                    line=dict(color="black", width=2.4))
    fig.add_scatter(x=ff, y=np.abs(Hf), mode="lines", row=1, col=1, name="recovered",
                    line=dict(color=sp.GOLD, width=2, dash="dash"))
    fig.add_scatter(x=ff, y=np.unwrap(np.angle(Ho)) * 180 / np.pi, mode="lines", row=2, col=1,
                    line=dict(color="black", width=2.4), showlegend=False)
    fig.add_scatter(x=ff, y=np.unwrap(np.angle(Hf)) * 180 / np.pi, mode="lines", row=2, col=1,
                    line=dict(color=sp.GOLD, width=2, dash="dash"), showlegend=False)
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_xaxes(type="log", title_text="frequency [Hz]", row=2, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    if title:
        fig.update_layout(title=title)
    return sp.style(fig, height=height)


def tensor_grid(d, *, height=920):
    """6×6 |FRF| grid: analytic oracle (line) vs open-loop recovered (A4, points),
    with the closed-loop recovery (A3) overlaid in green on the diagonal."""
    dofs, freq, G = d.dofs, d.freq, d.G
    n = len(dofs)
    fig = make_subplots(rows=n, cols=n, shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.012, vertical_spacing=0.018,
                        column_titles=[f"drive {x}" for x in dofs],
                        row_titles=[f"sense {x}" for x in dofs])
    for i in range(n):
        for j in range(n):
            r, c = i + 1, j + 1
            first = (i == 0 and j == 0)
            fig.add_scatter(x=freq, y=np.abs(G[:, i, j]), mode="lines", row=r, col=c,
                            line=dict(color="black", width=1.4), name="oracle",
                            legendgroup="o", showlegend=first)
            fig.add_scatter(x=freq, y=np.abs(d.H_open[i, j]), mode="markers", row=r, col=c,
                            marker=dict(color=sp.ROSE, size=2.7), name="A4 open-loop",
                            legendgroup="a4", showlegend=first)
            if i == j:
                fig.add_scatter(x=freq, y=np.abs(d.H_closed[i, j]), mode="markers", row=r, col=c,
                                marker=dict(color=sp.GREEN, size=3.2, symbol="x"),
                                name="A3 closed-loop", legendgroup="a3", showlegend=first)
            fig.update_xaxes(type="log", row=r, col=c)
            fig.update_yaxes(type="log", row=r, col=c)
    return sp.style(fig, height=height)


def convergence_fig(fracs, *, target=0.05, height=360):
    fig = go.Figure()
    fig.add_scatter(x=list(range(1, len(fracs) + 1)), y=fracs, mode="lines+markers",
                    line=dict(color=sp.SKY, width=2.5), marker=dict(size=9), name="σ/θ (worst)")
    fig.add_hline(y=target, line=dict(color=sp.ROSE, dash="dot"),
                  annotation_text=f"target {target:g}")
    fig.update_xaxes(title_text="pass", dtick=1)
    fig.update_yaxes(title_text="fractional uncertainty", type="log")
    return sp.style(fig, height=height)


def modes_table(oracle, fit, caption="HSTS drive→sensor modes — analytic vs recovered"):
    om, fm = orc.plant_modes(oracle), orc.plant_modes(fit)
    rows = []
    for (f0, q0), (f1, q1) in zip(om, fm):
        rows.append([f"{f0:.3f}", f"{f1:.3f}", f"{abs(f1 - f0) / f0 * 100:.2f}%",
                     f"{q0:.0f}", f"{q1:.0f}"])
    return sp.param_table(["oracle f₀ [Hz]", "recovered f₀", "Δf₀", "oracle Q", "recovered Q"],
                          rows, caption=caption)


def diag_table(d):
    rows = [[dof, f"{d.rel_open[i, i] * 100:.3f}%", f"{d.rel_closed[i, i] * 100:.3f}%"]
            for i, dof in enumerate(d.dofs)]
    return sp.param_table(["DoF", "A4 open-loop", "A3 closed-loop"], rows,
                          caption="Median |FRF| error vs the state-space oracle, diagonal elements")
