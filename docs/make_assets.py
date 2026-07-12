"""Generate static docs assets (gallery thumbnails + Open Graph card) with kaleido.

Run from docs/:   conda run -n sysid python make_assets.py
Writes:           examples/thumbnails/0{1..7}.svg (vector, LFS) + assets/og-card.png
                  (the Open-Graph social card — raster only because social platforms
                  do not render SVG previews; still committed via LFS).

These are cheap signature figures (no twin simulation) rendered in the house
style, exported to PNG for the Examples card-gallery and social-card preview.
"""
from __future__ import annotations

import pathlib

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sysid_plots as sp
from system_ident.model import TFModel
from system_ident.plant import double_pendulum, coupled_suspension
from system_ident.design.pintelon import optimal_excitation

HERE = pathlib.Path(__file__).parent
THUMBS = HERE / "examples" / "thumbnails"
ASSETS = HERE / "assets"
THUMBS.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

NAVY = "#0B1F33"


def _thumb_axes(fig, logx=True):
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False,
                     type="log" if logx else "linear")
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, type="log")
    fig.update_layout(template="plotly_white", showlegend=False,
                      margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="white", plot_bgcolor="white")
    return fig


def _curve_thumb(freq, mag, drive=None, extra=None):
    fig = go.Figure()
    if drive is not None:
        d = drive / drive.max() * mag.max()
        fig.add_trace(go.Scatter(x=freq, y=d, mode="lines", fill="tozeroy",
                      fillcolor="rgba(200,151,58,0.18)", line=dict(color=sp.GOLD, width=3)))
    fig.add_trace(go.Scatter(x=freq, y=mag, mode="lines", line=dict(color=sp.SKY, width=4)))
    for tr in (extra or []):
        fig.add_trace(tr)
    return _thumb_axes(fig)


def thumb_01():
    f = np.linspace(0.2, 8, 600)
    m = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    Pxx = optimal_excitation(f, m, np.ones_like(f), 1.0, n_iter=6)
    return _curve_thumb(f, np.abs(m.eval(f)), np.sqrt(Pxx))


def thumb_02():
    f = np.linspace(0.1, 5, 600)
    m = double_pendulum()
    Pxx = optimal_excitation(f, m, np.ones_like(f), 1.0, n_iter=6)
    return _curve_thumb(f, np.abs(m.eval(f)), np.sqrt(Pxx))


def thumb_03():
    f = np.logspace(0, 3, 600)
    wc = 2 * np.pi * 100
    m = TFModel.from_zpk([], [-wc], wc)
    return _curve_thumb(f, np.abs(m.eval(f)))


def thumb_04():
    f = np.linspace(0.1, 5, 600)
    plants = [TFModel.from_resonances([(0.6, 20), (1.5, 30)], 300),
              TFModel.from_resonances([(0.8, 15), (2.2, 25)], 120),
              TFModel.from_resonances([(1.2, 18)], 80)]
    cols = [sp.SKY, sp.GOLD, sp.GREEN]
    fig = go.Figure()
    for m, c in zip(plants, cols):
        fig.add_trace(go.Scatter(x=f, y=np.abs(m.eval(f)), mode="lines",
                      line=dict(color=c, width=4)))
    return _thumb_axes(fig, logx=False)


def thumb_05():
    f = np.logspace(np.log10(0.1), np.log10(5), 600)
    P = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    # crude suppressed peak for the motif
    Tsup = np.abs(P.eval(f)) / (1 + 6 * np.exp(-((np.log(f) - np.log(1.0)) ** 2) / 0.02))
    extra = [go.Scatter(x=f, y=Tsup, mode="lines",
             line=dict(color=sp.GRAY, width=3, dash="dot"))]
    return _curve_thumb(f, np.abs(P.eval(f)), extra=extra)


def thumb_06():
    f = np.logspace(np.log10(0.2), np.log10(3), 600)
    H = coupled_suspension([(0.43, 20), (1.00, 30)], [(0.56, 18), (1.31, 28)],
                           coupling=0.20, gain=100.0)
    cols = {("POS", "POS"): sp.SKY, ("PIT", "PIT"): sp.GOLD,
            ("PIT", "POS"): sp.ROSE, ("POS", "PIT"): sp.GREEN}
    fig = go.Figure()
    for key, c in cols.items():
        w = 4 if key[0] == key[1] else 3
        fig.add_trace(go.Scatter(x=f, y=np.abs(H[key].eval(f)), mode="lines",
                      line=dict(color=c, width=w)))
    return _thumb_axes(fig)


def thumb_07():
    # The compiled-twin HSTS plant: 5 real modes (~0.67–3.78 Hz, Q≈50) under the
    # optimal drive, with a faint cross-coupling curve hinting the closed-loop MIMO.
    f = np.linspace(0.3, 8, 700)
    m = TFModel.from_resonances([(0.67, 50), (1.01, 50), (1.52, 50),
                                 (2.81, 50), (3.78, 50)], 100.0)
    Pxx = optimal_excitation(f, m, np.ones_like(f), 1.0, n_iter=6)
    coup = TFModel.from_resonances([(0.67, 50), (2.81, 50)], 18.0)
    extra = [go.Scatter(x=f, y=np.abs(coup.eval(f)), mode="lines",
             line=dict(color=sp.ROSE, width=2.5))]
    return _curve_thumb(f, np.abs(m.eval(f)), np.sqrt(Pxx), extra=extra)


def thumb_09():
    # Joint-MIMO modal fit: the recovered open-loop magnitude (six shared modes) with
    # its ±σ envelope, a faint cross-coupling curve, over the band-limited drive — the
    # signature of the rank-1 modal recovery through the live damping loops.
    f = np.linspace(0.3, 2.6, 700)
    m = TFModel.from_resonances([(0.45, 20), (0.6, 25), (0.8, 18),
                                 (1.0, 30), (1.5, 35), (2.2, 28)], 100.0)
    mag = np.abs(m.eval(f))
    Pxx = optimal_excitation(f, m, np.ones_like(f), 1.0, n_iter=6)
    coup = TFModel.from_resonances([(0.6, 25), (1.5, 35)], 20.0)
    extra = [go.Scatter(x=f, y=np.abs(coup.eval(f)), mode="lines",
             line=dict(color=sp.ROSE, width=2.5))]
    return _curve_thumb(f, mag, np.sqrt(Pxx), extra=extra)


def thumb_11():
    # Reduced-QUAD MIMO modal fit: the recovered yaw-chain magnitude (the 4 yaw
    # rigid-body modes of the real 59-state reduced aLIGO quad, 0.6–3.0 Hz) over the
    # band-limited drive, with a faint cross-coupling curve — the flagship rank-1 modal
    # recovery on a real reduced plant, run in the browser (numpy-only, no twin).
    f = np.linspace(0.4, 3.5, 800)
    m = TFModel.from_resonances([(0.599, 60), (1.349, 90), (2.391, 120), (3.036, 140)], 100.0)
    mag = np.abs(m.eval(f))
    Pxx = optimal_excitation(f, m, np.ones_like(f), 1.0, n_iter=6)
    coup = TFModel.from_resonances([(0.599, 60), (2.391, 120)], 20.0)
    extra = [go.Scatter(x=f, y=np.abs(coup.eval(f)), mode="lines",
             line=dict(color=sp.ROSE, width=2.5))]
    return _curve_thumb(f, mag, np.sqrt(Pxx), extra=extra)


def thumb_12():
    # Closed-loop reduced-QUAD: the open-loop yaw plant recovered THROUGH the damping loops
    # (controller cancelled by the reference-based FRF), over the band-limited drive — the
    # flagship closed-loop method on a real reduced aLIGO quad via python-control.
    f = np.linspace(0.4, 3.5, 800)
    m = TFModel.from_resonances([(0.599, 60), (1.349, 90), (2.391, 120), (3.036, 140)], 100.0)
    mag = np.abs(m.eval(f))
    Pxx = optimal_excitation(f, m, np.ones_like(f), 1.0, n_iter=6)
    coup = TFModel.from_resonances([(1.349, 90), (3.036, 140)], 25.0)
    extra = [go.Scatter(x=f, y=np.abs(coup.eval(f)), mode="lines",
             line=dict(color=sp.ROSE, width=2.5))]
    return _curve_thumb(f, mag, np.sqrt(Pxx), extra=extra)


def thumb_10():
    # SRM modal identification through the real L1-SRM loops: the recovered HSTS
    # magnitude (the 13 resolvable Q≈50 modes, 0.67–3.78 Hz) over the band-limited
    # drive, with a faint cross-coupling curve — the signature of the rank-1 joint
    # modal recovery through the real production damping loops.
    f = np.linspace(0.3, 8, 800)
    m = TFModel.from_resonances([(0.674, 50), (0.848, 50), (1.005, 50), (1.092, 50),
                                 (1.484, 50), (2.038, 50), (2.184, 50), (2.762, 50),
                                 (2.807, 50), (2.982, 50), (3.209, 50), (3.424, 50),
                                 (3.781, 50)], 100.0)
    mag = np.abs(m.eval(f))
    Pxx = optimal_excitation(f, m, np.ones_like(f), 1.0, n_iter=6)
    coup = TFModel.from_resonances([(0.674, 50), (2.038, 50)], 18.0)
    extra = [go.Scatter(x=f, y=np.abs(coup.eval(f)), mode="lines",
             line=dict(color=sp.ROSE, width=2.5))]
    return _curve_thumb(f, mag, np.sqrt(Pxx), extra=extra)


def thumb_13():
    # DARM drift tracking: a slowly-drifting κ(t) (GOLD) with its CRB band and the
    # per-record snapshot points (SKY) — the signature of the round-1 time-varying sysID.
    # Linear time axis (not the log-frequency of the other thumbs).
    rng = np.random.default_rng(13)
    t = np.linspace(0.0, 60.0, 400)
    k = 1.0 + 0.05 * np.sin(2 * np.pi * t / 150.0)       # gentle hour-scale drift
    band = 0.012
    ts = np.linspace(3.0, 57.0, 12)
    ks = 1.0 + 0.05 * np.sin(2 * np.pi * ts / 150.0) + rng.normal(0, 0.009, ts.size)
    fig = go.Figure()
    fig.add_scatter(x=t, y=k + band, mode="lines", line=dict(width=0))
    fig.add_scatter(x=t, y=k - band, mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(200,151,58,0.18)")
    fig.add_scatter(x=t, y=k, mode="lines", line=dict(color=sp.GOLD, width=4))
    fig.add_scatter(x=ts, y=ks, mode="markers", marker=dict(color=sp.SKY, size=10))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(template="plotly_white", showlegend=False,
                      margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="white", plot_bgcolor="white")
    return fig


def og_card():
    f = np.linspace(0.2, 8, 800)
    m = TFModel.from_resonances([(1.0, 20.0)], 100.0)
    Pxx = optimal_excitation(f, m, np.ones_like(f), 1.0, n_iter=6)
    mag = np.abs(m.eval(f))
    drive = np.sqrt(Pxx); drive = drive / drive.max() * mag.max()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=drive, mode="lines", fill="tozeroy",
                  fillcolor="rgba(200,151,58,0.22)", line=dict(color=sp.GOLD, width=4)))
    fig.add_trace(go.Scatter(x=f, y=mag, mode="lines", line=dict(color=sp.SKY, width=5)))
    fig.update_xaxes(visible=False, type="log")
    fig.update_yaxes(visible=False, type="log")
    fig.add_annotation(x=0.045, y=0.93, xref="paper", yref="paper", showarrow=False,
        text="<b>system_ident</b>", xanchor="left",
        font=dict(family=sp.FONT, size=72, color=NAVY))
    fig.add_annotation(x=0.045, y=0.77, xref="paper", yref="paper", showarrow=False,
        text="Optimal-excitation system identification for LIGO suspensions",
        xanchor="left", font=dict(family=sp.FONT, size=30, color="#3A4A5C"))
    fig.add_annotation(x=0.045, y=0.14, xref="paper", yref="paper", showarrow=False,
        text="Pintelon–Schoukens · leakage-free FRF · maximum likelihood · Cramér–Rao",
        xanchor="left", font=dict(family=sp.FONT, size=24, color=sp.GOLD))
    fig.update_layout(template="plotly_white", showlegend=False,
                      margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="white", plot_bgcolor="white")
    return fig


THUMBS_FNS = {"01": thumb_01, "02": thumb_02, "03": thumb_03,
              "04": thumb_04, "05": thumb_05, "06": thumb_06, "07": thumb_07,
              "09": thumb_09, "10": thumb_10, "11": thumb_11, "12": thumb_12,
              "13": thumb_13}

if __name__ == "__main__":
    # Plots are SVG (vector) and live in Git LFS — hard rule, no PNG plots.
    for name, fn in THUMBS_FNS.items():
        out = THUMBS / f"{name}.svg"
        fn().write_image(str(out), format="svg", width=640, height=420)
        print("wrote", out.relative_to(HERE))
    # og-card is the Open-Graph social preview: social platforms won't render SVG,
    # so it stays raster (PNG) — the rare "absolutely necessary" graphic — in LFS.
    og_card().write_image(str(ASSETS / "og-card.png"), format="png",
                          width=1200, height=630, scale=1)
    print("wrote", (ASSETS / "og-card.png").relative_to(HERE))
