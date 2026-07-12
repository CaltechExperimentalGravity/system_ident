# docs/darm_tv_demo.py
"""Presentation-only glue for the DARM drift-tracking example page (13).

NOT package API — the docs sibling of ``darm_demo``. It runs the round-1
time-varying campaign (inject a known slow drift into one actuation strength,
snapshot it leakage-free, fit kappa(t) with the Lataire-Pintelon basis) and
builds the page's plotly panels in the house style. Every figure exports to
SVG (Git LFS) by the page's render.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go

_DOCS = Path(__file__).resolve().parent
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))

import sysid_plots as sp  # noqa: E402
from system_ident import darm_tv as tv  # noqa: E402
from system_ident.darm import DARMLoop, drift_profile  # noqa: E402

# The ESD (TST) stage is the physically drift-prone one (charge on the reaction mass),
# so it is the natural target; its weaker strength also makes the CRB *visible* rather
# than the effectively-noise-free recovery the strong UIM stage gives.
NAME = "TST"
K0 = 0.08              # nominal strength
AMP = 0.05             # 5 % drift amplitude (round-1 placeholder)
PERIOD = 7200.0        # hour-scale drift timescale [s]
TSPAN = 3600.0         # measurement span [s] (one hour of snapshots)
N_SNAP = 25            # number of leakage-free snapshots across the span
NPER = 16              # periods per snapshot (P_eff≈9 → genuine per-bin variance)
ORDER = 5              # Legendre time-basis order


def _twin():
    # Same representative two-component noise floor as example 08.
    loop = DARMLoop.default()
    loop.disturbance_asd = 3.0e-4     # length-noise floor [m/√Hz]
    loop.sensor_asd = 300.0           # DARM readout noise [ct/√Hz]
    return loop


def campaign(seed=777):
    """Snapshot the drifting κ_TST across the span and fit κ(t) with its CRB band."""
    loop = _twin()
    prof = functools.partial(drift_profile, base=K0, amp_frac=AMP,
                             period_s=PERIOD, kind="sine")
    times = np.linspace(0.0, TSPAN, N_SNAP)
    t, khat, sig = tv.track_kappa(loop, NAME, times, prof,
                                  nperseg=4096, n_periods=NPER, seed=seed)
    fit = tv.fit_tv(t, khat, sig, kind="legendre", order=ORDER)
    tg = np.linspace(0.0, TSPAN, 400)
    theta, s_theta, theta_dot, s_dot = fit.predict(tg)
    res = tv.resolvability(fit, base=K0, amp_frac=AMP, period_s=PERIOD, kind="sine",
                           record_s=NPER * 4096 / loop.fs)
    snap_frac = float(np.median(sig / khat))
    return SimpleNamespace(loop=loop, t=t, khat=khat, sig=sig, ktrue=prof(t),
                           tg=tg, theta=theta, s_theta=s_theta, theta_dot=theta_dot,
                           s_dot=s_dot, tg_true=prof(tg), res=res, snap_frac=snap_frac)


# ── figures (house style; linear time axes) ─────────────────────────────────────────
def drift_fig(c, *, height=520):
    """Injected κ(t), the leakage-free snapshots ± σ, and the LP fit ± CRB band."""
    tmin, tgmin = c.t / 60.0, c.tg / 60.0
    fig = go.Figure()
    fig.add_scatter(x=tgmin, y=c.tg_true, mode="lines", name="injected κ(t) — truth",
                    line=dict(color=sp.INK, width=2.4))
    # recovered ± CRB shaded band
    fig.add_scatter(x=tgmin, y=c.theta + c.s_theta, mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=tgmin, y=c.theta - c.s_theta, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=sp._fade(sp.GOLD, 0.20),
                    name="recovered ± CRB", hoverinfo="skip")
    fig.add_scatter(x=tgmin, y=c.theta, mode="lines", name="recovered κ(t) — LP fit",
                    line=dict(color=sp.GOLD, width=2.6))
    fig.add_scatter(x=tmin, y=c.khat, mode="markers", name="per-record snapshots κ̂ ± σ",
                    marker=dict(color=sp.SKY, size=sp.MK_DATA),
                    error_y=dict(type="data", array=c.sig, visible=True,
                                 color=sp._fade(sp.SKY, 0.5), width=0, thickness=1.1))
    fig.update_xaxes(title_text="time [min]")
    fig.update_yaxes(title_text=f"actuation strength  κ_{NAME}")
    fig.update_layout(title=f"Tracking a drifting ESD strength κ_{NAME}(t) — "
                            "snapshots + Lataire–Pintelon fit ± CRB")
    return sp.style(fig, height=height)


def tracking_error_fig(c, *, height=430):
    """Recovered − truth against the ±CRB band — the fit stays inside its own error."""
    tgmin = c.tg / 60.0
    resid = c.theta - c.tg_true
    fig = go.Figure()
    fig.add_scatter(x=tgmin, y=c.s_theta, mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=tgmin, y=-c.s_theta, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=sp._fade(sp.GOLD, 0.18),
                    name="± CRB band", hoverinfo="skip")
    fig.add_scatter(x=tgmin, y=resid, mode="lines", name="recovered − truth",
                    line=dict(color=sp.ROSE, width=2.2))
    fig.add_hline(y=0.0, line=dict(color=sp.GRAY, width=1, dash="dot"))
    fig.update_xaxes(title_text="time [min]")
    fig.update_yaxes(title_text=f"κ_{NAME} tracking error")
    fig.update_layout(title="Tracking error vs the Cramér–Rao band — the recovery is honest")
    return sp.style(fig, height=height)


def resolvability_table(c):
    """The feasibility gate as numbers: is the injected drift resolvable?"""
    r = c.res
    rows = [
        ["injected drift amplitude", f"{r['drift_amp']:.4f}", f"{AMP*100:.0f}% of κ"],
        ["per-snapshot σ_κ (median)", f"{c.snap_frac*K0:.4f}", f"{c.snap_frac*100:.2f}% of κ"],
        ["tracking σ_κ (LP fit, median)", f"{r['sigma_theta_med']:.4f}",
         f"{r['sigma_theta_med']/K0*100:.2f}% of κ"],
        ["resolve ratio  =  amp / tracking σ", f"{r['resolve_ratio']:.0f}×",
         "≫ 1 ⇒ drift resolved, not noise"],
        ["local-stationarity error", f"{r['local_stationarity_err']*100:.2f}%",
         "record ÷ drift timescale — the approximation cost"],
    ]
    return sp.param_table(["feasibility quantity", "value", "note"], rows,
                          caption="Feasibility gate — the injected drift is resolvable "
                                  "(a computed bound, not an eyeball)")
