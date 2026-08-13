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
from system_ident.darm import DARMLoop, darm_o4_asd, drift_profile  # noqa: E402

# The ESD (TST) stage is the physically drift-prone one (charge on the reaction mass),
# so it is the natural κ target; its weaker strength also makes the CRB *visible*.
NAME = "TST"
K0 = 1.0               # nominal strength (hierarchical loop uses κ_i = 1, force-unit offload)
AMP = 0.05             # 5 % drift amplitude (round-1 placeholder)
PERIOD = 7200.0        # hour-scale drift timescale [s]
TSPAN = 3600.0         # measurement span [s] (one hour of snapshots)
N_SNAP = 17            # number of leakage-free snapshots across the span
NPER = 16              # periods per snapshot (P_eff≈9 → genuine per-bin variance)
ORDER = 5              # Legendre time-basis order

# Joint + stochastic round: optical gain κ_C (=g_c), SRC detuning δ, and ESD strength κ_TST all
# drifting as random (Ornstein–Uhlenbeck) wanders at once, recovered in one joint fit per record.
N_JOINT = 13           # snapshots for the (heavier) joint campaign
TAU_S = 1500.0         # drift correlation time [s]
G_C0 = 1.0e6           # nominal optical gain (κ_C carrier)
PX_REAL = (5.0e-17) ** 2  # m² — Pcal multisine drive power vs the real DARM floor (see darm_demo)

# SRC-detuning drift: the *new* physics the coupled plant exposes — the error point wanders around
# a slightly-detuned operating point, moving the (split) cavity pole. δ is a sensing parameter,
# recovered from the Pcal FRF (immune to κ drift), so it composes with the κ snapshot.
DELTA0_DEG = 5.0       # slightly-detuned operating point [deg]
DELTA_AMP = 0.05       # 5 % drift on δ


def _twin():
    """The **new** DARM plant: the M0-damped reduced-QUAD hierarchical actuation (M0/PUM/ESD) with
    the coupled detuned-cavity sensing (SRC detuning splits the cavity pole). ``fmin=10 Hz`` keeps
    the snapshot band in the smooth region above the quad-mode forest. Real Advanced-LIGO O4 DARM
    displacement-noise floor (same as example 08)."""
    loop = DARMLoop.default_reduced(fmin=10.0, hierarchical=True)
    loop.noise_asd = darm_o4_asd   # real Advanced-LIGO O4 DARM displacement-noise floor
    return loop


def campaign(seed=777):
    """Snapshot the drifting κ_TST across the span and fit κ(t) with its CRB band."""
    loop = _twin()
    prof = functools.partial(drift_profile, base=K0, amp_frac=AMP,
                             period_s=PERIOD, kind="sine")
    times = np.linspace(0.0, TSPAN, N_SNAP)
    t, khat, sig = tv.track_kappa(loop, NAME, times, prof,
                                  nperseg=4096, n_periods=NPER, px_total=PX_REAL, seed=seed)
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
    fig.update_layout(title="κ tracking error against the Cramér–Rao band")
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


# ── SRC-detuning drift δ(t): the split cavity pole wandering (new-plant physics) ─────
def campaign_delta(seed=778):
    """Snapshot the drifting SRC detuning δ(t) around the operating point and fit it with its CRB.
    δ is recovered from the Pcal FRF shape (the coupled-cavity split), independent of the κ drift."""
    loop = _twin()
    base = np.radians(DELTA0_DEG)
    prof = functools.partial(drift_profile, base=base, amp_frac=DELTA_AMP,
                             period_s=PERIOD, kind="sine")
    times = np.linspace(0.0, TSPAN, N_SNAP)
    t, dhat, sig = tv.track_delta(loop, times, prof, nperseg=4096, n_periods=NPER, px_total=PX_REAL, seed=seed)
    fit = tv.fit_tv(t, dhat, sig, kind="legendre", order=ORDER)
    tg = np.linspace(0.0, TSPAN, 400)
    theta, s_theta, _, _ = fit.predict(tg)
    res = tv.resolvability(fit, base=base, amp_frac=DELTA_AMP, period_s=PERIOD, kind="sine",
                           record_s=NPER * 4096 / loop.fs)
    snap_frac = float(np.median(sig / dhat))
    return SimpleNamespace(loop=loop, t=t, dhat=dhat, sig=sig, dtrue=prof(t),
                           tg=tg, theta=theta, s_theta=s_theta, tg_true=prof(tg),
                           res=res, snap_frac=snap_frac, base=base)


def delta_drift_fig(c, *, height=520):
    """Injected δ(t), the leakage-free Pcal snapshots ± σ, and the LP fit ± CRB band — in degrees."""
    tmin, tgmin = c.t / 60.0, c.tg / 60.0
    d = np.degrees
    fig = go.Figure()
    fig.add_scatter(x=tgmin, y=d(c.tg_true), mode="lines", name="injected δ(t) — truth",
                    line=dict(color=sp.INK, width=2.4))
    fig.add_scatter(x=tgmin, y=d(c.theta + c.s_theta), mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=tgmin, y=d(c.theta - c.s_theta), mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=sp._fade(sp.ROSE, 0.20),
                    name="recovered ± CRB", hoverinfo="skip")
    fig.add_scatter(x=tgmin, y=d(c.theta), mode="lines", name="recovered δ(t) — LP fit",
                    line=dict(color=sp.ROSE, width=2.6))
    fig.add_scatter(x=tmin, y=d(c.dhat), mode="markers", name="per-record snapshots δ̂ ± σ",
                    marker=dict(color=sp.GREEN, size=sp.MK_DATA),
                    error_y=dict(type="data", array=d(c.sig), visible=True,
                                 color=sp._fade(sp.GREEN, 0.5), width=0, thickness=1.1))
    fig.update_xaxes(title_text="time [min]")
    fig.update_yaxes(title_text="SRC detuning  δ  [deg]")
    fig.update_layout(title="Tracking a drifting SRC detuning δ(t) — the split cavity pole wandering "
                            "(recovered from the Pcal FRF ± CRB)")
    return sp.style(fig, height=height)


def delta_resolvability_table(c):
    r = c.res
    d = np.degrees
    rows = [
        ["injected drift amplitude", f"{d(r['drift_amp']):.4f}°", f"{DELTA_AMP*100:.0f}% of δ"],
        ["per-snapshot σ_δ (median)", f"{d(c.snap_frac*c.base):.4f}°",
         f"{c.snap_frac*100:.2f}% of δ"],
        ["tracking σ_δ (LP fit, median)", f"{d(r['sigma_theta_med']):.4f}°",
         f"{r['sigma_theta_med']/c.base*100:.2f}% of δ"],
        ["resolve ratio  =  amp / tracking σ", f"{r['resolve_ratio']:.0f}×",
         "≫ 1 ⇒ drift resolved, not noise"],
        ["local-stationarity error", f"{r['local_stationarity_err']*100:.2f}%",
         "record ÷ drift timescale"],
    ]
    return sp.param_table(["feasibility quantity", "value", "note"], rows,
                          caption="SRC-detuning drift is resolvable — δ tracked from the Pcal FRF, "
                                  "independent of the κ drift (the two snapshots compose).")


# ── Round 2: everything drifts at once, randomly — joint recovery + untangling ───────
_JOINT_LABELS = {"g_c": "κ_C (optical gain)", "delta": "δ (SRC detuning)",
                 "kappa_TST": "κ_ESD (test-mass drive)"}
_JOINT_COLORS = {"g_c": sp.SKY, "delta": sp.ROSE, "kappa_TST": sp.GOLD}


def campaign_joint(seed=2024):
    """Drift κ_C, δ and κ_ESD **together** as random (OU) wanders and recover all three in one joint
    fit per record. Returns per-parameter truth/snapshots/LP-fit (as *fractional* drift, % of
    nominal, so the three share an axis) plus the mean snapshot correlation matrix."""
    loop = _twin().with_params(delta=np.radians(DELTA0_DEG))     # detuned operating point
    nom = {"g_c": G_C0, "delta": np.radians(DELTA0_DEG), "kappa_TST": K0}
    times = np.linspace(0.0, TSPAN, N_JOINT)
    series = {"g_c": tv.stochastic_drift(times, G_C0, amp_frac=0.04, tau_s=TAU_S, seed=seed),
              "delta": tv.stochastic_drift(times, np.radians(DELTA0_DEG), amp_frac=0.05,
                                           tau_s=TAU_S, seed=seed + 1),
              "kappa_TST": tv.stochastic_drift(times, K0, amp_frac=0.05, tau_s=TAU_S, seed=seed + 2)}
    t, th, sg, corr, names = tv.track_joint(loop, series, times, nperseg=4096, n_periods=NPER,
                                            px_total=PX_REAL, seed=seed + 10)
    tg = np.linspace(0.0, TSPAN, 400)
    curves = {}
    for n in names:
        fit = tv.fit_tv(t, th[n], sg[n], kind="legendre", order=4)
        theta, s_theta, _, _ = fit.predict(tg)
        pc = 100.0 / nom[n]                                     # → % of nominal
        curves[n] = dict(t=t / 60.0, snap=(th[n] / nom[n] - 1) * 100, snap_sig=sg[n] * pc,
                         tg=tg / 60.0, fit=(theta / nom[n] - 1) * 100, band=s_theta * pc,
                         truth=(series[n] / nom[n] - 1) * 100)
    return SimpleNamespace(curves=curves, names=names, corr=corr,
                           labels=[_JOINT_LABELS[n] for n in names])


def joint_drift_fig(cj=None, *, height=560):
    """All three drifts at once, as fractional deviation from nominal — injected random wander,
    per-record joint snapshots ± σ, and the LP fit ± CRB band, on one axis."""
    if cj is None:
        cj = campaign_joint()
    fig = go.Figure()
    for n in cj.names:
        c = cj.curves[n]; col = _JOINT_COLORS[n]; lab = _JOINT_LABELS[n]
        fig.add_scatter(x=c["tg"], y=c["truth"], mode="lines", showlegend=False, legendgroup=n,
                        line=dict(color=col, width=1.4, dash="dot"),
                        hovertemplate=lab + " truth: %{y:.2f}%<extra></extra>")
        fig.add_scatter(x=c["tg"], y=c["fit"] + c["band"], mode="lines", line=dict(width=0),
                        showlegend=False, legendgroup=n, hoverinfo="skip")
        fig.add_scatter(x=c["tg"], y=c["fit"] - c["band"], mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=sp._fade(col, 0.15), showlegend=False,
                        legendgroup=n, hoverinfo="skip")
        fig.add_scatter(x=c["tg"], y=c["fit"], mode="lines", name=lab,
                        line=dict(color=col, width=2.6), legendgroup=n)
        fig.add_scatter(x=c["t"], y=c["snap"], mode="markers", showlegend=False, legendgroup=n,
                        marker=dict(color=col, size=sp.MK_DATA - 1),
                        error_y=dict(type="data", array=c["snap_sig"], visible=True,
                                     color=sp._fade(col, 0.5), width=0, thickness=1.0))
    fig.add_scatter(x=[None], y=[None], mode="lines", name="injected truth (dotted)",
                    line=dict(color=sp.GRAY, width=1.4, dash="dot"))
    fig.update_xaxes(title_text="time [min]")
    fig.update_yaxes(title_text="drift  [% of nominal]")
    fig.update_layout(title="Three parameters drifting at once (random wander) — recovered jointly, "
                            "each within its CRB")
    return sp.style(fig, height=height, legend="v")


def joint_corr_fig(cj=None, *, height=430):
    """The snapshot correlation matrix — which drifts are hard to tell apart."""
    if cj is None:
        cj = campaign_joint()
    lab = [_JOINT_LABELS[n].split(" ")[0] for n in cj.names]
    fig = go.Figure(go.Heatmap(z=cj.corr, x=lab, y=lab, zmin=-1, zmax=1, colorscale="RdBu",
                               reversescale=True, zmid=0,
                               text=np.round(cj.corr, 2), texttemplate="%{text}",
                               colorbar=dict(title="corr")))
    fig.update_layout(title="Separability of the drifts — off-diagonal ≈ 0 is separable, ±1 is "
                            "degenerate")
    return sp.style(fig, height=height, legend="v")


# ── Round 3: the P&S-OPTIMAL designed drive — a few placed lines, not a broadband multisine ───────
# The corrected tracking picture: ONE plant (opto-mechanical DARM sensing C + the quad suspension),
# ONE joint θ, read out by a FEW cal lines whose frequencies are chosen (Bayesian A-optimal, under
# the real force budgets) to minimise the joint estimator variance on the parameter *drifts*. Scope
# = the set the O4 floor + ≥10 Hz lines make identifiable: κ_C (g_c), δ, and the two in-band stages
# κ_PUM, κ_ESD. (κ_M0 acts <0.5 Hz — off this band; f_cc/τ want a wideband high-f spread — the
# design CRB quantifies both. See the honest-gaps note.)
from system_ident import darm_callines as _cl  # noqa: E402

_SCOPE = ("g_c", "delta", "kappa_PUM", "kappa_TST")
_SCOPE_LABELS = {"g_c": "κ_C (optical gain)", "delta": "δ (SRC detuning)",
                 "kappa_PUM": "κ_PUM (L2 drive)", "kappa_TST": "κ_ESD (test-mass drive)"}
_SCOPE_COLORS = {"g_c": sp.SKY, "delta": sp.ROSE, "kappa_PUM": sp.MOSS if hasattr(sp, "MOSS") else sp.INK,
                 "kappa_TST": sp.GOLD}
# Drift priors: κ_C from Sun et al. 2020 (measured 1–2 %); the rest are REPRESENTATIVE placeholders
# (the O3/O4 papers do not quantify per-parameter drift for δ/κ_PUM/κ_ESD).
_DESIGN_PRIORS = {"g_c": _cl.KAPPA_C_DRIFT_FRAC, "delta": 0.02, "kappa_PUM": 0.02, "kappa_TST": 0.02}
_PORT_COLOR = {"PCAL": sp.SKY, "PUM": sp.INK, "TST": sp.GOLD}
N_LINE_SNAP = 9        # snapshots for the (heavy, simulated) designed-line tracking campaign
NPER_LINE = 24         # periods per designed-line snapshot


def design_campaign():
    """Design the A-optimal cal-line roster for the identifiable joint set and score it against a
    broadband multisine of the same footprint. Returns the roster, per-parameter drift-resolvability
    margins, the two A-optimal costs, and the DARM floor for the stem plot."""
    loop = _twin().with_params(delta=np.radians(DELTA0_DEG))
    d = _cl.design_lines(loop, _DESIGN_PRIORS, T=NPER_LINE * 4096 / loop.fs, n_pcal=3, names=_SCOPE)
    # broadband reference: equal Pcal lines flat across the band (the OLD white-ish drive)
    caps = _cl.stage_force_caps(loop, names=_SCOPE)
    nom = _cl._nominal(loop, _SCOPE)
    frac = {"g_c", "kappa_PUM", "kappa_TST"}
    abs_std = {n: (_DESIGN_PRIORS[n] * abs(nom[i]) if n in frac else _DESIGN_PRIORS[n])
               for i, n in enumerate(_SCOPE)}
    P = _cl._prior_cov(abs_std, _SCOPE)
    bb = _cl.broadband_roster(loop, _SCOPE, caps, n_per_port=10)   # power spread thin on every port
    cost_bb = _cl.a_optimal_cost(_cl.joint_fisher(loop, bb, NPER_LINE * 4096 / loop.fs, names=_SCOPE)[0], P)
    ff = np.geomspace(loop.fmin, min(loop.fmax, 1400.0), 1200)
    floor = loop.displacement_noise_asd(ff)
    return SimpleNamespace(loop=loop, roster=d["roster"], margins=d["margins"],
                           sigma_frac=d["sigma_frac"], cost=d["cost"], cost_bb=cost_bb,
                           full_rank=d["full_rank"], ff=ff, floor=floor, priors=abs_std, nom=nom)


def design_fig(dc=None, *, height=520):
    """The designed cal lines as stems at their DARM-displacement amplitude over the real O4 floor —
    a few optimally-placed lines (Pcal + one per in-band stage), not a broadband comb."""
    if dc is None:
        dc = design_campaign()
    fig = go.Figure()
    fig.add_scatter(x=dc.ff, y=dc.floor, mode="lines", name="O4 DARM displacement floor",
                    line=dict(color=sp.INK, width=2.2, dash="dot"), hoverinfo="skip")
    seen = set()
    for f, port, disp in dc.roster:
        col = _PORT_COLOR.get(port, sp.GRAY)
        floor_f = float(np.interp(f, dc.ff, dc.floor))
        fig.add_scatter(x=[f, f], y=[floor_f, disp], mode="lines", showlegend=False,
                        line=dict(color=col, width=2.2))
        fig.add_scatter(x=[f], y=[disp], mode="markers",
                        name=port if port not in seen else None,
                        showlegend=port not in seen,
                        marker=dict(color=col, size=sp.MK_BIG, symbol="diamond",
                                    line=dict(color=sp.INK, width=1.0)),
                        hovertemplate=f"{port} line<br>%{{x:.1f}} Hz<br>%{{y:.2e}} m<extra></extra>")
        seen.add(port)
    yr = sp._logy_range([dc.floor, np.array([d for _, _, d in dc.roster])], decades=6)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="DARM displacement [m rms]")
    fig.update_layout(title="A-optimal cal lines over the O4 floor — placed where the joint drift "
                            "estimate is sharpest, under the real force budgets")
    return sp.style(fig, height=height, legend="v")


def design_table(dc=None):
    """Per-parameter drift prior, the designed-roster snapshot 1σ, and the resolvability margin
    (prior/σ; >1 ⇒ a snapshot resolves the drift), plus the A-optimal cost vs a broadband drive."""
    if dc is None:
        dc = design_campaign()
    prov_note = {"g_c": "Sun 2020 (measured 1–2 %)", "delta": "representative placeholder",
                 "kappa_PUM": "representative placeholder", "kappa_TST": "representative placeholder"}
    rows = []
    for i, n in enumerate(_SCOPE):
        frac = n in {"g_c", "kappa_PUM", "kappa_TST"}
        prior_frac = dc.priors[n] / (abs(dc.nom[i]) if frac else 1.0)
        rows.append([_SCOPE_LABELS[n],
                     f"{prior_frac*100:.1f}%" if frac else f"{np.degrees(dc.priors[n]):.2f}°",
                     f"{dc.sigma_frac[n]:.2e}", f"{dc.margins[n]:.0f}×", prov_note[n]])
    return sp.param_table(["parameter", "drift prior (1σ)", "snapshot σ", "margin", "prior source"],
                          rows,
                          caption=f"A-optimal designed drive: joint cost {dc.cost:.3f} vs {dc.cost_bb:.3f} "
                                  f"for a broadband multisine of the same budget. The cost is "
                                  f"Σ (snapshot σ / drift prior)² = Σ 1/margin² (lower = better; "
                                  f"< 1 per parameter ⇒ its drift is resolved). Every drift here is "
                                  f"resolved (margin > 1). κ_C's prior is measured (Sun 2020); the "
                                  f"others are representative placeholders.")


def tracked_lines_campaign(seed=4041):
    """Track the identifiable joint set drifting together (random OU wander), read out by the
    **designed** cal lines (not a broadband multisine). Returns per-parameter fractional-drift
    curves (truth, per-snapshot ± σ, LP fit ± CRB) for the recovery figure."""
    dc = design_campaign()
    loop = dc.loop
    nom = {"g_c": G_C0, "delta": np.radians(DELTA0_DEG), "kappa_PUM": K0, "kappa_TST": K0}
    times = np.linspace(0.0, TSPAN, N_LINE_SNAP)
    series = {n: tv.stochastic_drift(times, nom[n], amp_frac=(0.04 if n == "g_c" else 0.05),
                                     tau_s=TAU_S, seed=seed + k) for k, n in enumerate(_SCOPE)}
    t, th, sg, corr, names = tv.track_joint_lines(loop, series, times, dc.roster,
                                                  nperseg=4096, n_periods=NPER_LINE, seed=seed + 20)
    tg = np.linspace(0.0, TSPAN, 400)
    curves = {}
    for n in names:
        fit = tv.fit_tv(t, th[n], sg[n], kind="legendre", order=3)
        theta, s_theta, _, _ = fit.predict(tg)
        pc = 100.0 / nom[n]
        curves[n] = dict(t=t / 60.0, snap=(th[n] / nom[n] - 1) * 100, snap_sig=sg[n] * pc,
                         tg=tg / 60.0, fit=(theta / nom[n] - 1) * 100, band=s_theta * pc,
                         truth=(series[n] / nom[n] - 1) * 100)
    return SimpleNamespace(curves=curves, names=names, corr=corr,
                           labels=[_SCOPE_LABELS[n] for n in names])


def tracked_lines_fig(tc=None, *, height=560):
    """The identifiable joint set recovered from the DESIGNED cal-line readout — injected random
    wander, per-record joint snapshots ± σ, and the LP fit ± CRB, as % of nominal on one axis."""
    if tc is None:
        tc = tracked_lines_campaign()
    fig = go.Figure()
    for n in tc.names:
        c = tc.curves[n]; col = _SCOPE_COLORS.get(n, sp.INK); lab = _SCOPE_LABELS[n]
        fig.add_scatter(x=c["tg"], y=c["truth"], mode="lines", showlegend=False, legendgroup=n,
                        line=dict(color=col, width=1.4, dash="dot"),
                        hovertemplate=lab + " truth: %{y:.2f}%<extra></extra>")
        fig.add_scatter(x=c["tg"], y=c["fit"] + c["band"], mode="lines", line=dict(width=0),
                        showlegend=False, legendgroup=n, hoverinfo="skip")
        fig.add_scatter(x=c["tg"], y=c["fit"] - c["band"], mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=sp._fade(col, 0.15), showlegend=False,
                        legendgroup=n, hoverinfo="skip")
        fig.add_scatter(x=c["tg"], y=c["fit"], mode="lines", name=lab,
                        line=dict(color=col, width=2.6), legendgroup=n)
        fig.add_scatter(x=c["t"], y=c["snap"], mode="markers", showlegend=False, legendgroup=n,
                        marker=dict(color=col, size=sp.MK_DATA - 1),
                        error_y=dict(type="data", array=c["snap_sig"], visible=True,
                                     color=sp._fade(col, 0.5), width=0, thickness=1.0))
    fig.add_scatter(x=[None], y=[None], mode="lines", name="injected truth (dotted)",
                    line=dict(color=sp.GRAY, width=1.4, dash="dot"))
    fig.update_xaxes(title_text="time [min]")
    fig.update_yaxes(title_text="drift  [% of nominal]")
    fig.update_layout(title="Tracked by the designed cal lines — the identifiable joint set "
                            "recovered within its CRB from a few optimal tones")
    return sp.style(fig, height=height, legend="v")
