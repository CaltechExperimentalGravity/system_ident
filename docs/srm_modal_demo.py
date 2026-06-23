"""Presentation-only glue for the SRM modal example page (10).

NOT package API — the docs sibling of ``sysid_plots``. The SRM is an HSTS
suspension identified through its REAL production L1-SRM top-mass dampers, closed
around the shared bare-M1 6x6 HSTS plant, on the Phase-1 RTSfreerun digital twin.

This helper does NOT re-drive the twin at render time. The slow MIMO campaign
(one Pintelon-Schoukens multisine per DoF through the closed loops, ~13 min on the
twin) is run once by ``experiments/rtsfreerun/run_srm6dof_modal.py`` and cached to
``srm_campaign_cache_n65536.npz``. Here we LOAD that cache and re-run only the fast,
offline part: recover the open-loop FRF, fit the rank-1 joint modal model, propagate
the Cramer-Rao bound, and build the page's plotly panels in the house style. If the
cache is genuinely missing, we fall back to running the campaign (the page is
``freeze: true``, so it executes once and the committed freeze artifact is what CI
deploys). Every figure is exported to SVG (Git LFS) by the page's render.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go

_DOCS = Path(__file__).resolve().parent
_ROOT = _DOCS.parent
for _p in (_DOCS, _ROOT / "experiments" / "rtsfreerun", _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sysid_plots as sp  # noqa: E402
import srm6dof_loop as s6  # noqa: E402
import run_srm6dof_modal as R  # noqa: E402  (reuse the experiment's load + fit + score path)
from system_ident.design.resolution import recommend_resolution  # noqa: E402

# The fit picks the most modes recovered well in both f0 and Q; the experiment's sweep
# settles on 13 (the resolvable design-mode count). We fit that directly here so the
# offline render is one deterministic fit, not the full sweep.
N_MODES = 13
DOFS = ["L", "T", "V", "R", "P", "Y"]


@lru_cache(maxsize=1)
def run():
    """Load the cached SRM campaign and re-run only the offline fit/oracle/CRB path.

    Returns everything the figures and tables need. Cached so all panels reuse a
    single deterministic fit under ``freeze: true``.
    """
    m6 = s6.SRM6DOF()                       # cheap: builds the compiled model, no campaign
    exps, freq = R.load_campaign()
    if exps is None:                        # cache missing -> run the full experiment once
        cal, taus, stable, _mu, ora, rows = R.main()
        exps, freq = R.load_campaign()
        m6.set_cal({d: cal[d] for d in m6.dofs})
    else:
        # re-apply the tuned CAL so the live model (used here only for the analytic
        # oracle) matches the campaign that produced the cache.
        cp = R._cal_cache()
        if cp.exists():
            d = np.load(cp)
            m6.set_cal({k: float(d["cal"][i]) for i, k in enumerate(m6.dofs)})

    ora = R.oracle_modes(m6)                # 16 analytic in-band poles
    distinct = R.distinct_oracle_modes(ora)  # 13 RESOLVABLE design modes (doublets collapse)

    # open-loop recovery through the closed SRM loops + analytic oracle tensor
    Xmat = np.stack([exps[l][1] for l in range(6)], axis=-1)
    Ymat = np.stack([exps[l][0] for l in range(6)], axis=-1)
    Gnp = R.recover_open_loop(Xmat, Ymat)
    Goracle = m6.oracle_tensor(freq)
    diag_rel = float(np.median([np.median(np.abs(Gnp[:, i, i] - Goracle[:, i, i])
                                          / np.abs(Goracle[:, i, i])) for i in range(6)]))

    # the rank-1 joint MIMO modal fit (prior-seeded from the resolvable design modes)
    model, res, _Gnp, mu, dof = R.fit_modal(exps, freq, N_MODES, prior_modes=distinct)
    Gfit = model.eval(res.theta, freq)
    rows, sc = R.score_fit(mu, ora)

    # tau-driven design: what resolution does the prior demand to resolve these Qs?
    fs = 256.0                              # campaign fs (cache is fs=256 Hz, nperseg=65536)
    reco_n, reco_df, reco_nt = recommend_resolution(distinct, fs, bins_per_fwhm=4)
    used_df = float(freq[1] - freq[0])
    used_n = int(round(fs / used_df))

    # per-mode ringdown / FWHM of the prior modes (for the design figure)
    design = [dict(f0=f0, Q=q, fwhm=f0 / q, tau=q / (np.pi * f0)) for f0, q in distinct]

    return SimpleNamespace(
        freq=freq, Gnp=Gnp, Gfit=Gfit, Goracle=Goracle, diag_rel=diag_rel,
        rows=rows, sc=sc, mu=mu, dof=dof, cost=res.cost, n_iter=res.n_iter,
        ora=ora, distinct=distinct, design=design,
        fs=fs, used_df=used_df, used_n=used_n,
        reco_n=reco_n, reco_df=reco_df, reco_nt=reco_nt)


# numbers the prose pulls inline ----------------------------------------------------
def headline():
    d = run()
    n_good = d.sc["n_good"]
    q_med = d.sc["q_med_wellsep"] * 100.0
    df_med = float(np.median([abs(r["df_pct"]) for r in d.rows]))
    return SimpleNamespace(
        diag_rel=d.diag_rel, n_modes=len(d.rows), n_good=n_good, q_med=q_med,
        df_med=df_med, n_wellsep=d.sc["n_wellsep"], dof=d.dof,
        used_df=d.used_df, used_period=d.used_n / d.fs,
        reco_df=d.reco_df, reco_period=d.reco_n / d.fs, reco_nt=d.reco_nt)


# ── figure (a): through-resonance recovery on a representative diagonal element ──
def diag_recovery_fig(dof="L", *, height=560):
    """Recovered open-loop diagonal FRF for one DoF: nonparametric Y.X^-1 (noisy near
    the peaks), the rank-1 joint modal fit, and the analytic state-space oracle. The
    fit threads the resonances where the per-bin inverse scatters.
    """
    d = run()
    j = DOFS.index(dof)
    f = d.freq
    gnp = np.abs(d.Gnp[:, j, j])
    gfit = np.abs(d.Gfit[:, j, j])
    gora = np.abs(d.Goracle[:, j, j])
    fig = go.Figure()
    fig.add_scatter(x=f, y=gnp, mode="markers", name="recovered  Y·X⁻¹  (nonparametric)",
                    marker=dict(color=sp._fade(sp.GRAY, 0.7), size=sp.MK_SMALL))
    fig.add_scatter(x=f, y=gfit, mode="lines", name="rank-1 joint modal fit",
                    line=dict(color=sp.GOLD, width=2.6))
    fig.add_scatter(x=f, y=gora, mode="lines", name="analytic state-space oracle",
                    line=dict(color=sp.SKY, width=2.0, dash="dash"))
    yr = sp._logy_range([gora, gfit], decades=4)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text=f"|G[{dof},{dof}](f)|")
    fig.update_layout(
        title=f"{dof}→{dof} open-loop recovery through the closed SRM loops — "
              f"fit vs nonparametric vs oracle")
    return sp.style(fig, height=height)


# ── figure (b): the tau-driven resolution design (FWHM / ringdown vs the prior) ──
def resolution_design_fig(*, height=560):
    """The measurement-design point: each prior mode (f0, Q≈50) has FWHM Δf = f0/Q and
    ringdown τ = Q/(π f0). Resolving Q needs df ≤ Δf/(a few bins); the high-Q modes ring
    a long time, so the period T = 1/df is long. Plot Δf per mode with the achieved df and
    the prior-recommended df marked, plus τ on a second axis.
    """
    d = run()
    f0 = np.array([m["f0"] for m in d.design])
    fwhm = np.array([m["fwhm"] for m in d.design])
    tau = np.array([m["tau"] for m in d.design])
    fig = go.Figure()
    # FWHM of each prior mode (markers)
    fig.add_scatter(x=f0, y=fwhm, mode="markers", name="prior mode FWHM  Δf = f₀/Q",
                    marker=dict(color=sp.SKY, size=sp.MK_BIG,
                                line=dict(color="white", width=1.5)))
    # the achieved campaign df and the physics-recommended df
    fig.add_hline(y=d.used_df, line=dict(color=sp.GOLD, width=2.2),
                  annotation_text=f"achieved df = {d.used_df:.4f} Hz  "
                                  f"({d.used_n / d.fs:.0f} s/period)",
                  annotation_position="top left",
                  annotation_font=dict(color=sp.GOLD))
    fig.add_hline(y=d.reco_df, line=dict(color=sp.ROSE, width=2.0, dash="dot"),
                  annotation_text=f"recommend_resolution → df = {d.reco_df:.4f} Hz  "
                                  f"({d.reco_n / d.fs:.0f} s/period)",
                  annotation_position="bottom left",
                  annotation_font=dict(color=sp.ROSE))
    # ringdown tau on the right axis
    fig.add_scatter(x=f0, y=tau, mode="markers", name="ringdown  τ = Q/(π f₀)  [s]",
                    yaxis="y2", marker=dict(color=sp.GREEN, size=sp.MK_DATA,
                                            symbol="diamond"))
    fig.update_xaxes(type="log", title_text="mode frequency f₀ [Hz]")
    fig.update_yaxes(type="log", title_text="Δf , df  [Hz]")
    fig.update_layout(
        yaxis2=dict(title="ringdown τ [s]", overlaying="y", side="right",
                    showgrid=False, type="log"),
        title="τ-driven measurement design — Q resolution is gated by the ringdown")
    return sp.style(fig, height=height)


# ── figure (c) / table: the recovered modal table vs oracle ────────────────────
def modal_table():
    """Recovered shared modal poles (f0/Q ± CRB) vs the analytic oracle, with the two
    collapsed doublet clusters flagged. 'collapsed' = |df|≥1% (the unresolvable doublets).
    """
    d = run()
    rows = []
    for r in d.rows:
        collapsed = abs(r["df_pct"]) >= 1.0 or not np.isfinite(r["q_err"])
        flag = " ⚠ collapsed cluster" if collapsed else ""
        rows.append([
            f"{r['f0']:.4f} ± {r['f0_std']:.1e}",
            f"{r['Q']:.1f} ± {r['Q_std']:.1e}" if np.isfinite(r["Q"]) else f"{r['Q']:.1f}",
            f"{r['f0_oracle']:.4f}",
            f"{r['Q_oracle']:.0f}",
            f"{r['df_pct']:+.3f}",
            (f"{r['q_err'] * 100:.1f}" if np.isfinite(r["q_err"]) else "—") + flag,
        ])
    return sp.param_table(
        ["f₀ fit ± CRB [Hz]", "Q fit ± CRB", "f₀ oracle [Hz]", "Q oracle",
         "Δf₀ [%]", "ΔQ [%]"],
        rows,
        caption=f"Shared modal poles (n_modes={len(d.rows)}) recovered through the "
                f"production loops vs the analytic oracle")


def modal_recovery_fig(*, height=520):
    """(fitted − oracle) f0 in %, per mode, with the Cramer-Rao ±1σ bars; the two
    collapsed doublet clusters drawn in rose, the well-separated modes in gold."""
    d = run()
    fig = go.Figure()
    fig.add_hline(y=0.0, line_color="rgba(100,120,160,0.6)")
    for collapsed, color, name in ((False, sp.GOLD, "well-separated mode"),
                                   (True, sp.ROSE, "collapsed doublet cluster")):
        sel = [r for r in d.rows
               if (abs(r["df_pct"]) >= 1.0 or not np.isfinite(r["q_err"])) == collapsed]
        if not sel:
            continue
        x = [r["f0_oracle"] for r in sel]
        y = [r["df_pct"] for r in sel]
        ey = [r["f0_std"] / r["f0_oracle"] * 100.0 for r in sel]
        fig.add_scatter(x=x, y=y, mode="markers", name=name,
                        marker=dict(color=color, size=sp.MK_BIG,
                                    line=dict(color="white", width=1.5)),
                        error_y=dict(type="data", array=ey, visible=True,
                                     color=sp._fade(color, 0.5), thickness=2, width=8))
    fig.update_xaxes(type="log", title_text="oracle mode frequency [Hz]")
    fig.update_yaxes(title_text="(fitted − oracle) f₀  [%]")
    fig.update_layout(
        title="Modal frequency recovery — fit − oracle with the Cramér–Rao bound")
    return sp.style(fig, height=height)
