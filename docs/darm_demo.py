# docs/darm_demo.py
"""Presentation-only glue for the DARM calibration example page (08).

NOT package API — the docs sibling of ``sysid_plots``. It runs the DARM twin
campaigns (Pcal response, sensing fit, actuation kappas, swept-sine comparison)
and builds the page's plotly panels in the shared house style.  Every figure is
exported to SVG (Git LFS) by the page's render.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_DOCS = Path(__file__).resolve().parent
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))

import sysid_plots as sp  # noqa: E402
from system_ident.darm import (  # noqa: E402
    DARMLoop, recover_response, fit_sensing, recover_actuation,
    multisine_response_sigma, swept_sine_response_sigma, sweep_time_to_match_coverage,
)
from system_ident.backends.darm_adapter import DARMBackend  # noqa: E402
from system_ident.excitation import multisine_from_psd  # noqa: E402
from system_ident.loop import SysIDLoop  # noqa: E402

# NPER=16 so the 3 s ramp leaves ~10 full periods → a genuine per-bin variance
# (P_eff≈9). NPER=8 would leave only 2 full periods → P_eff=1 → fabricated CRB bars.
NPERSEG, NPER = 4096, 16


def _grid(loop):
    fa = np.fft.rfftfreq(NPERSEG, 1 / loop.fs)
    band = (fa >= loop.fmin) & (fa <= loop.fmax)
    return fa, band, fa[band]


def _twin(seed=1):
    # Representative noise tuned so the per-bin σ(R)/R is a visible ~1% (the O4-era
    # cal target scale), NOT the effectively-noise-free ~1e-8 that a tiny sensor_asd
    # against g_C=1e6 gives. disturbance (length noise, via C/(1+G)) is comparable at
    # the low band; sensor (readout, via 1/(1+G)) dominates higher — a two-component floor.
    loop = DARMLoop.default()
    loop.disturbance_asd = 3.0e-4     # representative length-noise floor [m/√Hz]
    loop.sensor_asd = 300.0           # representative DARM readout noise [ct/√Hz]
    return loop


# ── campaigns ─────────────────────────────────────────────────────────────────
def pcal_audit(seed=1):
    loop = _twin()
    fa, band, freq = _grid(loop)
    px_total = 1.0
    Pxx = np.full_like(freq, px_total / (freq[-1] - freq[0]))
    x = multisine_from_psd(Pxx, loop.fs, NPERSEG, NPER, freq, seed=np.random.default_rng(seed))
    be = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", seed=seed)
    be.inject("PCAL_EXC", x, loop.fs)
    dur = (NPERSEG * NPER) / loop.fs
    seg = be.read(["PCAL_EXC", "DARM_ERR"], dur)
    H, H_err, coh = SysIDLoop._estimate_tf_periodic(seg["PCAL_EXC"], seg["DARM_ERR"],
                                                    loop.fs, NPERSEG, band, n_transient=1)
    excited = np.isfinite(H_err)
    R, R_sig = recover_response(H, H_err)
    one_plus_G = 1.0 + loop.G(freq)
    C_meas = H * one_plus_G
    p, s = fit_sensing(freq, C_meas, H_err * np.abs(one_plus_G), p0=(0.8e6, 300.0, 50e-6))
    t = np.arange(len(seg["PCAL_EXC"])) / loop.fs
    ff = np.geomspace(loop.fmin, loop.fmax, 600)
    return SimpleNamespace(loop=loop, t=t, drive=seg["PCAL_EXC"], derr=seg["DARM_ERR"],
                           freq=freq, band=band, H=H, H_err=H_err, coh=coh, excited=excited,
                           R=R, R_sigma=R_sig, C_meas=C_meas, fit=p, sigma=s, ff=ff)


def actuation_campaign(seed=2):
    loop = _twin()
    fa, band, freq = _grid(loop)
    Pxx = np.full_like(freq, 1.0 / (freq[-1] - freq[0]))
    # Pcal reference
    bp = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", seed=seed)
    xp = multisine_from_psd(Pxx, loop.fs, NPERSEG, NPER, freq, seed=np.random.default_rng(seed))
    bp.inject("PCAL_EXC", xp, loop.fs)
    sp_seg = bp.read(["PCAL_EXC", "DARM_ERR"], (NPERSEG * NPER) / loop.fs)
    Hp, Hp_err, _ = SysIDLoop._estimate_tf_periodic(sp_seg["PCAL_EXC"], sp_seg["DARM_ERR"],
                                                    loop.fs, NPERSEG, band, n_transient=1)
    rows = []
    for name in ("UIM", "PUM", "TST"):
        be = DARMBackend(loop, {"EXC": name}, "DARM_ERR", seed=seed + 1)
        xi = multisine_from_psd(Pxx, loop.fs, NPERSEG, NPER, freq, seed=np.random.default_rng(seed + 2))
        be.inject("EXC", xi, loop.fs)
        si = be.read(["EXC", "DARM_ERR"], (NPERSEG * NPER) / loop.fs)
        Hi, Hi_err, _ = SysIDLoop._estimate_tf_periodic(si["EXC"], si["DARM_ERR"],
                                                        loop.fs, NPERSEG, band, n_transient=1)
        tf, true_k = loop.stages[name]
        N = tf.eval(freq)
        comb = np.hypot(Hi_err / np.abs(Hi), Hp_err / np.abs(Hp)) * np.abs(Hi / Hp)
        k, ks = recover_actuation(freq, Hi, Hp, N, comb)
        rows.append((name, true_k, k, ks))
    return SimpleNamespace(loop=loop, rows=rows)


def comparison(seed=0):
    loop = _twin()
    freq, R, R_sig, T = multisine_response_sigma(loop, nperseg=NPERSEG, n_periods=NPER,
                                                 px_total=1.0, seed=seed)
    # equal wall-clock sweep: in the SAME T, a 4-period dwell resolves only ~T*fs/(4*nperseg)
    # frequencies (= NPER//4 = 4 points here), vs the multisine's whole-band coverage.
    # 4 periods are the minimum for a genuine per-bin variance (P_eff≥3); 2 periods give
    # P_eff=2 which underflows to the estimator's 1e-9 floor and produces a fabricated σ.
    n_pts = NPER // 4
    pts = np.geomspace(loop.fmin, loop.fmax, n_pts)
    fp, ssweep, T_used = swept_sine_response_sigma(loop, pts, nperseg=NPERSEG,
                                                   dwell_periods=4, px_total=1.0, seed=seed)
    t_cover = sweep_time_to_match_coverage(loop, nperseg=NPERSEG, dwell_periods=4)
    return SimpleNamespace(loop=loop, freq=freq, frac_ms=R_sig / np.abs(loop.R(freq)),
                           excited=np.isfinite(R_sig), pts=fp,
                           frac_sweep=ssweep / np.abs(loop.R(fp)),
                           T=T, T_used=T_used, t_cover=t_cover,
                           n_bins=int(np.isfinite(R_sig).sum()), n_pts=n_pts)


# ── figures (house style; data-driven y-ranges) ───────────────────────────────
def truth_fig(height=560):
    loop = _twin()
    ff = np.geomspace(loop.fmin, loop.fmax, 700)
    fig = make_subplots(rows=1, cols=1)
    series = [("|C| sensing", np.abs(loop.C(ff)), sp.SKY),
              ("|A| actuation", np.abs(loop.A(ff)), sp.GOLD),
              ("|G| open-loop", np.abs(loop.G(ff)), sp.GREEN),
              ("|R| response", np.abs(loop.R(ff)), sp.ROSE)]
    for name, y, c in series:
        fig.add_scatter(x=ff, y=y, mode="lines", name=name, line=dict(color=c, width=2.4))
    yr = sp._logy_range([y for _, y, _ in series], decades=13)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="magnitude")
    fig.update_layout(title="DARM twin — sensing C, actuation A, open-loop G, response R")
    return sp.style(fig, height=height)


def pcal_timeseries_fig(a, *, height=520):
    drive_tr = [("Pcal multisine (3 s Tukey on/off)", a.drive, sp.GOLD)]
    motion_tr = [("DARM error d_err", a.derr, sp.SKY)]
    return sp.timeseries(a.t, drive_tr, motion_tr, height=height,
                         drive_unit="x_pc drive [a.u.]", motion_unit="d_err [ct]",
                         titles=["<b>Pcal excitation</b> — the injected multisine",
                                 "<b>DARM error</b> — response under disturbance + readout noise"])


def pcal_bode_fig(a, *, height=760):
    traces = [dict(name="analytic C/(1+G)", H=a.loop.frf_pcal(a.freq), color=sp.INK, width=2.4),
              dict(name="measured FRF", H=a.H, color=sp.ROSE, mode="markers",
                   err=a.H_err, mask=a.excited)]
    return sp.bode(a.freq, traces, coh=a.coh, coh_mask=a.excited, height=height,
                   ylabel="|d_err/x_pc|")


def response_envelope_fig(a, *, height=520):
    """R(f) with its ±σ CRB envelope vs the analytic truth."""
    m = a.excited
    f, R, s = a.freq[m], np.abs(a.R[m]), a.R_sigma[m]
    fig = go.Figure()
    fig.add_scatter(x=a.ff, y=np.abs(a.loop.R(a.ff)), mode="lines",
                    name="analytic R", line=dict(color=sp.INK, width=2.4))
    fig.add_scatter(x=f, y=R, mode="markers", name="recovered R",
                    marker=dict(color=sp.GOLD, size=sp.MK_DATA),
                    error_y=dict(type="data", array=s, visible=True,
                                 color=sp._fade(sp.GOLD, 0.4), width=0, thickness=1.1))
    yr = sp._logy_range([np.abs(a.loop.R(a.ff)), R], decades=4)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="|R(f)|  [m/ct]")
    fig.update_layout(title="DARM response R(f) = 1/(d_err/x_pc) — recovered ± CRB vs truth")
    return sp.style(fig, height=height)


def sensing_table(a):
    p, s = a.fit, a.sigma
    rows = [["optical gain g_C [ct/m]", f"{a.loop.g_c:.4g}", f"{p['g_c']:.4g}", f"{s['g_c']:.2g}"],
            ["cavity pole f_cc [Hz]", f"{a.loop.f_cc:.2f}", f"{p['f_cc']:.2f}", f"{s['f_cc']:.2f}"],
            ["delay τ [µs]", f"{a.loop.tau*1e6:.1f}", f"{p['tau']*1e6:.1f}", f"{s['tau']*1e6:.2f}"]]
    return sp.param_table(["sensing parameter", "true", "recovered", "σ (CRB)"], rows,
                          caption="Sensing function C — representative truth vs P&S recovery")


def actuation_table(d):
    rows = [[n, f"{tk:.3f}", f"{k:.3f}", f"{ks:.2g}", f"{abs(k-tk)/tk*100:.2f}%"]
            for (n, tk, k, ks) in d.rows]
    return sp.param_table(["stage", "true κ", "recovered κ", "σ (CRB)", "|Δ|"], rows,
                          caption="Actuation strengths κ — Pcal as the absolute ruler")


def comparison_fig(c, *, height=520):
    """Equal wall-clock: the multisine's dense whole-band σ(R)/R envelope vs the few
    frequencies a swept sine resolves in the same time. The sweep points sit on (or
    below) the envelope because each spends full power on one line — but it only reaches
    `n_pts` frequencies; matching the multisine's coverage costs it `t_cover` (annotated)."""
    m = c.excited
    fig = go.Figure()
    fig.add_scatter(x=c.freq[m], y=c.frac_ms[m], mode="lines",
                    name=f"P&S multisine — all {c.n_bins} bins in one {c.T:.0f} s window",
                    line=dict(color=sp.GOLD, width=2.6))
    fig.add_scatter(x=c.pts, y=c.frac_sweep, mode="markers",
                    name=f"swept sine — {c.n_pts} points in the same {c.T_used:.0f} s",
                    marker=dict(color=sp.GRAY, size=sp.MK_BIG, symbol="x",
                                line=dict(width=1.5)))
    yr = sp._logy_range([c.frac_ms[m], c.frac_sweep], decades=4)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="σ(R)/|R|")
    fig.add_annotation(x=0.5, y=1.0, xref="paper", yref="paper", yanchor="bottom",
                       showarrow=False, font=dict(size=sp.SZ_ANNOT, color=sp.INK),
                       text=f"same band coverage by sweep ≈ {c.t_cover/60:.0f} min "
                            f"({c.t_cover/c.T:.0f}× the one multisine window)")
    fig.update_layout(title="Fractional response uncertainty — same twin, same noise, "
                            "equal wall-clock")
    return sp.style(fig, height=height)


def fom_table(c=None):
    rows = [
        ["Frequencies per measurement", "1 (dwell)", "all band bins at once"],
        ["Leakage", "windowed / settle each point", "leakage-free (periodic)"],
        ["Noise model", "assumed / averaged", "per-bin, from period-to-period variance"],
        ["Budget allocation", "uniform / manual", "CRB-optimal (Fisher-matched)"],
        ["Loop handling", "model out the servo", "FRF cancels it (reference-based)"],
    ]
    if c is not None:
        rows.append([f"Time for full-band coverage", f"≈{c.t_cover/60:.0f} min",
                     f"{c.T:.0f} s (one window)"])
    return sp.param_table(["figure of merit", "swept sine", "P&S multisine"], rows,
                          caption="Where the multisine method differs for DARM "
                                  "(representative; the efficiency is shown above, not asserted)")
