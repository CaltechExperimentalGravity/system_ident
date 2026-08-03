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
    DARMLoop, darm_design_asd, recover_response, fit_sensing, recover_actuation,
    multisine_response_sigma, swept_sine_response_sigma, sweep_time_to_match_coverage,
)
from system_ident.backends.darm_adapter import DARMBackend  # noqa: E402
from system_ident.excitation import multisine_from_psd  # noqa: E402
from system_ident.loop import SysIDLoop  # noqa: E402

# NPER=16 so the 3 s ramp leaves ~10 full periods → a genuine per-bin variance
# (P_eff≈9). NPER=8 would leave only 2 full periods → P_eff=1 → fabricated CRB bars.
NPERSEG, NPER = 4096, 16

# Representative injected drive against the REAL DARM floor (darm_design_asd, ~1.5e-20 m/√Hz
# mid-band). A Pcal multisine of ~5e-17 m RMS displacement is a realistic strong line and gives a
# sane per-record SNR (σ(R)/R ~ %). PX_REAL is its total power [m²]; A_TOT_REAL is the equivalent
# total displacement for the Fisher/Pareto sizing. (Exact per-line amplitudes are issue #3.)
PX_REAL = (5.0e-17) ** 2       # m² — multisine drive power against the real floor
A_TOT_REAL = 5.0e-17           # m — total injected displacement for cal-line sizing


def _grid(loop):
    fa = np.fft.rfftfreq(NPERSEG, 1 / loop.fs)
    band = (fa >= loop.fmin) & (fa <= loop.fmax)
    return fa, band, fa[band]


def _twin(seed=1):
    # Real Advanced-LIGO DARM displacement-noise floor (darm_design_asd: strain × 4 km, the
    # design-era bucket ~1.5e-20 m/√Hz mid-band). The lumped default() sensing/actuation is kept
    # for the intro campaigns; the physical floor makes the SNRs and σ's real, not representative.
    loop = DARMLoop.default()
    loop.noise_asd = darm_design_asd
    return loop


# ── campaigns ─────────────────────────────────────────────────────────────────
def pcal_audit(seed=1):
    loop = _twin()
    fa, band, freq = _grid(loop)
    Pxx = np.full_like(freq, PX_REAL / (freq[-1] - freq[0]))
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
    Pxx = np.full_like(freq, PX_REAL / (freq[-1] - freq[0]))
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
                                                 px_total=PX_REAL, seed=seed)
    # equal wall-clock sweep: in the SAME T, a 4-period dwell resolves only ~T*fs/(4*nperseg)
    # frequencies (= NPER//4 = 4 points here), vs the multisine's whole-band coverage.
    # 4 periods are the minimum for a genuine per-bin variance (P_eff≥3); 2 periods give
    # P_eff=2 which underflows to the estimator's 1e-9 floor and produces a fabricated σ.
    n_pts = NPER // 4
    pts = np.geomspace(loop.fmin, loop.fmax, n_pts)
    fp, ssweep, T_used = swept_sine_response_sigma(loop, pts, nperseg=NPERSEG,
                                                   dwell_periods=4, px_total=PX_REAL, seed=seed)
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


# ── hierarchical actuation: per-stage authority + inter-stage crossovers ────────
def _crossings(freq, mag_a, mag_b):
    """Frequencies where |A_a| = |A_b| (linear-interpolated sign changes of the log ratio)."""
    d = np.log(mag_a) - np.log(mag_b)
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    out = []
    for i in idx:
        f0, f1, y0, y1 = freq[i], freq[i + 1], d[i], d[i + 1]
        out.append(float(np.exp(np.interp(0.0, [y0, y1], [np.log(f0), np.log(f1)]))) if y1 != y0 else float(f0))
    return np.array(out)


def hierarchical_actuation(loop=None, *, fmin=0.1, fmax=100.0, npts=3000, height=560):
    """Per-stage DARM actuation |A_i(f)| for the hierarchical (nested-offload) loop, with the
    inter-stage crossovers marked.

    Each stage's DARM-referred authority is A_i = κ_i·D_i(f)·N_i(f): the M0-damped reduced-QUAD
    compliance N_i (M0→L3, PUM→L3, TST→L3) shaped by the nested-offload distribution filter
    D_M0=O_A·O_B, D_PUM=O_A, D_TST=1, with κ_i=1 (the offload runs in force units — the compliances
    carry the hierarchy). The strong, slow M0 owns low frequency and hands off up the chain: M0→PUM
    near F_PT≈0.5 Hz, PUM→TST near F_EP≈10 Hz. Those crossovers are exactly what per-stage
    calibration lines measure — the frequency where each stage's authority equals the one below.
    Above its handoff a stage's authority collapses (M0 has effectively none in the DARM band), so
    the y-range is clamped to ~10 decades: enough to show both crossovers and TST's high-band
    dominance without the M0 tail (→1e-22 by 1 kHz) dictating the scale. The 0.4–3 Hz peaks are the
    L1/L2/L3 forest, undamped by the M0-only local loop (faithful to the real plant)."""
    if loop is None:
        loop = DARMLoop.default_reduced(fmin=fmin, hierarchical=True)
    freq = np.geomspace(fmin, fmax, npts)
    stages = ["M0", "PUM", "TST"]
    labels = {"M0": "M0 (top — strong, slow)", "PUM": "PUM (mid)", "TST": "TST (test mass — fast)"}
    colors = {"M0": sp.SKY, "PUM": sp.GOLD, "TST": sp.GREEN}
    mag = {s: np.abs(loop.stage(s, freq)) for s in stages}
    yr = sp._logy_range(list(mag.values()), decades=10)

    fig = go.Figure()
    for s in stages:
        fig.add_scatter(x=freq, y=mag[s], mode="lines", name=labels[s],
                        line=dict(color=colors[s], width=2.6),
                        hovertemplate="%{x:.3f} Hz   |A| = %{y:.4g}<extra>" + s + "</extra>")

    # inter-stage crossovers: the design targets the cal lines measure
    for a, b, target, tag in [("M0", "PUM", 0.5, "F_PT"), ("PUM", "TST", 10.0, "F_EP")]:
        xs = _crossings(freq, mag[a], mag[b])
        if xs.size == 0:
            continue
        xc = float(xs[np.argmin(np.abs(xs - target))])   # the handoff nearest the design target
        yc = float(np.interp(np.log(xc), np.log(freq), mag[a]))
        fig.add_vline(x=xc, line=dict(color=sp.GRAY, width=1.4, dash="dash"))
        fig.add_scatter(x=[xc], y=[yc], mode="markers", showlegend=False,
                        marker=dict(color=sp.INK, size=sp.MK_BIG, symbol="circle-open",
                                    line=dict(width=2.5)),
                        hovertemplate=f"{a}/{b} crossover<br>%{{x:.3f}} Hz<extra></extra>")
        fig.add_annotation(x=np.log10(xc), y=(yr[1] if yr else np.log10(yc)), yanchor="top",
                           text=f"<b>{a}/{b} ≈ {xc:.2f} Hz</b><br>{tag} ≈ {target:g} Hz",
                           showarrow=False, yshift=-6, font=dict(size=sp.SZ_ANNOT, color=sp.INK),
                           bgcolor="rgba(255,255,255,0.75)")

    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="|A_i(f)|   [DARM disp / count]")
    fig.update_layout(title="Hierarchical DARM actuation — per-stage authority and the "
                            "crossovers the cal lines measure")
    return sp.style(fig, height=height)


# ── parameter-error convergence: σ on every cal parameter vs measurement time ───
def convergence_campaign(seed=3, periods=(16, 32, 64, 128)):
    """σ on each calibration parameter as the record lengthens, so a cal engineer can read the
    exposure needed for a target precision. Six parameters — sensing g_C / f_cc / τ (one Pcal
    multisine → weighted complex LS) and the three actuator strengths κ_UIM / κ_PUM / κ_TST
    (each stage vs the shared Pcal ruler) — recovered at P = 8…128 periods (T = P·nperseg/fs).
    Period-averaging makes every σ fall as 1/√T (the CRB scaling); the fit/period-variance σ IS
    the CRB estimate, so the points land on the 1/√T law with no free knob."""
    loop = _twin()
    fa, band, freq = _grid(loop)
    one_plus_G = 1.0 + loop.G(freq)
    Pxx = np.full_like(freq, PX_REAL / (freq[-1] - freq[0]))
    stages = ("UIM", "PUM", "TST")
    truth = {"g_C": loop.g_c, "f_cc": loop.f_cc, "τ": loop.tau,
             "κ_UIM": loop.stages["UIM"][1], "κ_PUM": loop.stages["PUM"][1],
             "κ_TST": loop.stages["TST"][1]}
    order = ["g_C", "f_cc", "τ", "κ_UIM", "κ_PUM", "κ_TST"]
    T = []
    frac = {p: [] for p in order}
    for P in periods:
        dur = NPERSEG * P / loop.fs
        T.append(dur)
        bp = DARMBackend(loop, {"PCAL_EXC": "PCAL"}, "DARM_ERR", seed=seed)
        xp = multisine_from_psd(Pxx, loop.fs, NPERSEG, P, freq, seed=np.random.default_rng(seed))
        bp.inject("PCAL_EXC", xp, loop.fs)
        sp_seg = bp.read(["PCAL_EXC", "DARM_ERR"], dur)
        Hp, Hp_err, _ = SysIDLoop._estimate_tf_periodic(sp_seg["PCAL_EXC"], sp_seg["DARM_ERR"],
                                                        loop.fs, NPERSEG, band, n_transient=1)
        p, s = fit_sensing(freq, Hp * one_plus_G, Hp_err * np.abs(one_plus_G),
                           p0=(0.8e6, 300.0, 50e-6))
        frac["g_C"].append(s["g_c"] / truth["g_C"])
        frac["f_cc"].append(s["f_cc"] / truth["f_cc"])
        frac["τ"].append(s["tau"] / truth["τ"])
        for name in stages:
            be = DARMBackend(loop, {"EXC": name}, "DARM_ERR", seed=seed + 1)
            xi = multisine_from_psd(Pxx, loop.fs, NPERSEG, P, freq, seed=np.random.default_rng(seed + 2))
            be.inject("EXC", xi, loop.fs)
            si = be.read(["EXC", "DARM_ERR"], dur)
            Hi, Hi_err, _ = SysIDLoop._estimate_tf_periodic(si["EXC"], si["DARM_ERR"],
                                                            loop.fs, NPERSEG, band, n_transient=1)
            N = loop.stages[name][0].eval(freq)
            comb = np.hypot(Hi_err / np.abs(Hi), Hp_err / np.abs(Hp)) * np.abs(Hi / Hp)
            _, ks = recover_actuation(freq, Hi, Hp, N, comb)
            frac[f"κ_{name}"].append(ks / truth[f"κ_{name}"])
    return SimpleNamespace(T=np.array(T), frac={k: np.array(v) for k, v in frac.items()},
                           order=order, periods=periods)


def convergence_fig(cv=None, *, height=560):
    """Fractional 1σ on each cal parameter vs record length, with the 1/√T CRB law."""
    if cv is None:
        cv = convergence_campaign()
    colors = {"g_C": sp.SKY, "f_cc": sp.GREEN, "τ": sp.INK,
              "κ_UIM": sp.GOLD, "κ_PUM": sp.ROSE, "κ_TST": "#7C5CBF"}
    fig = go.Figure()
    for p in cv.order:
        fig.add_scatter(x=cv.T, y=cv.frac[p], mode="lines+markers", name=p,
                        line=dict(color=colors[p], width=2.4),
                        marker=dict(color=colors[p], size=sp.MK_DATA),
                        hovertemplate=p + ": σ/val = %{y:.3g} @ %{x:.0f} s<extra></extra>")
    # 1/√T CRB reference, anchored at the geometric-mean level of the first column
    y0 = float(np.exp(np.mean([np.log(cv.frac[p][0]) for p in cv.order])))
    Tref = np.array([cv.T[0], cv.T[-1]])
    fig.add_scatter(x=Tref, y=y0 * np.sqrt(cv.T[0] / Tref), mode="lines",
                    name="1/√T  (CRB scaling)", line=dict(color=sp.GRAY, width=2.0, dash="dash"),
                    hoverinfo="skip")
    allv = np.concatenate([cv.frac[p] for p in cv.order])
    fig.update_xaxes(type="log", title_text="record length T = P·nperseg/f_s  [s]")
    fig.update_yaxes(type="log", title_text="fractional 1σ   σ(θ)/θ",
                     range=sp._logy_range([allv], decades=3))
    fig.update_layout(title="Parameter-error convergence — every cal parameter falls as 1/√T "
                            "onto the Cramér–Rao bound")
    return sp.style(fig, height=height)


# ── the calibration lines, at their real height in the DARM spectrum ────────────
import scipy.signal as _sig  # noqa: E402


def cal_line_spectrum(seed=5, *, P=256, snr_targets=(("M0", 0.45, 90.0),
                                                     ("PUM", 3.0, 140.0),
                                                     ("TST", 40.0, 260.0))):
    """A real calibrated-DARM displacement spectrum with the per-stage calibration lines injected,
    simulated (disturbance + readout noise), and referred back to metres.

    Each stage carries one line (M0 low, PUM mid, TST in-band); the drive on each is set so the
    line reaches a stated SNR over the record `T = P·nperseg/fs`, then the loop is SIMULATED and
    the DARM error is deconvolved to displacement `x = d_err·(1+G)/C`. The spectrum is a
    full-record periodogram (resolution `df = 1/T`), so a coherent line stands EXACTLY its SNR
    above the `√(disturbance² + (readout/|C|)²)` floor — the number a cal engineer reads straight
    off the plot — and the per-record strength precision is `σκ/κ ≈ 1/SNR`. Returns the measured
    ASD (thinned for display, line bins kept exact), the analytic floor, and the per-line table."""
    loop = DARMLoop.default_reduced(fmin=0.3, hierarchical=True)
    loop.noise_asd = darm_design_asd                        # real aLIGO DARM displacement floor
    fs, n = loop.fs, NPERSEG * P
    T = n / fs
    fgrid = np.fft.rfftfreq(n, 1.0 / fs)
    C_grid = loop.C(fgrid)
    one_plus_G = 1.0 + loop.G(fgrid)

    # real DARM displacement noise floor (aLIGO design bucket)
    ff = np.geomspace(loop.fmin, loop.fmax, 1400)
    floor = loop.displacement_noise_asd(ff)

    # Simulate ONLY the noise through the real loop TFs (disturbance C/(1+G), readout 1/(1+G) —
    # both analytic in C,G, no per-bin plant solve), deconvolve to displacement, then place each
    # coherent line at its exact recovered height A_i·c_i (deterministic: the loop is linear and
    # G,C are known, so this IS what driving the stage and deconvolving would yield). This avoids
    # evaluating the per-bin damped-quad stage over the megapoint grid — seconds, not minutes.
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    derr_noise = loop.simulate({}, n, rng)                          # [ct]: disturbance + readout
    Xf = np.fft.rfft(derr_noise) * np.where(np.abs(C_grid) > 0, one_plus_G / C_grid, 0.0)
    x = np.fft.irfft(Xf, n)                                          # calibrated displacement [m]
    rows, line_bins = [], {}
    for name, f_want, snr in snr_targets:
        k = int(round(f_want * n / fs))
        f_line = k * fs / n
        floor_l = float(np.interp(f_line, ff, floor))               # m/√Hz at the line
        height = snr * floor_l / np.sqrt(T)                          # line displacement [m rms]
        x += np.sqrt(2.0) * height * np.sin(2 * np.pi * f_line * t)  # coherent cal line
        rows.append((name, f_line, height, snr, 1.0 / snr))
        line_bins[name] = k

    fpsd, Pxx = _sig.periodogram(x, fs=fs, detrend=False)           # df = 1/T → line/floor = SNR
    asd = np.sqrt(Pxx)
    line_asd = {nm: (fpsd[k], float(asd[k])) for nm, k in line_bins.items()}
    # thin the dense periodogram for display (keep the exact line bins)
    band = np.where((fpsd >= loop.fmin) & (fpsd <= loop.fmax))[0]
    keep = np.unique(np.geomspace(band[0], band[-1], 2500).astype(int))
    keep = np.union1d(keep, list(line_bins.values()))
    return SimpleNamespace(loop=loop, fpsd=fpsd[keep], asd=asd[keep], ff=ff, floor=floor,
                           lines=rows, line_asd=line_asd, T=T, P=P)


def cal_line_spectrum_fig(cs=None, *, height=560):
    """Calibrated DARM displacement ASD (periodogram, df=1/T) with the injected calibration lines
    standing exactly their SNR above the disturbance + readout floor."""
    if cs is None:
        cs = cal_line_spectrum()
    colors = {"M0": sp.SKY, "PUM": sp.ROSE, "TST": sp.GOLD}
    fig = go.Figure()
    fig.add_scatter(x=cs.fpsd, y=cs.asd, mode="lines",
                    name=f"calibrated DARM + cal lines  (T={cs.T:.0f} s, df=1/T)",
                    line=dict(color=sp.GRAY, width=1.0), opacity=0.7,
                    hovertemplate="%{x:.2f} Hz   %{y:.3g} m/√Hz<extra></extra>")
    fig.add_scatter(x=cs.ff, y=cs.floor, mode="lines",
                    name="noise floor  √(disturbance² + (readout/|C|)²)",
                    line=dict(color=sp.INK, width=2.4, dash="dot"), hoverinfo="skip")
    for name, f_line, h_disp, snr, prec in cs.lines:
        yl = cs.line_asd[name][1]
        fig.add_scatter(x=[f_line, f_line], y=[float(np.interp(f_line, cs.ff, cs.floor)), yl],
                        mode="lines", showlegend=False,
                        line=dict(color=colors[name], width=2.0))
        fig.add_scatter(x=[f_line], y=[yl], mode="markers", showlegend=False,
                        marker=dict(color=colors[name], size=sp.MK_BIG, symbol="diamond",
                                    line=dict(color=sp.INK, width=1.2)),
                        hovertemplate=f"{name} line<br>%{{x:.2f}} Hz<extra></extra>")
        fig.add_annotation(x=np.log10(f_line), y=np.log10(yl), yanchor="bottom", yshift=7,
                           text=f"<b>{name}</b> {f_line:.2f} Hz<br>SNR≈{snr:.0f} · σκ/κ≈{prec*100:.1f}%",
                           showarrow=False, font=dict(size=sp.SZ_ANNOT, color=sp.INK),
                           bgcolor="rgba(255,255,255,0.82)")
    yr = sp._logy_range([cs.asd, cs.floor], decades=6)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="DARM displacement ASD  [m/√Hz]")
    fig.update_layout(title=f"Calibration lines in the DARM spectrum — each stands its SNR above "
                            f"the floor over a {cs.T:.0f} s record")
    return sp.style(fig, height=height)


def cal_line_table(cs=None):
    if cs is None:
        cs = cal_line_spectrum()
    rows = [[name, f"{f_line:.2f}", f"{h_disp:.3g}", f"{snr:.0f}", f"{prec*100:.2f}"]
            for name, f_line, h_disp, snr, prec in cs.lines]
    return sp.param_table(["stage line", "f [Hz]", "displacement [m rms]",
                           f"SNR ({cs.T:.0f} s)", "σκ/κ [%]"], rows,
                          caption="Per-stage calibration lines: displacement amplitude, in-record "
                                  "SNR, and the per-record strength precision (≈ 1/SNR). Longer "
                                  "integration scales SNR as √T.")


# ── Fisher-optimal cal-line sizing: 0.1% on every TDCF in < 5 minutes ───────────
from system_ident import darm_callines as _cl  # noqa: E402


def cal_sizing(seed=0, t_pns_target=90.0):
    """Size the calibration lines (Pcal + M0/PUM/ESD) so every TDCF reaches 0.1% fractional 1σ,
    and compare the P&S-optimal placement to the LIGO O3/O4 line positions at equal total drive.

    The relative result (per-parameter time-to-0.1%, and the P&S/O3/O4 ratio) is scale-invariant;
    the absolute total drive is representative and is set here so the P&S-optimal scheme reaches
    the target in ``t_pns_target`` seconds (matching the papers' actual drive amplitudes is a later
    phase). σ ∝ 1/(drive·√T), so rescaling the drive just rescales every time uniformly."""
    loop = _cl.default_cal_loop(delta_deg=5.0)
    pns = _cl.size_lines_for_target(loop, A_tot=A_TOT_REAL, target=1e-3, T_ref=60.0, seed=seed)
    o3 = _cl.reference_scheme(loop, _cl.O3_LINES, A_tot=A_TOT_REAL)
    o4 = _cl.reference_scheme(loop, _cl.O4_LINES, A_tot=A_TOT_REAL)
    scale = np.sqrt(pns["t_req_max"] / t_pns_target)     # A_tot so P&S hits target in t_pns_target
    for r in (pns, o3, o4):                               # rescale times to the common drive
        r["t_req"] = {k: v / scale ** 2 for k, v in r["t_req"].items()}
        r["t_req_max"] = r["t_req_max"] / scale ** 2
    return SimpleNamespace(loop=loop, pns=pns, o3=o3, o4=o4, scale=scale,
                           t_pns_target=t_pns_target)


def _sigma_curves(loop, res, scale, times):
    s = _cl.sigma_vs_time(loop, res["lines"], times)     # σ(t) at A_tot=1
    return {k: v / scale for k, v in s.items()}          # rescale to the common drive


def convergence_to_target_fig(cs=None, *, height=560):
    """Per-TDCF fractional σ(t) under the P&S-optimal roster — every parameter crossing the 0.1%
    target before the 5-minute mark."""
    if cs is None:
        cs = cal_sizing()
    times = np.geomspace(1.0, 600.0, 240)
    curves = _sigma_curves(cs.loop, cs.pns, cs.scale, times)
    colors = {"kappa_C": sp.SKY, "f_cc": sp.GREEN, "delta": sp.ROSE, "tau": sp.INK,
              "kappa_M0": sp.GOLD, "kappa_PUM": "#7C5CBF", "kappa_TST": "#0E7C7B"}
    labels = {"kappa_C": "κ_C", "f_cc": "f_cc", "delta": "δ (SRC)", "tau": "τ",
              "kappa_M0": "κ_M0", "kappa_PUM": "κ_PUM", "kappa_TST": "κ_ESD"}
    fig = go.Figure()
    for k in _cl.TDCF_PARAMS:
        fig.add_scatter(x=times, y=curves[k], mode="lines", name=labels[k],
                        line=dict(color=colors[k], width=2.4),
                        hovertemplate=labels[k] + ": %{y:.2e} @ %{x:.0f} s<extra></extra>")
    fig.add_hline(y=1e-3, line=dict(color=sp.RED, width=2, dash="dash"))
    fig.add_vline(x=300.0, line=dict(color=sp.GRAY, width=2, dash="dot"))
    fig.add_annotation(x=np.log10(1.2), y=np.log10(1e-3), yanchor="bottom", xanchor="left",
                       text="0.1% target", showarrow=False, font=dict(size=sp.SZ_ANNOT, color=sp.RED))
    fig.add_annotation(x=np.log10(300.0), y=0.02, yref="paper", xanchor="right",
                       text="5 min ", showarrow=False, font=dict(size=sp.SZ_ANNOT, color=sp.INK))
    allv = np.concatenate([curves[k] for k in curves])
    fig.update_xaxes(type="log", title_text="integration time  T  [s]")
    fig.update_yaxes(type="log", title_text="fractional 1σ   σ(θ)/θ",
                     range=sp._logy_range([allv], decades=4))
    fig.update_layout(title=f"P&S-optimal cal lines — every TDCF to 0.1% in {cs.pns['t_req_max']:.0f} s "
                            f"(< 5 min)")
    return sp.style(fig, height=height)


def scheme_bars_fig(cs=None, *, height=520):
    """Per-TDCF time-to-0.1% for the P&S-optimal, O3, and O4 line schemes at equal total drive."""
    if cs is None:
        cs = cal_sizing()
    labels = {"kappa_C": "κ_C", "f_cc": "f_cc", "delta": "δ", "tau": "τ",
              "kappa_M0": "κ_M0", "kappa_PUM": "κ_PUM", "kappa_TST": "κ_ESD"}
    x = [labels[k] for k in _cl.TDCF_PARAMS]
    fig = go.Figure()
    for name, res, color in [("P&S-optimal", cs.pns, sp.SKY), ("LIGO O3", cs.o3, sp.GOLD),
                             ("LIGO O4", cs.o4, sp.ROSE)]:
        fig.add_bar(x=x, y=[res["t_req"][k] for k in _cl.TDCF_PARAMS], name=name,
                    marker_color=color,
                    hovertemplate=name + ": %{y:.1f} s<extra>%{x}</extra>")
    fig.add_hline(y=300.0, line=dict(color=sp.GRAY, width=2, dash="dot"))
    fig.add_annotation(x=0.5, xref="paper", y=np.log10(300.0), yanchor="bottom", showarrow=False,
                       text="5 min", font=dict(size=sp.SZ_ANNOT, color=sp.INK))
    fig.update_yaxes(type="log", title_text="time to 0.1%  [s]")
    fig.update_layout(barmode="group",
                      title="Time to 0.1% per parameter — P&S-optimal vs LIGO O3/O4 (equal drive)")
    return sp.style(fig, height=height)


def sizing_table(cs=None):
    if cs is None:
        cs = cal_sizing()
    rows = [["P&S-optimal", f"{cs.pns['t_req_max']:.0f}", cs.pns["binding"], "1.0×"],
            ["LIGO O3 lines", f"{cs.o3['t_req_max']:.0f}", cs.o3["binding"],
             f"{cs.o3['t_req_max']/cs.pns['t_req_max']:.1f}×"],
            ["LIGO O4 lines", f"{cs.o4['t_req_max']:.0f}", cs.o4["binding"],
             f"{cs.o4['t_req_max']/cs.pns['t_req_max']:.1f}×"]]
    return sp.param_table(["line scheme", "time to 0.1% on all 7 [s]", "binding param",
                           "vs P&S-optimal"], rows,
                          caption="Time for every TDCF to reach 0.1% at equal total drive. The "
                                  "advantage is concentrated in δ and τ (which LIGO monitors but "
                                  "does not correct to 0.1%); for the κ's the schemes are "
                                  "comparable. Absolute times are representative (drive-match "
                                  "deferred); the ratio is scale-invariant.")


def sized_lines_fig(cs=None, *, height=520):
    """Where the sized calibration lines sit over the DARM noise floor: the three actuator lines in
    their hierarchy bands (M0 low, PUM mid, TST ~tens of Hz) and the four Fisher-placed Pcal lines,
    with the O3/O4 line frequencies shown for comparison."""
    if cs is None:
        cs = cal_sizing()
    loop = cs.loop
    ff = np.geomspace(0.3, 1500.0, 1400)
    floor = _cl.floor_asd(loop, ff)
    labels = {"kappa_C": "κ_C", "f_cc": "f_cc", "delta": "δ", "tau": "τ",
              "kappa_M0": "κ_M0", "kappa_PUM": "κ_PUM", "kappa_TST": "κ_ESD"}
    fig = go.Figure()
    fig.add_scatter(x=ff, y=floor, mode="lines", name="DARM floor",
                    line=dict(color=sp.INK, width=2.2, dash="dot"), hoverinfo="skip")
    # O3/O4 line frequencies as faint reference ticks
    for res, color, name in [(cs.o3, sp.GOLD, "O3 lines"), (cs.o4, sp.ROSE, "O4 lines")]:
        fx = [ln.freq for ln in res["lines"]]
        fig.add_scatter(x=fx, y=[float(np.interp(f, ff, floor)) * 0.5 for f in fx], mode="markers",
                        name=name, marker=dict(color=color, size=8, symbol="line-ns-open"),
                        hovertemplate=name + ": %{x:.1f} Hz<extra></extra>")
    # P&S-optimal lines as labelled stems
    for ln in cs.pns["lines"]:
        y0 = float(np.interp(ln.freq, ff, floor))
        yl = y0 * 6.0
        c = sp.SKY if ln.kind == "PCAL" else sp.GREEN
        fig.add_scatter(x=[ln.freq, ln.freq], y=[y0, yl], mode="lines", showlegend=False,
                        line=dict(color=c, width=2.4))
        fig.add_scatter(x=[ln.freq], y=[yl], mode="markers", showlegend=False,
                        marker=dict(color=c, size=sp.MK_DATA + 2,
                                    symbol="diamond" if ln.kind == "PCAL" else "square",
                                    line=dict(color=sp.INK, width=1)),
                        hovertemplate=f"{ln.kind} → {labels.get(ln.target, ln.target)}<br>%{{x:.2f}} Hz<extra></extra>")
        fig.add_annotation(x=np.log10(ln.freq), y=np.log10(yl), yanchor="bottom", yshift=4,
                           text=labels.get(ln.target, ln.target), showarrow=False,
                           font=dict(size=sp.SZ_ANNOT, color=c))
    # legend proxies for the two P&S line kinds
    fig.add_scatter(x=[None], y=[None], mode="markers", name="P&S Pcal (sensing)",
                    marker=dict(color=sp.SKY, size=sp.MK_DATA + 2, symbol="diamond"))
    fig.add_scatter(x=[None], y=[None], mode="markers", name="P&S actuator (hierarchy band)",
                    marker=dict(color=sp.GREEN, size=sp.MK_DATA + 2, symbol="square"))
    yr = sp._logy_range([floor], decades=4)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="DARM displacement ASD  [m/√Hz]")
    fig.update_layout(title="Where the sized lines sit — actuator lines in their hierarchy bands, "
                            "Pcal lines Fisher-placed")
    return sp.style(fig, height=height)


def response_budget_fig(sz=None, *, height=640):
    """The cal-line statistical uncertainty propagated into the detector response δR/R(f) — magnitude
    [%] and phase [deg] — laid against the published Advanced-LIGO O3 systematic-error budget."""
    if sz is None:
        sz = cal_sizing()
    loop = sz.loop
    ls = sz.pns["lineset"]
    amps = np.array([ln.amp for ln in sz.pns["lines"]], float)
    f = np.geomspace(10.0, 2000.0, 700)
    # design point (all TDCFs at 0.1%) and a 5-minute integration, at A_tot=1
    worst = max(_cl.sigma(ls, amps, 60.0).values())
    T_design = 60.0 * (worst / 1e-3) ** 2
    curves = [(T_design, sp.SKY, f"P&S lines, all TDCFs at 0.1% ({T_design:.0f} s)"),
              (300.0, sp.GREEN, "P&S lines, 5 min integration")]
    B = _cl.O3_BUDGET
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("magnitude  |δR/R|  [%]", "phase  ∠(δR/R)  [deg]"))
    for T, color, name in curves:
        mag, ph = _cl.response_budget(loop, ls, amps, T, f)
        fig.add_scatter(x=f, y=mag, mode="lines", name=name, line=dict(color=color, width=2.6),
                        legendgroup=name, hovertemplate="%{x:.0f} Hz  %{y:.3f}%<extra></extra>",
                        row=1, col=1)
        fig.add_scatter(x=f, y=ph, mode="lines", name=name, line=dict(color=color, width=2.6),
                        legendgroup=name, showlegend=False,
                        hovertemplate="%{x:.0f} Hz  %{y:.3f}°<extra></extra>", row=2, col=1)
    # O3 reference budget levels
    for row, tot, syst, unit in [(1, B["total_mag_pct"], B["syst_mag_pct"], "%"),
                                 (2, B["total_phase_deg"], B["syst_phase_deg"], "°")]:
        fig.add_hline(y=tot, line=dict(color=sp.RED, width=2, dash="dash"), row=row, col=1)
        fig.add_hline(y=syst, line=dict(color=sp.GOLD, width=2, dash="dot"), row=row, col=1)
        fig.add_annotation(x=np.log10(12), y=np.log10(tot), yanchor="bottom", xanchor="left",
                           text=f"O3 total budget ({tot:g}{unit}, 68%)", showarrow=False,
                           font=dict(size=sp.SZ_ANNOT, color=sp.RED), row=row, col=1)
        fig.add_annotation(x=np.log10(12), y=np.log10(syst), yanchor="bottom", xanchor="left",
                           text=f"O3 systematic floor ({syst:g}{unit})", showarrow=False,
                           font=dict(size=sp.SZ_ANNOT, color=sp.GOLD), row=row, col=1)
    fig.update_xaxes(type="log", row=2, col=1, title_text="frequency [Hz]")
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_layout(title="Response-error budget — cal-line statistics vs the O3 systematic budget")
    return sp.style(fig, height=height)


# ── the amplitude↔time Pareto: reach O3/O4 random levels gently & fast ───────────
def pareto_campaign(seed=0):
    """Response-optimal P&S line design vs the O3/O4 fixed-line and naive-broadband baselines, as
    the injected-energy cost K = amplitude²·time to reach each random-error target level. K is
    scheme-characteristic (σ_R ∝ 1/√(A²T)); the iso-precision contour is A(T)=√(K/T)."""
    loop = _cl.default_cal_loop(delta_deg=5.0)
    pns = _cl.size_lines_for_response(loop, A_tot=A_TOT_REAL, T_ref=60.0, seed=seed)
    o3 = _cl.reference_scheme(loop, _cl.O3_LINES, A_tot=A_TOT_REAL)
    o4 = _cl.reference_scheme(loop, _cl.O4_LINES, A_tot=A_TOT_REAL)
    nb_ls, nb_amps = _cl.naive_broadband(loop, A_tot=A_TOT_REAL)
    schemes = {"P&S response-optimal": (pns["lineset"], pns["amps"]),
               "O3/O4 fixed-line": (o3["lineset"], np.array([l.amp for l in o3["lines"]])),
               "naive broadband": (nb_ls, nb_amps)}
    K = {s: {t: _cl.pareto_cost(loop, ls, a, _cl.rho_of_target(*lvl))
             for t, lvl in _cl.TARGET_LEVELS.items()} for s, (ls, a) in schemes.items()}
    # fiducial floor (mid-band, ~best sensitivity) for the amplitude-axis normalisation
    floor_ref = float(loop.displacement_noise_asd(np.array([200.0]))[0])
    return SimpleNamespace(loop=loop, K=K, floor=floor_ref,
                           targets=_cl.TARGET_LEVELS, pns=pns)


def pareto_fig(pc=None, *, height=600):
    """The amplitude↔time design plane: iso-precision contours A(T)=√(K/T) (amplitude referenced to
    the DARM floor) for the P&S response-optimal scheme at each random-error target, with the
    O3/O4-fixed and naive contours at the O3 level to show how much gentler/faster P&S is."""
    if pc is None:
        pc = pareto_campaign()
    T = np.geomspace(1.0, 3.0e4, 200)
    kp = pc.K["P&S response-optimal"]
    tcolor = {"O3 random": sp.SKY, "O4-class (prov.)": sp.GREEN, "0.1% stretch": sp.ROSE}
    fig = go.Figure()
    # shaded "gentleness gap": between the P&S and fixed-line contours at the O3 level
    A_pns_o3 = np.sqrt(kp["O3 random"] / T) / pc.floor
    A_fix_o3 = np.sqrt(pc.K["O3/O4 fixed-line"]["O3 random"] / T) / pc.floor
    fig.add_scatter(x=np.r_[T, T[::-1]], y=np.r_[A_pns_o3, A_fix_o3[::-1]], fill="toself",
                    fillcolor=sp._fade(sp.SKY, 0.10), line=dict(width=0), hoverinfo="skip",
                    showlegend=False)
    # P&S contours for every target (the design slider)
    for t, c in tcolor.items():
        fig.add_scatter(x=T, y=np.sqrt(kp[t] / T) / pc.floor, mode="lines",
                        name=f"P&S response-optimal — {t}", line=dict(color=c, width=2.8),
                        hovertemplate=t + ": %{x:.0f} s, A/floor=%{y:.3g}<extra></extra>")
    # baselines at the O3 level (the ×energy factor is the same for every target contour)
    for scheme, color, dash in [("O3/O4 fixed-line", sp.INK, "dash"),
                                ("naive broadband", "#7C5CBF", "dot")]:
        fac = pc.K[scheme]["O3 random"] / kp["O3 random"]
        fig.add_scatter(x=T, y=np.sqrt(pc.K[scheme]["O3 random"] / T) / pc.floor, mode="lines",
                        name=f"{scheme} @ O3 level  (×{fac:.0f} energy)",
                        line=dict(color=color, width=2.0, dash=dash), hoverinfo="skip")
    fac_fix = pc.K["O3/O4 fixed-line"]["O3 random"] / kp["O3 random"]
    fig.add_annotation(x=np.log10(30), y=np.log10(np.sqrt(kp["O3 random"] / 30) / pc.floor),
                       yanchor="top", yshift=-6,
                       text=f"P&S reaches O3 level with ×{fac_fix:.0f} less energy<br>"
                            f"(×{np.sqrt(fac_fix):.1f} amplitude, or ×{fac_fix:.0f} time)",
                       showarrow=False, font=dict(size=sp.SZ_ANNOT, color=sp.SKY),
                       bgcolor="rgba(255,255,255,0.8)")
    fig.update_xaxes(type="log", title_text="integration time  T  [s]")
    fig.update_yaxes(type="log", title_text="injected amplitude / noise floor  [√Hz]")
    fig.update_layout(title="Amplitude↔time Pareto — slide along a contour to trade drive for time")
    return sp.style(fig, height=height, legend="v")


def pareto_table(pc=None):
    if pc is None:
        pc = pareto_campaign()
    kp = pc.K["P&S response-optimal"]
    rows = []
    for t in pc.targets:
        fac_fix = kp and pc.K["O3/O4 fixed-line"][t] / kp[t]
        fac_nb = pc.K["naive broadband"][t] / kp[t]
        rows.append([t, f"{pc.targets[t][0]:.2g}% / {pc.targets[t][1]:.2g}°",
                     f"{fac_fix:.1f}×", f"{fac_nb:.1f}×",
                     f"{np.sqrt(fac_fix):.1f}× / {fac_fix:.0f}×"])
    return sp.param_table(["random-error target", "|δR/R| level",
                           "energy vs fixed-line", "energy vs naive",
                           "→ less amplitude / less time (vs fixed)"], rows,
                          caption="Injected energy A²·T to reach each random-error level: P&S "
                                  "response-optimal vs the O3/O4 fixed-line and naive-broadband "
                                  "schemes. The energy saving converts to √(factor)× less amplitude "
                                  "(at equal time) or factor× less time (at equal amplitude). Ratios "
                                  "are scale-invariant; absolute amplitude awaits real line heights "
                                  "(issue #3).")
