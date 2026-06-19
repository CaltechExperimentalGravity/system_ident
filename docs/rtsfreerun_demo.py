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

import scipy.signal as _sig  # noqa: E402

import sysid_plots as sp  # noqa: E402
from sysid_campaign import run_siso_passes  # noqa: E402
from system_ident.backends import rtsfreerun_oracle as orc  # noqa: E402
from system_ident.backends.rtsfreerun_adapter import RTSfreerunBackend  # noqa: E402
from system_ident.excitation import multisine_from_psd  # noqa: E402
from system_ident.design.pintelon import optimal_excitation  # noqa: E402
from system_ident.loop import SysIDLoop  # noqa: E402

CONFIG = _ROOT / "src" / "system_ident" / "configs" / "rtsfreerun_hsts.yml"


def tukey_multisine(Pxx, fs, nperseg, n_periods, freq, *, ramp_s=3.0, seed=0):
    """A periodic multisine, tiled ``n_periods`` times, under a **Tukey amplitude
    envelope** that ramps the drive **up over ``ramp_s`` seconds and down over ``ramp_s``
    seconds** — the realistic control-room injection (you ramp an actuator on and off,
    you don't slam a suspension). The flat middle periods stay a clean integer-period
    multisine, so the leakage-free FRF is taken there. Returns ``(drive, envelope)``.
    """
    base = multisine_from_psd(Pxx, fs, nperseg, n_periods, freq,
                              seed=np.random.default_rng(seed))
    n = len(base)
    alpha = min(2.0 * ramp_s * fs / n, 1.0)          # ramp_s of cosine taper at each end
    env = _sig.windows.tukey(n, alpha=alpha)
    return base * env, env


def _steady_frf(seg, xch, rsp, fs, nperseg, band, *, n_lead, n_trail):
    """FRF + coherence from the flat (un-tapered) periods of a windowed record."""
    x, y = seg[xch], seg[rsp]
    a, b = n_lead * nperseg, len(x) - n_trail * nperseg
    H, H_err, coh = SysIDLoop._estimate_tf_periodic(x[a:b], y[a:b], fs, nperseg, band,
                                                    n_transient=0)
    return H, H_err, coh


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


def a2_audit(*, seed: int = 1, n_periods: int = 8, ramp_s: float = 3.0,
             n_lead: int = 2, n_trail: int = 1) -> SimpleNamespace:
    """One raw, auditable open-loop measurement on the compiled x1hsts plant: inject a
    **Tukey-windowed** optimal multisine under the twin's real seismic+readout noise and
    capture the actual injected drive, the after-actuator drive monitor, and the raw
    sensor readout — the time series you would watch in the control room — then form the
    leakage-free FRF (with coherence) from the steady periods and ML-fit it.
    """
    import yaml
    import x1hsts

    raw = yaml.safe_load(open(CONFIG))
    scen = orc.load_scenario(raw["rtsfreerun"]["scenario"])
    oracle = orc.analytic_plant(scen)
    fs = float(raw["measurement"]["fs"])
    nper = int(round(raw["measurement"]["segment_duration"] * fs))
    fa = np.fft.rfftfreq(nper, 1 / fs)
    band = (fa >= raw["measurement"]["freq_min"]) & (fa <= raw["measurement"]["freq_max"])
    freq = fa[band]
    exc, xmon, rb = (raw["channels"]["excitation"]["POS"], raw["channels"]["drive"]["POS"],
                     raw["channels"]["readback"]["POS"])

    mdl = x1hsts.x1hsts()
    orc.apply_scenario_init(mdl, scen)
    mdl.fm_clear_history(*sorted({op["fm"] for op in scen.get("init", []) if "fm" in op}))

    # A broadband (flat in-band) multisine for the raw audit — it illuminates *every*
    # line, so the readout shows the full plant and the noise floor and the coherence is
    # interpretable across the whole band. (The point-optimal ASD, which concentrates the
    # budget at the modes, is shown alongside in the excitation-design panel.) Tukey-ramped;
    # warmup_s=0 — the ramp-up is the settle, so the backend injects the drive verbatim.
    px_total = float(raw["measurement"]["px_total"])
    Pxx_flat = np.full_like(freq, px_total / (freq[-1] - freq[0]))
    Pxx_opt = optimal_excitation(freq, oracle, np.ones_like(freq), px_total, n_iter=6)
    drive, env = tukey_multisine(Pxx_flat, fs, nper, n_periods, freq, ramp_s=ramp_s)
    be = RTSfreerunBackend(mdl=mdl, exc_channels={exc: "POS"}, readback_channels={rb: "POS"},
                           noise=raw["rtsfreerun"]["noise"], fs=fs, warmup_s=0.0,
                           ramp_s=0.0, seed=seed)   # the audit ramps the drive itself
    be.inject(exc, drive, fs)
    seg = be.read([exc, xmon, rb], len(drive) / fs)
    H, H_err, coh = _steady_frf(seg, xmon, rb, fs, nper, band, n_lead=n_lead, n_trail=n_trail)
    from system_ident.estimators.gml import GMLEstimator
    fit = GMLEstimator().fit(freq, H, H_err, orc.prior_from_scenario(scen, perturb=0.08,
                                                                     rng=np.random.default_rng(7)))
    t = np.arange(len(drive)) / fs
    ff = np.geomspace(raw["measurement"]["freq_min"], raw["measurement"]["freq_max"], 400)
    excited = np.isfinite(H_err)
    return SimpleNamespace(t=t, fs=fs, drive=seg[exc], drivemon=seg[xmon], response=seg[rb],
                           env=env, freq=freq, band=band, H=H, H_err=H_err, coh=coh,
                           excited=excited, oracle=oracle, fit=fit, ff=ff,
                           asd_flat=np.sqrt(Pxx_flat), asd_opt=np.sqrt(Pxx_opt),
                           limit=float(raw["safety"]["actuator_sat"]))


# ── A3 + A4 — the real closed-loop 6-DOF composite ────────────────────────────
def a34(*, n_passes: int = 2) -> SimpleNamespace:
    import hsts6dof_loop as h6

    m = h6.HSTS6DOF()
    fs, nper, nseg = 256.0, 4096, 6
    fa = np.fft.rfftfreq(nper, 1 / fs)
    band = (fa >= 0.3) & (fa <= 8.0)
    freq = fa[band]
    kw = dict(fs=fs, nperseg=nper, n_periods=nseg, band=band, freq=freq,
              n_passes=n_passes, warmup_s=32.0, seed=0)
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
    fs, nper, nseg = 256.0, 4096, 6
    fa = np.fft.rfftfreq(nper, 1 / fs)
    band = (fa >= 0.3) & (fa <= 8.0)
    freq = fa[band]
    hist = d.model.parametric_recovery(dof, fs=fs, nperseg=nper, n_periods=nseg, band=band,
                                       freq=freq, n_passes=n_passes, warmup_s=32.0)
    return SimpleNamespace(dof=dof, ff=np.geomspace(0.3, 8, 400),
                           oracle=_SSDiag(d.model, dof), oracle_modes=d.model.oracle_prior(dof),
                           fit=hist[-1]["model"], freq=freq, H_meas=hist[-1]["H_acc"],
                           fracs=[h["frac"] for h in hist])


def a3_audit(d, dof="L", *, n_periods=6, ramp_s=3.0, n_lead=1, n_trail=1, seed=0):
    """One raw, auditable *closed-loop* measurement on the 6-DOF composite: with all six
    real dampers engaged, inject a Tukey-windowed broadband multisine on one DoF and
    capture the injected drive, the **damper feedback** it provokes, the **reconstructed
    plant input** (drive − feedback, the ``"+-"`` ``COIL_DRV_SUM`` node), and the raw
    sensor readout. Forms the reference-based FRF (vs the plant input) from the steady
    periods — the open-loop plant recovered through the live loop.
    """
    m = d.model
    fs, nper = 256.0, 4096
    fa = np.fft.rfftfreq(nper, 1 / fs)
    band = (fa >= 0.3) & (fa <= 8.0)
    freq = fa[band]
    j = m.dofs.index(dof)
    px_total = 1.0e7
    Pxx_flat = np.full_like(freq, px_total / (freq[-1] - freq[0]))
    drive, env = tukey_multisine(Pxx_flat, fs, nper, n_periods, freq, ramp_s=ramp_s, seed=seed)
    be = m.backend(dof, fs=fs, warmup_s=0.0, seed=seed, closed=True, ramp_s=0.0)
    be.inject(m.exc(dof), drive, fs)
    seg = be.read([m.exc(dof), m.plant_in(dof), m.damp_out(dof), m.readout(dof)], len(drive) / fs)
    H, H_err, coh = _steady_frf(seg, m.plant_in(dof), m.readout(dof), fs, nper, band,
                                n_lead=n_lead, n_trail=n_trail)
    t = np.arange(len(drive)) / fs
    return SimpleNamespace(dof=dof, t=t, fs=fs, drive=seg[m.exc(dof)],
                           feedback=seg[m.damp_out(dof)], plant_in=seg[m.plant_in(dof)],
                           response=seg[m.readout(dof)], env=env, freq=freq, band=band,
                           H=H, H_err=H_err, coh=coh, excited=np.isfinite(H_err),
                           oracle=_SSDiag(m, dof), ff=np.geomspace(0.3, 8, 400))


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


def tensor_grid(d, *, height=980, decades=6):
    """6×6 |FRF| grid: analytic oracle (line) vs open-loop recovered (A4, points),
    with the closed-loop recovery (A3) overlaid in green on the diagonal.

    All 36 panels share one fixed ``decades``-wide log-magnitude range with sparse
    (2-decade) power-format ticks shown only on the outer panels — otherwise plotly
    auto-ranges each panel over ~10 decades and stacks an unreadable wall of ticks.
    """
    dofs, freq, G = d.dofs, d.freq, d.G
    n = len(dofs)
    mags = np.abs(G).ravel()
    yhi = float(np.ceil(np.log10(mags[mags > 0].max())))      # top decade (diagonal peaks)
    ylo = yhi - decades
    xt = [0.3, 1.0, 3.0]
    fig = make_subplots(rows=n, cols=n, shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.008, vertical_spacing=0.014,
                        column_titles=[f"drive {x}" for x in dofs],
                        row_titles=[f"sense {x}" for x in dofs])
    for i in range(n):
        for j in range(n):
            r, c = i + 1, j + 1
            first = (i == 0 and j == 0)
            fig.add_scatter(x=freq, y=np.abs(G[:, i, j]), mode="lines", row=r, col=c,
                            line=dict(color="black", width=1.3), name="oracle",
                            legendgroup="o", showlegend=first)
            fig.add_scatter(x=freq, y=np.abs(d.H_open[i, j]), mode="markers", row=r, col=c,
                            marker=dict(color=sp.ROSE, size=2.6), name="A4 open-loop",
                            legendgroup="a4", showlegend=first)
            if i == j:
                fig.add_scatter(x=freq, y=np.abs(d.H_closed[i, j]), mode="markers", row=r, col=c,
                                marker=dict(color=sp.GREEN, size=3.0, symbol="x"),
                                name="A3 closed-loop", legendgroup="a3", showlegend=first)
            fig.update_xaxes(type="log", row=r, col=c, tickvals=xt,
                             showticklabels=(i == n - 1), tickfont=dict(size=9),
                             tickangle=0)
            fig.update_yaxes(type="log", row=r, col=c, range=[ylo, yhi], dtick=2,
                             exponentformat="power", showticklabels=(j == 0),
                             tickfont=dict(size=9))
    fig.update_xaxes(title_text="frequency [Hz]", row=n, col=1)
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


# ── audit figures (the control-room view) ─────────────────────────────────────
def exc_design_fig(a, *, height=560):
    """Plant magnitude with the broadband audit drive vs the point-optimal ASD."""
    mag = np.abs(a.oracle.eval(a.freq))
    ann = [(f0, f"{f0:.2f} Hz") for f0, _ in orc.plant_modes(a.oracle)]
    return sp.excitation_design(a.freq, mag, a.asd_opt, a.asd_flat, annotations=ann, height=height)


def drive_response_fig(a, *, height=520):
    """The injected drive (with its 3 s Tukey ramp on/off, envelope overlaid) and the
    raw sensor readout, exactly as logged during the measurement."""
    pk = float(np.max(np.abs(a.drive)))
    drive_tr = [("drive  [cts]", a.drive, sp.GOLD),
                ("Tukey envelope", a.env * pk, sp.GRAY)]
    motion_tr = [("readout  (sensor)", a.response, sp.SKY)]
    return sp.timeseries(a.t, drive_tr, motion_tr, height=height,
                         drive_unit="drive [cts]", motion_unit="readout [cts]",
                         titles=["<b>Excitation</b> — injected drive (3 s Tukey ramp on/off)",
                                 "<b>Sensor response</b> — raw readout during the measurement"])


def loop_timeseries_fig(a, *, height=620):
    """Closed-loop control-room view: the injected drive, the damper feedback it
    provokes, the reconstructed plant input (drive − feedback), and the raw readout."""
    drive_tr = [("DRIVE_EXC  (injected)", a.drive, sp.GOLD),
                ("plant input = drive − feedback", a.plant_in, sp.SKY),
                ("MC2 damper feedback", a.feedback, sp.ROSE)]
    motion_tr = [("readout  (sensor)", a.response, sp.GREEN)]
    return sp.timeseries(a.t, drive_tr, motion_tr, height=height,
                         drive_unit="drive [cts]", motion_unit="readout [cts]",
                         titles=["<b>Drive, feedback &amp; plant input</b> — the COIL_DRV_SUM \"+-\" node",
                                 "<b>Sensor response</b> — raw readout, loops live"])


def bode_audit_fig(a, *, fit=None, height=760):
    """Measured FRF (markers + σ) over the analytic oracle, with per-line coherence —
    the honest 'where do we trust this measurement' Bode."""
    traces = [dict(name="analytic oracle", H=a.oracle.eval(a.freq), color=sp.INK, width=2.4),
              dict(name="measured FRF", H=a.H, color=sp.ROSE, mode="markers",
                   err=a.H_err, mask=a.excited)]
    if fit is not None:
        traces.append(dict(name="recovered fit", H=fit.eval(a.freq), color=sp.GOLD, dash="dash"))
    return sp.bode(a.freq, traces, coh=a.coh, coh_mask=a.excited, height=height)


def residuals_fig(a, fit, *, height=380):
    """Normalised FRF residual ``(measured − fit)/σ`` over the excited lines."""
    resid = (a.H - fit.eval(a.freq)) / a.H_err
    return sp.residuals(a.freq, resid, mask=a.excited, height=height)
