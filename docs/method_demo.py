# docs/method_demo.py
"""Presentation-only glue for the "Why optimal-excitation system ID" tutorial page.

NOT package API — the docs sibling of ``sysid_plots`` / ``darm_demo`` /
``rtsfreerun_demo``. It runs the real head-to-head campaigns that the page argues
from (broadband random vs leakage-free multisine; swept sine vs whole-band
multisine; flat vs Fisher-optimal multisine; open- vs closed-loop reference FRF)
and builds the page's plotly panels in the shared house style.

Every number and figure on the page comes from here, so the comparison is
reproducible and nothing is asserted that the package did not actually produce.
The campaigns reuse the library internals verbatim — ``SysIDLoop._estimate_tf_periodic``
(the leakage-free P&S FRF), ``optimal_excitation`` (the Fisher/dispersion design),
``fisher.parameter_covariance`` (the Cramer-Rao bound), and the
``sysid_campaign.run_siso_passes`` multi-pass driver — so this is the same one
pipeline, only the *excitation* (and the analysis it enables) varies between arms.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.signal as sig
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_DOCS = Path(__file__).resolve().parent
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))

import sysid_plots as sp  # noqa: E402
from sysid_campaign import run_siso_passes  # noqa: E402

from system_ident.model import TFModel  # noqa: E402
from system_ident.plant import (  # noqa: E402
    SuspensionPlant, double_pendulum, coupled_suspension,
    ALIGO_LONG_MODES_HZ, ALIGO_PITCH_MODES_HZ,
)
from system_ident.backends.twin import TwinBackend  # noqa: E402
from system_ident.loop import SysIDLoop  # noqa: E402
from system_ident.excitation import multisine_from_psd, timeseries_from_asd  # noqa: E402
from system_ident.fisher import parameter_covariance  # noqa: E402
from system_ident.design.pintelon import optimal_excitation  # noqa: E402

# ── shared measurement constants ───────────────────────────────────────────────
# A real interferometer subsystem: a few-Hz suspension band, a synchronous DFT
# window long enough that f_min sits well inside it, and a representative readout
# floor. NPER chosen so the leakage-free FRF has a genuine per-bin variance.
FS_SUS = 32.0          # suspension-band sample rate [Hz]
NPERSEG = 4096         # samples / period  -> df = fs/nperseg
SENSOR_ASD = 2.0e-3    # representative readout-noise ASD [response/sqrt(Hz)]
PX_TOTAL = 1.0         # drive-power budget (shared by every arm for a fair race)


# ════════════════════════════════════════════════════════════════════════════
# 1 · The measurement problem — a representative high-Q suspension plant
# ════════════════════════════════════════════════════════════════════════════
def problem_fig(*, height=520):
    """A representative Advanced-LIGO suspension diagonal |G(f)| over a flat readout
    floor: sharp, high-Q rigid-body modes packed into a few Hz, measured against a
    sensor-noise background — the closed-loop, high-Q, SNR-limited problem every
    subsystem faces."""
    long_modes = [(f, 400.0) for f in ALIGO_LONG_MODES_HZ]   # Q≈400 rigid-body modes
    pitch_modes = [(f, 400.0) for f in ALIGO_PITCH_MODES_HZ]
    tfs = coupled_suspension(long_modes, pitch_modes, coupling=0.15, gain=100.0)
    G = tfs[("POS", "POS")]
    ff = np.geomspace(0.2, 6.0, 1400)
    mag = np.abs(G.eval(ff))
    floor = np.full_like(ff, np.median(mag) * 3e-3)          # representative readout floor
    fig = go.Figure()
    fig.add_scatter(x=ff, y=mag, mode="lines", name="|G(f)| — suspension plant",
                    line=dict(color=sp.SKY, width=2.6))
    fig.add_scatter(x=ff, y=floor, mode="lines", name="readout-noise floor (representative)",
                    line=dict(color=sp.GRAY, width=1.8, dash="dash"))
    for f0 in ALIGO_LONG_MODES_HZ:
        i = int(np.argmin(np.abs(ff - f0)))
        fig.add_annotation(x=ff[i], y=mag[i], text=f"{f0:.2f} Hz", showarrow=True,
                           arrowhead=2, arrowsize=1.0, arrowcolor="rgba(200,151,58,0.7)",
                           ax=20, ay=-34, font=dict(size=sp.SZ_ANNOT, color=sp.GOLD))
    yr = sp._logy_range([mag, floor], decades=6)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="|G(f)|")
    fig.update_layout(title="A representative suspension: high-Q modes, finite band, "
                            "finite SNR")
    return sp.style(fig, height=height)


# ════════════════════════════════════════════════════════════════════════════
# 2 · The three excitations, side by side (fair portrait)
# ════════════════════════════════════════════════════════════════════════════
def excitations_fig(*, height=620):
    """Three actuator drives that all carry the same in-band power, in the time
    domain: a swept sine (one frequency at a time), broadband random noise (every
    frequency, not periodic → its DFT leaks), and a periodic multisine (every
    frequency, period == analysis window → leakage-free). The crest factors printed
    in the titles are real but **plant-referred** (peak/RMS of the request, not at the
    DAC after the whitening/actuation chain); phasing is treated honestly in §2.1."""
    fs = 256.0
    band = np.geomspace(2.0, 40.0, 64)
    Pxx = np.full_like(band, 1.0 / (band[-1] - band[0]))

    # swept sine: a short logarithmic chirp through the band
    Tsw = 4.0
    t = np.arange(int(Tsw * fs)) / fs
    swept = sig.chirp(t, f0=band[0], t1=Tsw, f1=band[-1], method="logarithmic")
    swept *= np.std(timeseries_from_asd(Tsw, fs, band, np.sqrt(Pxx), seed=0)) / np.std(swept)

    # broadband random (random phase) and periodic multisine (Schroeder), same PSD
    nper = int(round(fs / (band[1] - band[0]))) if len(band) > 1 else 512
    nper = 1024
    rand = timeseries_from_asd(Tsw, fs, band, np.sqrt(Pxx), seed=1)
    ms = multisine_from_psd(Pxx, fs, nper, int(np.ceil(Tsw * fs / nper)), band,
                            seed=np.random.default_rng(1))
    ms = ms[: t.size]

    def crest(x):
        return float(np.max(np.abs(x)) / np.std(x))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[
                            f"<b>Swept sine</b> — one frequency at a time "
                            f"(crest {crest(swept):.1f}, plant-referred)",
                            f"<b>Broadband random noise</b> — all frequencies, not "
                            f"periodic → leaks (crest {crest(rand):.1f}, plant-referred)",
                            f"<b>Periodic multisine</b> — all frequencies, period == "
                            f"window → leakage-free (crest {crest(ms):.1f}, plant-referred)"])
    fig.add_scatter(x=t, y=swept, mode="lines", line=dict(color=sp.GRAY, width=1.2),
                    name="swept", row=1, col=1)
    fig.add_scatter(x=t, y=rand, mode="lines", line=dict(color=sp.ROSE, width=1.0),
                    name="random", row=2, col=1)
    fig.add_scatter(x=t, y=ms, mode="lines", line=dict(color=sp.GOLD, width=1.2),
                    name="multisine", row=3, col=1)
    # mark one period of the multisine to show it is exactly periodic
    fig.add_vline(x=nper / fs, line_dash="dot", line_color="rgba(100,120,160,0.6)",
                  row=3, col=1)
    fig.update_yaxes(title_text="drive [a.u.]", row=1, col=1)
    fig.update_yaxes(title_text="drive [a.u.]", row=2, col=1)
    fig.update_yaxes(title_text="drive [a.u.]", row=3, col=1)
    fig.update_xaxes(title_text="time [s]", row=3, col=1)
    return sp.style(fig, height=height, legend="h")


# ════════════════════════════════════════════════════════════════════════════
# 2.1 · Phasing — Schroeder vs random: a plant-referred crest fact, NOT a measurement
#       advantage, and moot at the DAC (which this sim does not model)
# ════════════════════════════════════════════════════════════════════════════
def _crest(x):
    return float(np.max(np.abs(x)) / np.std(x))


def phasing_crest_campaign(n_seeds=12):
    """Real Schroeder-vs-random crest factors (peak/RMS), **plant/force-referred**,
    as a function of (a) number of lines on a flat broadband spectrum and (b) the
    spectrum shape (flat vs the concentrated Fisher-optimal ASD). Random is averaged
    over ``n_seeds`` seeds. This is the only thing multisine phase touches; it is a
    time-domain property of the *request*, and (see ``phasing_invariance_*``) it does
    not enter the FRF or the CRB at all."""
    fs, nperseg, nper = 32.0, 4096, 4
    df = fs / nperseg

    # (a) flat broadband: crest vs number of lines
    Ns = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    cs = np.zeros(Ns.size)
    cr_mean = np.zeros(Ns.size)
    cr_std = np.zeros(Ns.size)
    kstart = int(round(1.0 * nperseg / fs))
    for i, N in enumerate(Ns):
        ks = np.arange(kstart, kstart + N)
        freq = ks * fs / nperseg
        Pxx = np.full(int(N), 1.0 / (freq[-1] - freq[0]))
        cs[i] = _crest(multisine_from_psd(Pxx, fs, nperseg, nper, freq, phase="schroeder"))
        cr = [_crest(multisine_from_psd(Pxx, fs, nperseg, nper, freq, phase="random",
                                        seed=np.random.default_rng(s)))
              for s in range(n_seeds)]
        cr_mean[i], cr_std[i] = float(np.mean(cr)), float(np.std(cr))

    # (b) concentrated optimal ASD vs flat, same band — the design this page actually uses
    true = double_pendulum()
    fa = np.fft.rfftfreq(nperseg, 1 / fs)
    band = (fa >= 0.1) & (fa <= 5.0)
    freq = fa[band]
    Pyy = np.ones_like(freq)
    Pxx_opt = optimal_excitation(freq, true, Pyy, PX_TOTAL, n_iter=6)
    Pxx_flat = np.full_like(freq, PX_TOTAL / (freq[-1] - freq[0]))
    n_sig = int(np.sum(Pxx_opt > 1e-6 * Pxx_opt.max()))
    shape = {}
    for label, Pxx in [("flat", Pxx_flat), ("optimal", Pxx_opt)]:
        c_s = _crest(multisine_from_psd(Pxx, fs, nperseg, nper, freq, phase="schroeder"))
        cr = [_crest(multisine_from_psd(Pxx, fs, nperseg, nper, freq, phase="random",
                                        seed=np.random.default_rng(s)))
              for s in range(n_seeds)]
        shape[label] = (c_s, float(np.mean(cr)), float(np.std(cr)))
    return SimpleNamespace(Ns=Ns, cs=cs, cr_mean=cr_mean, cr_std=cr_std,
                           shape=shape, n_sig=n_sig, n_band=int(freq.size),
                           n_seeds=n_seeds)


def phasing_crest_fig(c, *, height=460):
    """Plant-referred crest factor vs number of lines: Schroeder stays ~flat while
    random grows ~sqrt(ln N). The gap is real but plant-referred — see the caption /
    text for why it does not transfer to the DAC."""
    fig = go.Figure()
    fig.add_scatter(x=c.Ns, y=c.cs, mode="lines+markers", name="Schroeder phase",
                    line=dict(color=sp.GOLD, width=2.6),
                    marker=dict(color=sp.GOLD, size=sp.MK_BIG))
    fig.add_scatter(x=c.Ns, y=c.cr_mean, mode="lines+markers",
                    name=f"random phase (mean ± std, {c.n_seeds} seeds)",
                    line=dict(color=sp.ROSE, width=2.6),
                    marker=dict(color=sp.ROSE, size=sp.MK_BIG),
                    error_y=dict(type="data", array=c.cr_std, visible=True,
                                 color=sp._fade(sp.ROSE, 0.5), thickness=1.2, width=4))
    span = np.concatenate([c.cs, c.cr_mean + c.cr_std, c.cr_mean - c.cr_std])
    pad = 0.1 * (span.max() - span.min())
    fig.update_xaxes(type="log", title_text="number of lines  N")
    fig.update_yaxes(title_text="crest factor  peak/RMS  (plant-referred)",
                     range=[max(0.0, span.min() - pad), span.max() + pad])
    fig.update_layout(title="Plant-referred crest factor, flat broadband multisine — "
                            "Schroeder vs random phase")
    return sp.style(fig, height=height)


def phasing_shape_md(c):
    """Inline bullets: plant-referred crest on the flat vs concentrated-optimal ASD."""
    from IPython.display import Markdown
    sf, rf, rfs = c.shape["flat"]
    so, ro, ros = c.shape["optimal"]
    return Markdown(
        f"- **flat spectrum** ({c.n_band} bins): Schroeder crest **{sf:.2f}**, "
        f"random **{rf:.2f} ± {rfs:.2f}** — random is {rf/sf:.2f}× higher.\n"
        f"- **concentrated optimal ASD** (~{c.n_sig} significant lines): Schroeder crest "
        f"**{so:.2f}**, random **{ro:.2f} ± {ros:.2f}** — random only {ro/so:.2f}× higher.")


def phasing_invariance_campaign(n_seeds=8):
    """Same plant, same flat drive PSD, same noise seeds — only the multisine *phase*
    differs (Schroeder vs random). Shows the leakage-free FRF, its per-bin σ, and the
    seed-averaged accuracy are statistically identical, and that the Cramér–Rao bound
    does not even take phase as an input (it is a function of the line PSD ``Pxx``).
    Phase changes the time-domain crest; it does not change the measurement."""
    fs, nperseg, nper = 32.0, 4096, 8
    sensor_asd = 5.0e-4
    true = double_pendulum()
    fa = np.fft.rfftfreq(nperseg, 1 / fs)
    band = (fa >= 0.2) & (fa <= 4.0)
    freq = fa[band]
    Pxx = np.full_like(freq, PX_TOTAL / (freq[-1] - freq[0]))
    T = nperseg * nper / fs
    truth = true.eval(freq)

    def estimate(phase, seed):
        be = TwinBackend(SuspensionPlant({"POS": true}, fs), {"E": "POS"}, {"R": "POS"},
                         fs=fs, sensor_asd=sensor_asd, seed=seed, ramp_s=0.0)
        x = multisine_from_psd(Pxx, fs, nperseg, nper, freq, phase=phase,
                               seed=np.random.default_rng(seed))
        be.inject("E", x, fs)
        seg = be.read(["E", "R"], T)
        return SysIDLoop._estimate_tf_periodic(seg["E"], seg["R"], fs, nperseg, band,
                                               n_transient=1)

    # representative single realisation (same noise seed) for the Bode overlay
    Hs, Hes, _ = estimate("schroeder", 0)
    Hr, Her, _ = estimate("random", 0)

    out = {}
    for phase in ["schroeder", "random"]:
        errs, sigs = [], []
        for s in range(n_seeds):
            H, He, coh = estimate(phase, s)
            g = np.isfinite(He) & (coh > 0.9)
            errs.append(float(np.sqrt(np.mean(
                (np.abs(H[g] - truth[g]) / np.abs(truth[g])) ** 2))))
            sigs.append(float(np.median(He[g])))
        out[phase] = (float(np.mean(errs)), float(np.std(errs)), float(np.mean(sigs)))
    cov = parameter_covariance(freq, true, Pxx, np.ones_like(freq), T)
    crb = float(np.sqrt(np.max(np.diag(cov))))
    return SimpleNamespace(freq=freq, truth=truth, Hs=Hs, Hr=Hr, Hes=Hes, Her=Her,
                           band_mask=np.isfinite(Hes) & np.isfinite(Her),
                           res=out, crb=crb, n_seeds=n_seeds)


def phasing_invariance_table(c):
    es, ss, sigs = c.res["schroeder"]
    er, sr, sigr = c.res["random"]
    rows = [
        ["RMS frac. FRF error to truth (coh>0.9)",
         f"{es*100:.2f}% ± {ss*100:.2f}%", f"{er*100:.2f}% ± {sr*100:.2f}%"],
        ["median per-bin σ(FRF)", f"{sigs:.3g}", f"{sigr:.3g}"],
        ["Cramér–Rao worst-parameter σ", f"{c.crb:.3g}", f"{c.crb:.3g}"],
    ]
    return sp.param_table(
        ["figure of merit (same plant, same PSD, same seeds)",
         "Schroeder phase", "random phase"], rows,
        caption=f"The estimate is phase-invariant: over {c.n_seeds} seeds the recovered "
                "FRF, its per-bin σ, and the CRB are identical within scatter — the CRB "
                "does not even take phase as an input (it is a function of the line PSD). "
                "Phase changes only the plant-referred crest factor, which this sim cannot "
                "refer to the DAC.")


# ════════════════════════════════════════════════════════════════════════════
# 3 · Broadband random leaks + is variance-inefficient (high-Q resonance)
# ════════════════════════════════════════════════════════════════════════════
# A single sharp mode (Q≈60) whose linewidth (f0/Q) spans only ~2 DFT bins — the
# regime where a windowed transform of a one-off random record leaks the peak.
_LEAK = SimpleNamespace(f0=1.0, Q=60.0, gain=300.0, fs=64.0, nperseg=8192, nper=8,
                        fmin=0.3, fmax=5.0)


def _leak_plant():
    return TFModel.from_resonances([(_LEAK.f0, _LEAK.Q)], _LEAK.gain)


def _leak_grid():
    fa = np.fft.rfftfreq(_LEAK.nperseg, 1 / _LEAK.fs)
    band = (fa >= _LEAK.fmin) & (fa <= _LEAK.fmax)
    return fa, band, fa[band]


def _leak_drive_psd(freq):
    return np.full_like(freq, PX_TOTAL / (freq[-1] - freq[0]))


def _multisine_frf(G, freq, band, seed):
    """One leakage-free P&S FRF: periodic multisine in, synchronous-DFT FRF out."""
    Pxx = _leak_drive_psd(freq)
    T = _LEAK.nperseg * _LEAK.nper / _LEAK.fs
    be = TwinBackend(SuspensionPlant({"POS": G}, _LEAK.fs), {"E": "POS"}, {"R": "POS"},
                     fs=_LEAK.fs, sensor_asd=SENSOR_ASD, seed=seed, ramp_s=0.0)
    x = multisine_from_psd(Pxx, _LEAK.fs, _LEAK.nperseg, _LEAK.nper, freq,
                           seed=np.random.default_rng(seed))
    be.inject("E", x, _LEAK.fs)
    seg = be.read(["E", "R"], T)
    return SysIDLoop._estimate_tf_periodic(seg["E"], seg["R"], _LEAK.fs, _LEAK.nperseg,
                                           band, n_transient=1)


def _welch_frf(G, freq, band, seed):
    """One windowed-Welch FRF from a broadband random record of equal power and time:
    H = S_yx / S_xx with a Hann window and 50% overlap — the standard random-noise
    estimator, which leaks a sharp resonance."""
    Pxx = _leak_drive_psd(freq)
    T = _LEAK.nperseg * _LEAK.nper / _LEAK.fs
    be = TwinBackend(SuspensionPlant({"POS": G}, _LEAK.fs), {"E": "POS"}, {"R": "POS"},
                     fs=_LEAK.fs, sensor_asd=SENSOR_ASD, seed=seed, ramp_s=0.0)
    x = timeseries_from_asd(T, _LEAK.fs, freq, np.sqrt(Pxx), seed=seed)
    be.inject("E", x, _LEAK.fs)
    seg = be.read(["E", "R"], T)
    _, Pxy = sig.csd(seg["R"], seg["E"], fs=_LEAK.fs, nperseg=_LEAK.nperseg,
                     noverlap=_LEAK.nperseg // 2, window="hann")
    _, Pxx_w = sig.welch(seg["E"], fs=_LEAK.fs, nperseg=_LEAK.nperseg,
                         noverlap=_LEAK.nperseg // 2, window="hann")
    return (Pxy / Pxx_w)[band]


def leakage_campaign(seed=0):
    """One realisation each of the leakage-free multisine FRF and the windowed-Welch
    random FRF on the same high-Q plant, same power, same wall-clock."""
    G = _leak_plant()
    fa, band, freq = _leak_grid()
    H_ms, He_ms, coh = _multisine_frf(G, freq, band, seed)
    H_w = _welch_frf(G, freq, band, seed)
    truth = G.eval(freq)
    ip = int(np.argmin(np.abs(freq - _LEAK.f0)))
    T = _LEAK.nperseg * _LEAK.nper / _LEAK.fs
    return SimpleNamespace(freq=freq, band=band, H_ms=H_ms, He_ms=He_ms, coh=coh,
                           H_w=H_w, truth=truth, ip=ip, G=G, T=T,
                           excited=np.isfinite(He_ms))


def leakage_variance(n_seeds=8):
    """Repeat both estimators over ``n_seeds`` and report the peak-bin bias and
    scatter — the bias is the leakage, the scatter is the variance inefficiency."""
    G = _leak_plant()
    fa, band, freq = _leak_grid()
    ip = int(np.argmin(np.abs(freq - _LEAK.f0)))
    truth = float(np.abs(G.eval(freq))[ip])
    ms, we = [], []
    for s in range(n_seeds):
        H_ms, _, _ = _multisine_frf(G, freq, band, s)
        H_w = _welch_frf(G, freq, band, s)
        ms.append(abs(H_ms[ip]))
        we.append(abs(H_w[ip]))
    ms, we = np.array(ms), np.array(we)
    rmse_ms = float(np.sqrt(np.mean((ms - truth) ** 2)))
    rmse_w = float(np.sqrt(np.mean((we - truth) ** 2)))
    return SimpleNamespace(
        truth=truth, n_seeds=n_seeds,
        ms_mean=float(ms.mean()), ms_std=float(ms.std()),
        w_mean=float(we.mean()), w_std=float(we.std()),
        ms_bias=float(ms.mean() / truth - 1.0), w_bias=float(we.mean() / truth - 1.0),
        rmse_ms=rmse_ms, rmse_w=rmse_w, rmse_ratio=rmse_w / max(rmse_ms, 1e-12),
        linewidth_bins=(_LEAK.f0 / _LEAK.Q) / (_LEAK.fs / _LEAK.nperseg))


def leakage_bode_fig(c, *, height=560):
    """The two FRFs over the true plant: the multisine lands on the peak; the
    windowed Welch estimate reads the sharp mode low (leakage bias)."""
    m = c.excited
    fig = go.Figure()
    fig.add_scatter(x=c.freq, y=np.abs(c.truth), mode="lines", name="true plant |G(f)|",
                    line=dict(color=sp.INK, width=2.6))
    fig.add_scatter(x=c.freq[m], y=np.abs(c.H_ms[m]), mode="markers",
                    name="leakage-free multisine (±σ)",
                    marker=dict(color=sp.GOLD, size=sp.MK_DATA),
                    error_y=dict(type="data", array=c.He_ms[m], visible=True,
                                 color=sp._fade(sp.GOLD, 0.4), width=0, thickness=1.1))
    fig.add_scatter(x=c.freq, y=np.abs(c.H_w), mode="markers", name="windowed Welch (random)",
                    marker=dict(color=sp.ROSE, size=sp.MK_SMALL, symbol="x"))
    yr = sp._logy_range([np.abs(c.truth), np.abs(c.H_ms[m]), np.abs(c.H_w)], decades=4)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="|G(f)|")
    fig.update_layout(title=f"High-Q mode (Q={_LEAK.Q:.0f}): leakage-free multisine vs "
                            f"windowed random — same power, same {c.T:.0f} s")
    return sp.style(fig, height=height)


def leakage_table(v):
    rows = [
        ["peak-bin estimate (mean)", f"{v.ms_mean:.1f}", f"{v.w_mean:.1f}", f"{v.truth:.1f}"],
        ["bias vs truth", f"{v.ms_bias*100:+.1f}%", f"{v.w_bias*100:+.1f}%", "0"],
        [f"scatter (std, {v.n_seeds} seeds)", f"{v.ms_std:.2f}", f"{v.w_std:.2f}", "—"],
        ["peak RMSE (bias+scatter)", f"{v.rmse_ms:.2f}", f"{v.rmse_w:.2f}", "—"],
    ]
    return sp.param_table(["peak-bin figure of merit", "leakage-free multisine",
                           "windowed random (Welch)", "truth"], rows,
                          caption=f"High-Q peak recovery (Q={_LEAK.Q:.0f}, linewidth "
                                  f"≈{v.linewidth_bins:.1f} bins): the random estimator "
                                  f"is biased AND noisier — combined it is "
                                  f"{v.rmse_ratio:.0f}× worse at the peak")


# ════════════════════════════════════════════════════════════════════════════
# 4 · Swept sine is slow (σ vs total time)
# ════════════════════════════════════════════════════════════════════════════
_SWEEP = SimpleNamespace(fs=32.0, nperseg=4096, fmin=0.2, fmax=4.0, nper=16,
                         dwell=4, sensor_asd=2.0e-3)


def _sweep_grid():
    fa = np.fft.rfftfreq(_SWEEP.nperseg, 1 / _SWEEP.fs)
    band = (fa >= _SWEEP.fmin) & (fa <= _SWEEP.fmax)
    return fa, band, fa[band]


def sweep_campaign(seed=0):
    """Same twin, same total drive power. The multisine measures every band bin in
    ONE window of T s; in that same T a swept sine — full power on one line at a time,
    a few-period dwell each — reaches only a handful of frequencies. Covering the
    whole band to the same per-bin σ costs the sweep ~N× longer."""
    true = double_pendulum()
    fa, band, freq = _sweep_grid()
    Pxx = np.full_like(freq, PX_TOTAL / (freq[-1] - freq[0]))
    T = _SWEEP.nperseg * _SWEEP.nper / _SWEEP.fs

    # whole-band multisine
    be = TwinBackend(SuspensionPlant({"POS": true}, _SWEEP.fs), {"E": "POS"}, {"R": "POS"},
                     fs=_SWEEP.fs, sensor_asd=_SWEEP.sensor_asd, seed=seed, ramp_s=0.0)
    x = multisine_from_psd(Pxx, _SWEEP.fs, _SWEEP.nperseg, _SWEEP.nper, freq,
                           seed=np.random.default_rng(seed))
    be.inject("E", x, _SWEEP.fs)
    seg = be.read(["E", "R"], T)
    H, H_err, _ = SysIDLoop._estimate_tf_periodic(seg["E"], seg["R"], _SWEEP.fs,
                                                  _SWEEP.nperseg, band, n_transient=1)
    excited = np.isfinite(H_err)
    frac_ms = H_err / np.abs(H)
    n_bins = int(excited.sum())

    # equal wall-clock sweep: dwell periods/point, n_pts = nper // dwell points
    n_pts = max(2, _SWEEP.nper // _SWEEP.dwell)
    pts = np.geomspace(_SWEEP.fmin, _SWEEP.fmax, n_pts)
    frac_sw = np.full(n_pts, np.nan)
    rng = np.random.default_rng(seed)
    for i, fp in enumerate(pts):
        k = int(np.argmin(np.abs(fa - fp)))
        b = (fa >= fa[k] - 1e-9) & (fa <= fa[k] + 1e-9)
        Pl = np.array([PX_TOTAL / (fa[1] - fa[0])])      # all power on the one line
        bei = TwinBackend(SuspensionPlant({"POS": true}, _SWEEP.fs), {"E": "POS"},
                          {"R": "POS"}, fs=_SWEEP.fs, sensor_asd=_SWEEP.sensor_asd,
                          seed=rng, ramp_s=0.0)
        xi = multisine_from_psd(Pl, _SWEEP.fs, _SWEEP.nperseg, _SWEEP.dwell,
                                np.array([fa[k]]), seed=rng)
        bei.inject("E", xi, _SWEEP.fs)
        si = bei.read(["E", "R"], _SWEEP.nperseg * _SWEEP.dwell / _SWEEP.fs)
        Hi, Hi_err, _ = SysIDLoop._estimate_tf_periodic(si["E"], si["R"], _SWEEP.fs,
                                                        _SWEEP.nperseg, b, n_transient=0)
        sel = np.isfinite(Hi_err) & (np.abs(Hi) > 0)
        if np.any(sel):
            frac_sw[i] = float(np.min(Hi_err[sel] / np.abs(Hi[sel])))
    T_used = n_pts * _SWEEP.dwell * _SWEEP.nperseg / _SWEEP.fs
    t_cover = n_bins * _SWEEP.dwell * _SWEEP.nperseg / _SWEEP.fs
    return SimpleNamespace(freq=freq, frac_ms=frac_ms, excited=excited, pts=pts,
                           frac_sw=frac_sw, T=T, T_used=T_used, t_cover=t_cover,
                           n_bins=n_bins, n_pts=n_pts, cover_factor=t_cover / T)


def sweep_fig(c, *, height=520):
    m = c.excited & np.isfinite(c.frac_ms) & (c.frac_ms > 0)
    sw_ok = np.isfinite(c.frac_sw) & (c.frac_sw > 0)
    fig = go.Figure()
    fig.add_scatter(x=c.freq[m], y=c.frac_ms[m], mode="lines",
                    name=f"P&S multisine — all {c.n_bins} bins in one {c.T:.0f} s window",
                    line=dict(color=sp.GOLD, width=2.6))
    fig.add_scatter(x=c.pts[sw_ok], y=c.frac_sw[sw_ok], mode="markers",
                    name=f"swept sine — {c.n_pts} points in the same {c.T_used:.0f} s",
                    marker=dict(color=sp.GRAY, size=sp.MK_BIG, symbol="x",
                                line=dict(width=1.5)))
    # span the full data (multisine envelope + every sweep point) so nothing clips
    span = np.concatenate([c.frac_ms[m], c.frac_sw[sw_ok]])
    decades = float(np.log10(span.max() / span.min())) + 0.6
    yr = sp._logy_range([c.frac_ms[m], c.frac_sw[sw_ok]], decades=decades)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="σ(FRF)/|FRF|  (per bin)")
    fig.add_annotation(x=0.5, y=1.0, xref="paper", yref="paper", yanchor="bottom",
                       showarrow=False, font=dict(size=sp.SZ_ANNOT, color=sp.INK),
                       text=f"full-band coverage by sweep ≈ {c.t_cover/3600:.0f} h "
                            f"({c.cover_factor:.0f}× the one multisine window)")
    fig.update_layout(title="Fractional FRF uncertainty — same twin, same power, "
                            "equal wall-clock")
    return sp.style(fig, height=height)


# ════════════════════════════════════════════════════════════════════════════
# 5 · Optimal (Fisher) multisine wins on time AND drive (the headline)
# ════════════════════════════════════════════════════════════════════════════
_HEAD = SimpleNamespace(fs=32.0, nperseg=4096, nper=6, fmin=0.1, fmax=5.0,
                        sensor_asd=1.0e-3, n_passes=4, target=0.05)


def _head_grid():
    fa = np.fft.rfftfreq(_HEAD.nperseg, 1 / _HEAD.fs)
    band = (fa >= _HEAD.fmin) & (fa <= _HEAD.fmax)
    return fa, band, fa[band]


def _passes_to_target(hist, target):
    for h in hist:
        if h["frac"] <= target:
            return h["pass"]
    return None


def _arm_summary(hist):
    pk = max(float(np.max(np.abs(h["drive"]))) for h in hist)
    rms = float(np.sqrt(np.mean([np.var(h["drive"]) for h in hist])))
    return pk, rms


def headline_campaign():
    """Three real campaigns on the double pendulum, identical except the excitation:
      • flat multisine at the full power budget,
      • Fisher-optimal multisine at the full budget (same drive → far lower σ, fewer
        passes), and
      • Fisher-optimal multisine at budget/F (matches flat's σ with much less drive).
    F is the exact Cramer-Rao variance-reduction factor (worst diagonal) at equal
    power and time, computed from ``fisher.parameter_covariance``."""
    true = double_pendulum()
    prior = TFModel.from_resonances([(0.55, 14.0), (1.6, 22.0)], 250.0)
    fa, band, freq = _head_grid()
    Pyy = np.ones_like(freq)
    T = _HEAD.nperseg * _HEAD.nper / _HEAD.fs

    # exact Fisher/CRB variance-reduction factor F (optimal vs flat, equal power+time)
    Pxx_opt = optimal_excitation(freq, true, Pyy, PX_TOTAL, n_iter=6)
    Pxx_flat = np.full_like(freq, PX_TOTAL / (freq[-1] - freq[0]))
    cov_opt = parameter_covariance(freq, true, Pxx_opt, Pyy, T)
    cov_flat = parameter_covariance(freq, true, Pxx_flat, Pyy, T)
    F = float(np.max(np.diag(cov_flat)) / np.max(np.diag(cov_opt)))

    def run(px_total, flat):
        tw = TwinBackend(SuspensionPlant({"POS": true}, _HEAD.fs), {"E": "POS"},
                         {"R": "POS"}, fs=_HEAD.fs, sensor_asd=_HEAD.sensor_asd, seed=0)
        return run_siso_passes(tw, "E", "R", prior, fs=_HEAD.fs, nperseg=_HEAD.nperseg,
                               n_periods=_HEAD.nper, band=band, freq=freq, Pyy=Pyy,
                               px_total=px_total, n_passes=_HEAD.n_passes,
                               prior_uncertainty=0.5, flat_drive=flat, seed=0)

    h_flat = run(PX_TOTAL, True)
    h_opt = run(PX_TOTAL, False)
    h_match = run(PX_TOTAL / F, False)          # optimal, dialled down to match flat's σ

    pk_flat, rms_flat = _arm_summary(h_flat)
    pk_opt, rms_opt = _arm_summary(h_opt)
    pk_match, rms_match = _arm_summary(h_match)
    return SimpleNamespace(
        true=true, prior=prior, freq=freq, Pyy=Pyy, T=T, F=F,
        Pxx_opt=Pxx_opt, Pxx_flat=Pxx_flat,
        h_flat=h_flat, h_opt=h_opt, h_match=h_match,
        sigma_flat=h_flat[-1]["frac"], sigma_opt=h_opt[-1]["frac"],
        sigma_match=h_match[-1]["frac"],
        pk_flat=pk_flat, rms_flat=rms_flat, pk_opt=pk_opt, rms_opt=rms_opt,
        pk_match=pk_match, rms_match=rms_match,
        ttt_flat=_passes_to_target(h_flat, _HEAD.target),
        ttt_opt=_passes_to_target(h_opt, _HEAD.target),
        target=_HEAD.target, t_pass=_HEAD.nperseg * _HEAD.nper / _HEAD.fs,
        peak_drop=pk_flat / pk_match, rms_drop=rms_flat / rms_match)


def headline_design_fig(h, *, height=560):
    """The optimal drive ASD concentrating the power budget at the two modes, vs the
    flat drive — annotated with the exact CRB variance-reduction factor F."""
    return sp.excitation_design(
        h.freq, np.abs(h.true.eval(h.freq)), np.sqrt(h.Pxx_opt), np.sqrt(h.Pxx_flat),
        prior_mag=np.abs(h.prior.eval(h.freq)), ratio=h.F,
        annotations=[(0.6, "0.6 Hz, Q=20"), (1.5, "1.5 Hz, Q=30")], height=height)


def headline_convergence_fig(h, *, height=440):
    """Worst-parameter σ/θ pass by pass, all three arms. Optimal at equal power
    crosses the target in a couple of passes; flat never does."""
    def series(hist, name, color, sym):
        return dict(name=name, x=[r["pass"] for r in hist],
                    y=[r["frac"] for r in hist], color=color, symbol=sym)
    return sp.convergence([
        series(h.h_flat, "flat multisine (power P)", sp.GRAY, "square"),
        series(h.h_opt, "optimal multisine (power P)", sp.GOLD, "circle"),
        series(h.h_match, f"optimal multisine (power P/{h.F:.0f})", sp.SKY, "diamond"),
    ], target=h.target, height=height)


def headline_table(h):
    rows = [
        ["flat multisine", "P", f"{h.rms_flat:.2f}",
         f"{h.sigma_flat:.3f}", f"{h.ttt_flat or '—'}"],
        ["optimal multisine", "P", f"{h.rms_opt:.2f}",
         f"{h.sigma_opt:.3f}", f"{h.ttt_opt or '—'}"],
        [f"optimal multisine", f"P/{h.F:.0f}", f"{h.rms_match:.2f}",
         f"{h.sigma_match:.3f}", f"{_passes_to_target(h.h_match, h.target) or '—'}"],
    ]
    return sp.param_table(
        ["excitation", "drive power", "RMS drive (plant-referred)", f"final σ/θ",
         f"passes to σ/θ≤{h.target:g}"], rows,
        caption=f"Same plant, same {_HEAD.n_passes} passes. At equal power the optimal "
                f"drive reaches {h.sigma_flat/h.sigma_opt:.1f}× lower σ in fewer passes "
                f"(the time win); dialled to P/F it MATCHES flat's σ at {h.rms_drop:.1f}× "
                f"less in-band drive power (RMS, F={h.F:.0f} from the CRB). RMS is "
                f"plant-referred; the binding actuator limit is DAC saturation after the "
                f"whitening/actuation chain, which this sim does not model — so no "
                f"peak/crest advantage is claimed.")


# ════════════════════════════════════════════════════════════════════════════
# 6 · Closed loop is not an obstacle (reference-based FRF)
# ════════════════════════════════════════════════════════════════════════════
_CL = SimpleNamespace(f0=1.0, Q=50.0, gain=300.0, fs=64.0, nperseg=4096, nper=8,
                      fmin=0.3, fmax=5.0, kd=0.02, wc=40.0, sensor_asd=1.0e-4)


def closed_loop_campaign(seed=0):
    """A single high-Q suspension mode with a velocity damper closed around it
    (the kind of loop a control room runs). Driving after the controller and reading
    the drive monitor u, the leakage-free reference FRF mean(Y)/mean(U) recovers the
    OPEN-loop plant G — even though the loop suppresses the resonance ~50×. The naive
    FRF taken against the injected reference instead returns the suppressed
    closed-loop response."""
    G = TFModel.from_resonances([(_CL.f0, _CL.Q)], _CL.gain)
    Gn, Gd = G.num, G.den
    Cn, Cd = np.array([_CL.kd, 0.0]), np.array([1.0 / _CL.wc, 1.0])
    fa = np.fft.rfftfreq(_CL.nperseg, 1 / _CL.fs)
    band = (fa >= _CL.fmin) & (fa <= _CL.fmax)
    freq = fa[band]
    # closed-loop poles (stability check) and the analytic suppressed response T
    D = np.polyadd(np.polymul(Gd, Cd), np.polymul(Gn, Cn))
    stable = bool(np.all(np.roots(D).real < 0))
    jw = 2j * np.pi * freq
    T_resp = np.polyval(np.polymul(Gn, Cd), jw) / np.polyval(D, jw)

    plant = SuspensionPlant({"POS": G}, _CL.fs)
    be = TwinBackend(plant, {"EXC": "POS"}, {"RSP": "POS"}, fs=_CL.fs,
                     sensor_asd=_CL.sensor_asd, controllers={"POS": (Cn, Cd)},
                     injection_point="after_controller", drive_channels={"DRV": "POS"},
                     seed=seed, ramp_s=0.0)
    Pxx = np.full_like(freq, PX_TOTAL / (freq[-1] - freq[0]))
    x = multisine_from_psd(Pxx, _CL.fs, _CL.nperseg, _CL.nper, freq,
                           seed=np.random.default_rng(seed))
    be.inject("EXC", x, _CL.fs)
    seg = be.read(["DRV", "RSP"], _CL.nperseg * _CL.nper / _CL.fs)
    # reference-based FRF: X = after-controller drive monitor u  → recovers open-loop G
    H_ref, He_ref, coh = SysIDLoop._estimate_tf_periodic(seg["DRV"], seg["RSP"], _CL.fs,
                                                         _CL.nperseg, band, n_transient=1)
    # naive FRF: X = the injected reference r  → recovers the suppressed loop response
    xr = x[: len(seg["RSP"])]
    H_naive, _, _ = SysIDLoop._estimate_tf_periodic(xr, seg["RSP"], _CL.fs, _CL.nperseg,
                                                    band, n_transient=1)
    truth = G.eval(freq)
    good = np.abs(H_ref) > 0
    rel = float(np.median(np.abs(H_ref[good] - truth[good]) / np.abs(truth[good])))
    ip = int(np.argmin(np.abs(freq - _CL.f0)))
    suppression = float(np.abs(truth[ip]) / np.abs(T_resp[ip]))
    return SimpleNamespace(freq=freq, truth=truth, T_resp=T_resp, H_ref=H_ref,
                           He_ref=He_ref, H_naive=H_naive, excited=np.isfinite(He_ref),
                           rel=rel, suppression=suppression, stable=stable, ip=ip)


def closed_loop_fig(c, *, height=560):
    m = c.excited
    fig = go.Figure()
    fig.add_scatter(x=c.freq, y=np.abs(c.truth), mode="lines", name="open-loop plant G(f)",
                    line=dict(color=sp.INK, width=2.6))
    fig.add_scatter(x=c.freq, y=np.abs(c.T_resp), mode="lines",
                    name="closed-loop response (controller suppresses)",
                    line=dict(color=sp.GRAY, width=2.0, dash="dash"))
    fig.add_scatter(x=c.freq[m], y=np.abs(c.H_naive[m]), mode="markers",
                    name="naive FRF (vs reference) — suppressed",
                    marker=dict(color=sp.ROSE, size=sp.MK_SMALL, symbol="x"))
    fig.add_scatter(x=c.freq[m], y=np.abs(c.H_ref[m]), mode="markers",
                    name="reference-based FRF (vs drive monitor) — recovers G",
                    marker=dict(color=sp.GOLD, size=sp.MK_DATA))
    yr = sp._logy_range([np.abs(c.truth), np.abs(c.T_resp), np.abs(c.H_ref[m])], decades=4)
    fig.update_xaxes(type="log", title_text="frequency [Hz]")
    fig.update_yaxes(type="log", range=yr, title_text="|G(f)|")
    fig.update_layout(title=f"Closed loop (resonance suppressed {c.suppression:.0f}×): "
                            f"the reference-based FRF recovers the open-loop plant")
    return sp.style(fig, height=height)


# ════════════════════════════════════════════════════════════════════════════
# 7 · Consolidated head-to-head (one representative plant, real numbers)
# ════════════════════════════════════════════════════════════════════════════
def headtohead_table(head, leak, sweep, cl):
    """One table over {swept, broadband random, flat multisine, optimal multisine},
    populated from the real campaigns above."""
    headers = ["excitation", "time to target σ", "RMS drive (plant-referred)",
               "leakage bias (high-Q peak)", "per-bin noise model?", "closed-loop safe?"]
    rows = [
        ["swept sine",
         f"~{sweep.cover_factor:.0f}× a multisine window", f"{head.rms_flat:.2f}",
         "none (1 line)", "per-line coherence", "yes, but slow"],
        ["broadband random",
         "never (biased)", f"{head.rms_flat:.2f}",
         f"{leak.w_bias*100:+.0f}% (leaks)", "no (assumed)", "no (S_yx/S_xx biased)"],
        ["flat multisine",
         f"{head.ttt_flat or '> budget'} passes",
         f"{head.rms_flat:.2f}", f"{leak.ms_bias*100:+.0f}% (leakage-free)",
         "yes (period scatter)", "yes (reference-based)"],
        ["optimal multisine",
         f"{head.ttt_opt} passes", f"{head.rms_match:.2f}",
         f"{leak.ms_bias*100:+.0f}% (leakage-free)", "yes (period scatter)",
         "yes (reference-based)"],
    ]
    return sp.param_table(headers, rows,
        caption="Head-to-head on the representative suspension plants of this page. "
                "RMS for the optimal arm is at the budget that MATCHES flat's σ "
                f"(F={head.F:.0f}); RMS is plant-referred (the binding limit is DAC "
                "saturation after the whitening chain, not modeled here, so no peak/crest "
                f"column is shown). The closed-loop reference FRF recovered the "
                f"open-loop plant to {cl.rel*100:.1f}% through a {cl.suppression:.0f}× "
                "suppressing loop.")
