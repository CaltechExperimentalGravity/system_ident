"""Shared Plotly styling + standard-panel builders for the docs/examples.

Presentation only — this module is *not* part of the `system_ident` package API
(it is never imported by the library, only by the documentation `.qmd` pages).
It exists so that **every** worked example shows the **same complete standard
plot set** in the **same style**, with the styling defined in exactly one place.

The standard set (call these in order in each example):

  1. `excitation_design`   |G| + optimal-vs-flat drive ASD (log-y)
  2. `timeseries`          drive + mirror motion (the "coaxing, not slamming" check)
  3. `bode`                magnitude + phase + coherence (true / meas±σ / prior / fit)
  4. `coherence`           γ²(f) on the excited bins        (also bundled into `bode`)
  5. `residuals`           normalized (meas−fit)/σ vs f, with histogram
  6. `parameter_recovery`  f0 / Q / gain : prior → fit(±CRB) → true
  7. `saturation`          drive peak & RMS vs the actuator limit, per pass
  8. `convergence`         max fractional uncertainty σ/θ vs pass
  9. `pass_overlay`        how the measured FRF + fit sharpen pass-by-pass

plus `param_table(...)` to reprint the prior/current/true parameters after each
step, so the reader watches the numbers move.

Styling is centralized in `style()` / the module constants: large axis-title,
tick, and legend fonts, and markers sized well above the Plotly default.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:  # rich parameter tables in the rendered page; degrade gracefully otherwise
    from IPython.display import Markdown
except Exception:  # pragma: no cover
    Markdown = None

# ── palette ──────────────────────────────────────────────────────────────────
SKY = "#1F8AC0"     # truth / primary
GOLD = "#C8973A"    # fit / designed drive
GRAY = "#94A3B8"    # measured points / flat drive / suppressed
GREEN = "#2D9D6B"   # third DoF
RED = "#E53E3E"     # prior (when shown as "wrong")
ROSE = "#D2698E"    # cross-coupling
INK = "#1B2733"     # text
GRID = "rgba(30,60,100,0.07)"
PRIOR = "rgba(200,151,58,0.55)"  # faint gold dotted prior overlay

FONT = "Outfit, Inter, sans-serif"

# ── type sizes (the "fonts are too small" fix lives here) ────────────────────
SZ_BASE = 16        # base layout font
SZ_AXIS = 19        # axis-title font
SZ_TICK = 15        # tick-label font
SZ_LEGEND = 16      # legend font
SZ_TITLE = 17       # subplot-title font
SZ_ANNOT = 14       # in-plot arrow annotations

# ── marker sizes (50 % larger than the old 4–5 px defaults) ──────────────────
MK_DATA = 8         # measured-FRF points
MK_BIG = 14         # parameter-recovery / convergence markers
MK_SMALL = 7        # dense scatter (e.g. cavity rolloff)
MK_OVERLAY = 9      # pass-by-pass overlay points

PARAM_COLORS = [SKY, GOLD, GREEN, ROSE, "#7C5CBF", RED, "#0E7C7B"]


# ── axis helpers: MATLAB/matplotlib-standard log ticks + sane log-y ranges ────
# Decade-major ticks labelled as powers of ten (10ⁿ), with unlabelled minor ticks
# at the 2–9 subdivisions — the standard `loglog`/`set_xscale('log')` look. Applied
# uniformly to every log axis in `style()`.
LOG_TICKS = dict(
    dtick=1, exponentformat="power", showexponent="all",
    ticks="outside", ticklen=6,
    minor=dict(dtick="D1", ticks="outside", ticklen=3, showgrid=True, gridcolor=GRID),
)


def _apply_logx(fig, fmin, fmax, *, row=None, col=None):
    """Set a log frequency x-axis with a tight [fmin, fmax] range.

    The tick *style* (decade majors + 2–9 minors, 10ⁿ labels) is applied to every
    log axis centrally in `style()`, so this only fixes the range.
    """
    kw = dict(type="log", range=[np.log10(fmin), np.log10(fmax)])
    fig.update_xaxes(**kw) if row is None else fig.update_xaxes(**kw, row=row, col=col)


def _logy_range(arrays, *, decades=4.0, hi_pad=2.0, lo_pad=0.7):
    """A clamped log-y range: hi just above the data, lo no more than `decades` below."""
    vals = [np.asarray(a, float).ravel() for a in arrays]
    vals = [a[np.isfinite(a) & (a > 0)] for a in vals]
    vals = [a for a in vals if a.size]
    if not vals:
        return None
    allv = np.concatenate(vals)
    hi = float(np.max(allv)) * hi_pad
    lo = max(float(np.min(allv)) * lo_pad, hi / 10 ** decades)
    return [np.log10(lo), np.log10(hi)]


def style(fig, height=None, legend="h", legend_y=1.02):
    """Apply the house style to *any* figure: fonts, grid, margins, legend.

    Every figure in the docs must pass through here — that is what keeps the
    type sizes and marker treatment uniform across all examples and tutorials.
    """
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT, size=SZ_BASE, color=INK),
        margin=dict(l=82, r=30, t=64, b=66),
        title_font=dict(family=FONT, size=SZ_AXIS + 2),
    )
    fig.update_xaxes(
        title_font=dict(size=SZ_AXIS), tickfont=dict(size=SZ_TICK), gridcolor=GRID
    )
    fig.update_yaxes(
        title_font=dict(size=SZ_AXIS), tickfont=dict(size=SZ_TICK), gridcolor=GRID
    )
    # MATLAB/matplotlib-standard log ticks (decade majors as 10ⁿ, 2–9 minors) on
    # every log axis — applied here so all panels share the same convention.
    for ax in fig.select_xaxes():
        if ax.type == "log":
            ax.update(**LOG_TICKS)
    for ax in fig.select_yaxes():
        if ax.type == "log":
            ax.update(**LOG_TICKS)
    if legend == "h":
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=legend_y, xanchor="right", x=1,
            font=dict(size=SZ_LEGEND), bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#DCE5F0", borderwidth=1,
        ))
    elif legend == "v":
        fig.update_layout(legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            font=dict(size=SZ_LEGEND), bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#DCE5F0", borderwidth=1,
        ))
    # Bump subplot titles (they have no explicit font size) but leave the
    # explicitly-sized arrow annotations alone.
    for a in fig.layout.annotations:
        if a.font.size is None:
            a.font.size = SZ_TITLE
    if height is not None:
        fig.update_layout(height=height)
    return fig


# ── physical-parameter helpers (dedupes the per-file `_f0_q`) ─────────────────
def modes(model):
    """List of ``(f0 [Hz], Q)`` for every resonant pole pair, low-f first."""
    p = np.roots(np.asarray(model.den, float))
    p = p[p.imag > 1e-9]
    out = []
    for pole in sorted(p, key=lambda z: abs(z)):
        w0 = abs(pole)
        out.append((w0 / (2 * np.pi), w0 / (2 * abs(pole.real))))
    return out


def f0_q(model):
    """``(f0, Q)`` of the first (lowest-frequency) resonance."""
    return modes(model)[0]


def dc_gain(model):
    """Low-frequency magnitude ``|G(f→0)|`` — a stable, units-free gain proxy."""
    return float(np.abs(model.eval(np.array([1e-6])))[0])


def param_table(headers, rows, caption=None):
    """Render a Markdown parameter table (prior / current / true / error …).

    `headers` is a list of column names; `rows` a list of row-lists (already
    formatted strings or numbers). Returns an `IPython.display.Markdown` so the
    table renders inline in the page; falls back to a plain string.
    """
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join(
        "| " + " | ".join(f"{c}" for c in row) + " |" for row in rows
    )
    md = "\n".join([head, rule, body])
    if caption:
        md = f"**{caption}**\n\n" + md
    return Markdown(md) if Markdown is not None else md


# ── 1. excitation design ──────────────────────────────────────────────────────
def excitation_design(freq, plant_mag, asd_opt, asd_flat, *, prior_mag=None,
                      ratio=None, annotations=(), height=560):
    """Panel 1 — plant magnitude (with prior overlay) over the optimal vs flat drive ASD.

    `annotations` is a list of ``(f0, text)`` resonance markers placed on |G|.
    """
    sub2 = "<b>Excitation design</b>  — optimal vs flat drive ASD"
    if ratio is not None:
        sub2 = f"<b>Excitation design</b>  — {ratio:.0f}× variance reduction over flat"
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=["<b>Plant magnitude</b>  |G(f)|", sub2],
    )
    fig.add_trace(go.Scatter(
        x=freq, y=plant_mag, mode="lines", line=dict(color=SKY, width=2.6),
        name="|G(f)| — true plant",
        hovertemplate="%{x:.3f} Hz   |G| = %{y:.4g}<extra></extra>"), row=1, col=1)
    if prior_mag is not None:
        fig.add_trace(go.Scatter(
            x=freq, y=prior_mag, mode="lines",
            line=dict(color=PRIOR, width=1.6, dash="dot"),
            name="|G(f)| — prior (design model)",
            hovertemplate="%{x:.3f} Hz   |G_prior| = %{y:.4g}<extra></extra>"),
            row=1, col=1)
    for f0, text in annotations:
        idx = int(np.argmin(np.abs(freq - f0)))
        fig.add_annotation(
            x=freq[idx], y=plant_mag[idx], xref="x", yref="y", text=f"<b>{text}</b>",
            showarrow=True, arrowhead=2, arrowsize=1.1,
            arrowcolor="rgba(200,151,58,0.75)", ax=58, ay=-40,
            font=dict(size=SZ_ANNOT, color=GOLD, family=FONT), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=freq, y=asd_opt, mode="lines", fill="tozeroy",
        fillcolor="rgba(200,151,58,0.15)", line=dict(color=GOLD, width=2.6),
        name="Optimal  (Fisher-matched)",
        hovertemplate="%{x:.3f} Hz   ASD = %{y:.4g}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=freq, y=asd_flat, mode="lines", line=dict(color=GRAY, width=2.0, dash="dash"),
        name="Flat  (equal total power)",
        hovertemplate="%{x:.3f} Hz   ASD = %{y:.4g}<extra></extra>"), row=2, col=1)
    mags = [plant_mag] + ([prior_mag] if prior_mag is not None else [])
    fig.update_yaxes(title_text="|G(f)|", type="log",
                     range=_logy_range(mags, decades=4), row=1, col=1)
    fig.update_yaxes(title_text="Drive ASD", type="log",
                     range=_logy_range([asd_opt, asd_flat], decades=2.5), row=2, col=1)
    fig.update_xaxes(title_text="Frequency  [Hz]", row=2, col=1)
    _apply_logx(fig, float(np.min(freq)), float(np.max(freq)))
    return style(fig, height=height)


# ── 2. measurement time series ────────────────────────────────────────────────
def timeseries(t, drive_traces, motion_traces, *, titles=None, height=480,
               drive_unit="drive [cts]", motion_unit="motion [a.u.]"):
    """Panel 2 — drive (top) and mirror motion (bottom) vs time.

    `drive_traces` / `motion_traces` are lists of ``(name, y, color)``.
    """
    if titles is None:
        titles = ["<b>Drive</b> — actuator (Schroeder multisine)",
                  "<b>Mirror motion</b> — response"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.11,
                        subplot_titles=titles)
    for name, y, color in drive_traces:
        fig.add_trace(go.Scatter(x=t, y=y[: t.size], mode="lines",
                      line=dict(color=color, width=1.4), name=name), row=1, col=1)
    for name, y, color in motion_traces:
        fig.add_trace(go.Scatter(x=t, y=y[: t.size], mode="lines",
                      line=dict(color=color, width=1.6), name=name), row=2, col=1)
    fig.update_yaxes(title_text=drive_unit, row=1, col=1)
    fig.update_yaxes(title_text=motion_unit, row=2, col=1)
    fig.update_xaxes(title_text="time [s]", row=2, col=1)
    return style(fig, height=height, legend_y=1.04)


# ── 3+4. Bode magnitude + phase + coherence ───────────────────────────────────
def bode(freq, traces, *, coh=None, coh_mask=None, height=720,
         ylabel="|G(f)|", logx=True):
    """Panels 3+4 — stacked magnitude, phase, and (optional) coherence, shared x.

    `traces` is a list of dicts, each:
        name, H (complex over freq), color,
        dash=None, mode="lines"|"markers", err=None (mag σ), mask=None, size=MK_DATA
    """
    rows = 3 if coh is not None else 2
    titles = ["<b>Magnitude</b>  |G(f)|", "<b>Phase</b>  ∠G(f)  [deg]"]
    if coh is not None:
        titles.append("<b>Coherence</b>  γ²(f)")
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.07, subplot_titles=titles)
    all_mags, line_phases = [], []
    for tr in traces:
        H = np.asarray(tr["H"])
        mask = tr.get("mask")
        f = freq if mask is None else freq[mask]
        Hm = H if mask is None else H[mask]
        mag = np.abs(Hm)
        all_mags.append(mag)
        # Unwrap on the *retained* samples only — unwrapping across the
        # zero-filled unexcited bins is what produced the −900° spikes.
        ph = np.unwrap(np.angle(Hm)) * 180 / np.pi
        mode = tr.get("mode", "lines")
        if mode != "markers":
            line_phases.append(ph)
        color = tr["color"]
        size = tr.get("size", MK_DATA)
        width = tr.get("width", 2.4)
        dash = tr.get("dash")
        err = tr.get("err")
        common = dict(name=tr["name"], legendgroup=tr["name"])
        if mode == "markers":
            mk = dict(color=color, size=size)
            ey = None
            if err is not None:
                e = err if mask is None else err[mask]
                ey = dict(type="data", array=e, visible=True,
                          color=_fade(color, 0.45), thickness=1.2, width=0)
            fig.add_trace(go.Scatter(x=f, y=mag, mode="markers", marker=mk,
                          error_y=ey, **common), row=1, col=1)
            fig.add_trace(go.Scatter(x=f, y=ph, mode="markers", marker=mk,
                          showlegend=False, legendgroup=tr["name"], name=tr["name"]),
                          row=2, col=1)
        else:
            ln = dict(color=color, width=width, dash=dash)
            fig.add_trace(go.Scatter(x=f, y=mag, mode="lines", line=ln, **common),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=f, y=ph, mode="lines", line=ln,
                          showlegend=False, legendgroup=tr["name"], name=tr["name"]),
                          row=2, col=1)
    if coh is not None:
        cf = freq if coh_mask is None else freq[coh_mask]
        cy = coh if coh_mask is None else np.asarray(coh)[coh_mask]
        fig.add_trace(go.Scatter(x=cf, y=cy, mode="markers",
                      marker=dict(color=SKY, size=MK_DATA), name="coherence",
                      showlegend=False), row=3, col=1)
        fig.update_yaxes(title_text="γ²", range=[0, 1.02], row=3, col=1)
    fig.update_yaxes(title_text=ylabel, type="log",
                     range=_logy_range(all_mags, decades=4), row=1, col=1)
    ph_range = None
    if line_phases:
        allp = np.concatenate(line_phases)
        lo = float(np.floor((np.min(allp) - 10) / 90.0) * 90.0)
        hi = float(np.ceil((np.max(allp) + 10) / 90.0) * 90.0)
        ph_range = [lo, hi]
    fig.update_yaxes(title_text="phase [deg]", dtick=90, range=ph_range, row=2, col=1)
    fig.update_xaxes(title_text="Frequency  [Hz]", row=rows, col=1)
    if logx:
        _apply_logx(fig, float(np.min(freq)), float(np.max(freq)))
    return style(fig, height=height)


# ── 4 (standalone). coherence ─────────────────────────────────────────────────
def coherence(freq, coh, mask=None, *, height=320):
    """Panel 4 (standalone) — measurement coherence γ²(f) on the excited bins."""
    f = freq if mask is None else freq[mask]
    y = coh if mask is None else np.asarray(coh)[mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=y, mode="markers",
                  marker=dict(color=SKY, size=MK_DATA), name="γ²(f)"))
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(100,120,160,0.5)")
    fig.update_xaxes(title_text="Frequency  [Hz]")
    fig.update_yaxes(title_text="coherence γ²", range=[0, 1.02])
    _apply_logx(fig, float(np.min(f)), float(np.max(f)))
    return style(fig, height=height, legend="h")


# ── 5. residuals ──────────────────────────────────────────────────────────────
def residuals(freq, resid, mask=None, *, height=360):
    """Panel 5 — normalized residual ``(meas−fit)/σ`` vs f, plus its histogram.

    A well-specified ML fit gives residuals scattered in ±~2 with no structure;
    the histogram should look standard-normal.
    """
    f = freq if mask is None else freq[mask]
    r = resid if mask is None else np.asarray(resid)[mask]
    r = np.asarray(r)[np.isfinite(r)]
    f = np.asarray(f)[: r.size]
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.72, 0.28], horizontal_spacing=0.09,
        subplot_titles=["<b>Normalized residual</b>  (meas − fit)/σ",
                        "<b>Distribution</b>"])
    fig.add_trace(go.Scatter(x=f, y=r, mode="markers",
                  marker=dict(color=GOLD, size=MK_DATA), name="residual"),
                  row=1, col=1)
    for yv in (-1, 1):
        fig.add_hline(y=yv, line_dash="dot", line_color="rgba(100,120,160,0.4)",
                      row=1, col=1)
    fig.add_hline(y=0, line_color="rgba(100,120,160,0.6)", row=1, col=1)
    fig.add_trace(go.Histogram(y=r, marker=dict(color=_fade(GOLD, 0.55)),
                  nbinsy=24, name="hist", showlegend=False), row=1, col=2)
    rms = float(np.sqrt(np.mean(r**2))) if r.size else float("nan")
    fig.add_annotation(x=1, y=1.08, xref="paper", yref="paper", showarrow=False,
                       text=f"RMS = {rms:.2f}σ", xanchor="right",
                       font=dict(size=SZ_ANNOT, color=INK, family=FONT))
    fig.update_xaxes(title_text="Frequency  [Hz]", type="log", row=1, col=1)
    fig.update_yaxes(title_text="(meas − fit)/σ", row=1, col=1)
    fig.update_xaxes(title_text="count", row=1, col=2)
    return style(fig, height=height, legend="h")


# ── 6. parameter recovery ─────────────────────────────────────────────────────
def parameter_recovery(groups, *, height=380):
    """Panel 6 — for each parameter, prior → fit(±CRB σ) → true on a small panel.

    `groups` = list of dicts: ``label, prior, fit, true, sigma=None, log=False``.
    """
    n = len(groups)
    fig = make_subplots(rows=1, cols=n, horizontal_spacing=0.08,
                        subplot_titles=[g["label"] for g in groups])
    for i, g in enumerate(groups, start=1):
        x = ["prior", "fit", "true"]
        y = [g["prior"], g["fit"], g["true"]]
        colors = [GRAY, GOLD, SKY]
        ey = None
        if g.get("sigma") is not None:
            arr = [0, g["sigma"], 0]
            ey = dict(type="data", array=arr, visible=True,
                      color=_fade(GOLD, 0.6), thickness=2, width=8)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", marker=dict(color=colors, size=MK_BIG,
            line=dict(color="white", width=1.5)), error_y=ey, showlegend=False,
            hovertemplate="%{x}: %{y:.4g}<extra></extra>"), row=1, col=i)
        fig.add_hline(y=g["true"], line_dash="dot",
                      line_color="rgba(31,138,192,0.4)", row=1, col=i)
        if g.get("log"):
            fig.update_yaxes(type="log", row=1, col=i)
        fig.update_xaxes(tickfont=dict(size=SZ_TICK), row=1, col=i)
    return style(fig, height=height, legend="h")


# ── 7. saturation / safety ────────────────────────────────────────────────────
def saturation(passes, *, limit=None, rms_ceiling=None, height=360):
    """Panel 7 — drive peak and readback RMS per pass against their safety limits.

    `passes` = list of dicts: ``pass, peak, rms``.
    """
    idx = [p["pass"] for p in passes]
    peak = [p["peak"] for p in passes]
    rms = [p.get("rms", float("nan")) for p in passes]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13,
                        subplot_titles=["<b>Drive peak</b> vs actuator limit",
                                        "<b>Readback RMS</b> vs ceiling"])
    fig.add_trace(go.Bar(x=idx, y=peak, marker_color=_fade(SKY, 0.8),
                  name="drive peak", showlegend=False), row=1, col=1)
    if limit is not None:
        fig.add_hline(y=limit, line_dash="dash", line_color=RED, row=1, col=1,
                      annotation_text=f" limit = {limit:g}", annotation_position="top left",
                      annotation_font=dict(size=SZ_ANNOT, color=RED, family=FONT))
    fig.add_trace(go.Bar(x=idx, y=rms, marker_color=_fade(GOLD, 0.8),
                  name="readback RMS", showlegend=False), row=1, col=2)
    if rms_ceiling is not None:
        fig.add_hline(y=rms_ceiling, line_dash="dash", line_color=RED, row=1, col=2,
                      annotation_text=f" ceiling = {rms_ceiling:g}",
                      annotation_position="top left",
                      annotation_font=dict(size=SZ_ANNOT, color=RED, family=FONT))
    fig.update_xaxes(title_text="pass", dtick=1, row=1, col=1)
    fig.update_xaxes(title_text="pass", dtick=1, row=1, col=2)
    fig.update_yaxes(title_text="peak [cts]", row=1, col=1)
    fig.update_yaxes(title_text="RMS [a.u.]", row=1, col=2)
    return style(fig, height=height, legend="h")


# ── 8. convergence ────────────────────────────────────────────────────────────
def convergence(series, *, target=None, height=420, xlabel="pass",
                ylabel="fractional uncertainty  σᵢ / θᵢ"):
    """Panel 8 — σ/θ (or any metric) vs pass for one or more DoFs/elements.

    `series` = list of dicts: ``name, x, y, color, symbol``.
    """
    symbols = ["circle", "square", "diamond", "triangle-up", "x"]
    fig = go.Figure()
    for k, s in enumerate(series):
        fig.add_trace(go.Scatter(
            x=list(s["x"]), y=list(s["y"]), mode="lines+markers",
            line=dict(color=s.get("color", SKY), width=2.4),
            marker=dict(symbol=s.get("symbol", symbols[k % len(symbols)]),
                        size=MK_BIG, color=s.get("color", SKY),
                        line=dict(color="white", width=1.5)),
            name=s["name"],
            hovertemplate=f"<b>{s['name']}</b><br>{xlabel} %{{x}}<br>%{{y:.3e}}<extra></extra>"))
    if target is not None:
        fig.add_hline(y=target, line_dash="dot", line_color="rgba(100,120,160,0.6)",
                      annotation_text=f"  target = {target:g}",
                      annotation_position="right",
                      annotation_font=dict(size=SZ_ANNOT, color="rgba(100,120,160,0.9)",
                                           family=FONT))
    fig.update_xaxes(title_text=xlabel, dtick=1)
    fig.update_yaxes(title_text=ylabel, type="log")
    return style(fig, height=height)


# ── 9. pass-by-pass overlay ───────────────────────────────────────────────────
def pass_overlay(freq, per_pass, true_mag, *, height=440, ylabel="|G(f)|"):
    """Panel 9 — measured FRF + fit, overlaid pass-by-pass (color = pass).

    `per_pass` = list of dicts (in pass order):
        ``meas_mag, fit_mag, mask=None``.
    Color ramps blue → gold as passes accumulate.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq, y=true_mag, mode="lines",
                  line=dict(color=INK, width=2.6), name="true plant"))
    n = len(per_pass)
    mags = [true_mag]
    for i, p in enumerate(per_pass):
        c = _ramp(i, n)
        mask = p.get("mask")
        f = freq if mask is None else freq[mask]
        mm = p["meas_mag"] if mask is None else np.asarray(p["meas_mag"])[mask]
        mags += [mm, p["fit_mag"]]
        fig.add_trace(go.Scatter(x=f, y=mm, mode="markers",
                      marker=dict(color=c, size=MK_OVERLAY, opacity=0.7,
                                  line=dict(color="white", width=0.6)),
                      name=f"meas · pass {i + 1}", legendgroup=f"p{i}"))
        fig.add_trace(go.Scatter(x=freq, y=p["fit_mag"], mode="lines",
                      line=dict(color=c, width=2.0, dash="dash"),
                      name=f"fit · pass {i + 1}", legendgroup=f"p{i}"))
    fig.update_yaxes(title_text=ylabel, type="log", range=_logy_range(mags, decades=4))
    _apply_logx(fig, float(np.min(freq)), float(np.max(freq)))
    return style(fig, height=height, legend="v")


# ── animated convergence (play button + slider over passes) ──────────────────
def _play_pause(steps_label="pass ", n=0, x=0.0, slider_y=-0.06):
    play = dict(type="buttons", showactive=False, x=x, y=1.16, xanchor="left",
                yanchor="top", direction="left", pad=dict(t=0, r=8),
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=750, redraw=True),
                                          fromcurrent=True, transition=dict(duration=300))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")])])
    steps = [dict(method="animate", label=f"{i + 1}",
                  args=[[f"{i + 1}"], dict(mode="immediate",
                        frame=dict(duration=600, redraw=True),
                        transition=dict(duration=300))]) for i in range(n)]
    slider = dict(active=0, x=0.1, len=0.88, y=slider_y, currentvalue=dict(prefix=steps_label),
                  pad=dict(t=4), steps=steps)
    return [play], [slider]


def animate_passes(freq, true_mag, per_pass, *, height=640):
    """Animated convergence over passes: Bode magnitude (true / measured / fit) above
    the reshaping drive ASD, with a play button + slider.

    `per_pass` = list of dicts: ``meas_mag, mask, fit_mag, drive_asd``.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=["<b>Bode magnitude</b> — the fit tightening onto the plant",
                        "<b>Drive ASD</b> — power reallocating toward the resonance"])
    p0 = per_pass[0]
    m0 = p0["mask"]
    fig.add_trace(go.Scatter(x=freq, y=true_mag, mode="lines",
        line=dict(color=INK, width=2.6), name="true plant"), row=1, col=1)
    fig.add_trace(go.Scatter(x=freq[m0], y=np.asarray(p0["meas_mag"])[m0], mode="markers",
        marker=dict(color=GRAY, size=MK_DATA), name="measured"), row=1, col=1)
    fig.add_trace(go.Scatter(x=freq, y=p0["fit_mag"], mode="lines",
        line=dict(color=GOLD, width=2.6, dash="dash"), name="ML fit"), row=1, col=1)
    fig.add_trace(go.Scatter(x=freq, y=p0["drive_asd"], mode="lines", fill="tozeroy",
        fillcolor="rgba(200,151,58,0.18)", line=dict(color=GOLD, width=2.4),
        name="drive ASD"), row=2, col=1)
    fig.frames = [go.Frame(name=f"{i + 1}", traces=[1, 2, 3], data=[
        go.Scatter(x=freq[p["mask"]], y=np.asarray(p["meas_mag"])[p["mask"]]),
        go.Scatter(x=freq, y=p["fit_mag"]),
        go.Scatter(x=freq, y=p["drive_asd"]),
    ]) for i, p in enumerate(per_pass)]
    fig.update_yaxes(title_text="|G(f)|", type="log", row=1, col=1)
    fig.update_yaxes(title_text="drive ASD", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Frequency  [Hz]", type="log", row=2, col=1)
    fig.update_xaxes(type="log", row=1, col=1)
    um, sl = _play_pause("pass ", n=len(per_pass))
    fig.update_layout(updatemenus=um, sliders=sl)
    return style(fig, height=height, legend_y=1.16)


def animate_design(freq, plant_mag, asd_iters, *, height=440):
    """Hero animation: the optimal drive ASD forming over design iterations,
    under the plant magnitude. `asd_iters` = list of ASD arrays (iter 0..N).
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq, y=plant_mag / np.max(plant_mag), mode="lines",
        line=dict(color=SKY, width=3), name="plant |G(f)|"))
    a0 = asd_iters[0]
    fig.add_trace(go.Scatter(x=freq, y=a0 / np.max(asd_iters[-1]), mode="lines", fill="tozeroy",
        fillcolor="rgba(200,151,58,0.20)", line=dict(color=GOLD, width=3), name="drive ASD"))
    fig.frames = [go.Frame(name=f"{i + 1}", traces=[1],
        data=[go.Scatter(x=freq, y=a / np.max(asd_iters[-1]))]) for i, a in enumerate(asd_iters)]
    fig.update_xaxes(title_text="Frequency  [Hz]", type="log")
    fig.update_yaxes(title_text="normalized", type="log", range=[-3, 0.2])
    um, sl = _play_pause("iteration ", n=len(asd_iters))
    fig.update_layout(updatemenus=um, sliders=sl)
    return style(fig, height=height, legend_y=1.16)


# ── small color utilities ─────────────────────────────────────────────────────
def _hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _fade(color, alpha):
    """Return an rgba() string for a hex (or pass through an rgba already)."""
    if color.startswith("#"):
        r, g, b = _hex2rgb(color)
        return f"rgba({r},{g},{b},{alpha})"
    return color


def _ramp(i, n):
    """Blue→gold ramp for pass index ``i`` of ``n`` (matplotlib-free)."""
    t = 0.0 if n <= 1 else i / (n - 1)
    a = np.array(_hex2rgb(SKY))
    b = np.array(_hex2rgb(GOLD))
    r, g, bl = (a + t * (b - a)).astype(int)
    return f"rgb({r},{g},{bl})"


def resolvability_map(ratio, budget, *, points=(), rayleigh=1.0, height=560,
                      title="When are two modes resolvable?"):
    """Log–log resolvability map for a parametric two-mode fit.

    ``x = Δf/Γ`` (mode separation in linewidths); ``y = budget`` is the measurement
    budget (``∝ SNR·N``) required for the CRB on the separation to fall below the
    separation itself (``σ_Δf = Δf``), normalised to the one-linewidth case. The gold
    curve is that **computed** CRB boundary: a pair plots **above** the curve → resolvable
    at that budget, **below** → not. The vertical line at ``Δf/Γ = rayleigh`` marks one
    linewidth — the classical *non-parametric* (Rayleigh / peak-pick) limit. Parametric ML
    lives to its LEFT by paying a steep but finite budget. ``points`` = iterable of
    ``(label, x, y)`` example modes.
    """
    ratio = np.asarray(ratio, float); budget = np.asarray(budget, float)
    ys = list(budget) + [p[2] for p in points]
    hi, lo = max(ys) * 4.0, min(ys) / 4.0
    fig = go.Figure()
    upper = np.full_like(ratio, hi)
    lower = np.full_like(ratio, lo)
    # shaded regions: below the boundary = unresolvable (rose), above = resolvable (green)
    fig.add_scatter(x=ratio, y=lower, mode="lines", line=dict(width=0),
                    hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=ratio, y=budget, mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=_fade(ROSE, 0.10), hoverinfo="skip",
                    name="unresolvable at this budget")
    fig.add_scatter(x=ratio, y=budget, mode="lines", line=dict(width=0),
                    hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=ratio, y=upper, mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=_fade(GREEN, 0.10), hoverinfo="skip",
                    name="resolvable at this budget")
    # the computed CRB boundary σ_Δf = Δf
    fig.add_scatter(x=ratio, y=budget, mode="lines", line=dict(color=GOLD, width=3.0),
                    name="CRB boundary  σ(Δf) = Δf")
    # the classical non-parametric (one-linewidth) limit
    fig.add_vline(x=rayleigh, line=dict(color=ROSE, width=2.0, dash="dash"),
                  annotation_text="one linewidth — non-parametric (Rayleigh) limit",
                  annotation_position="top right", annotation_font=dict(color=ROSE))
    for label, x, y in points:
        fig.add_scatter(x=[x], y=[y], mode="markers+text", text=[label],
                        textposition="top center", textfont=dict(color=SKY),
                        marker=dict(color=SKY, size=MK_BIG, symbol="diamond",
                                    line=dict(color="white", width=1.5)),
                        showlegend=False)
    fig.update_xaxes(type="log", title_text="mode separation  Δf / Γ   (linewidths)")
    fig.update_yaxes(type="log", range=[np.log10(lo), np.log10(hi)],
                     title_text="required budget  ∝ SNR·N   (rel. to one linewidth)")
    fig.update_layout(title=title)
    return style(fig, height=height)
