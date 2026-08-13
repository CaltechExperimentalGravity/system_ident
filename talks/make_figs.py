"""Figure + number generator for the `talks/` reveal.js deck.

Presentation-only glue — NOT package API. It reuses the docs house style
(``docs/sysid_plots.py``) and the existing docs demo modules (``method_demo``,
``srm_modal_demo``, ``darm_demo``, ``darm_tv_demo``) plus the package's own
``sos_campaign`` and the ``experiments/rtsfreerun`` rung, so every panel in the
deck is the *same* computation the documentation and the tests run — no parallel
plotting stack and no re-derived physics.

Outputs
-------
``talks/figs/*.svg``     every panel, **SVG** (Git LFS, per the repo graphics rule)
``talks/figs/numbers.json``  every scalar the slides quote, as computed
``talks/_variables.yml`` the same scalars, pre-formatted, consumed by the deck's
                         ``{{< var … >}}`` shortcodes — so no number in the talk is
                         transcribed by hand.

Cheap by default
----------------
**Nothing is recomputed if its SVG and its numbers are already on disk.** A full
run with everything cached is seconds; only ``--force`` (or a deleted figure)
re-runs a campaign. The expensive things are deliberately sourced from artifacts
the repo *already* produced: the DARM drift panels are rebuilt from the executed
``docs/_freeze`` of example 13 rather than re-driving a ~20-minute campaign, the
compiled-twin rung keeps its own log, and the 40m SOS campaign caches to
``figs/sos_recovery.json``. Rendering the deck itself executes no code at all.

Usage (always through the env)::

    conda run -n sysid python talks/make_figs.py             # every group (cached)
    conda run -n sysid python talks/make_figs.py method sos  # selected groups
    conda run -n sysid python talks/make_figs.py --force srm # force a recompute

Groups
------
``method``  the P&S argument on the double pendulum / high-Q resonance: the
            problem, the three excitations, leakage, swept-sine cost,
            Fisher-optimal design, convergence, the closed loop, and the CRB
            pull test.  (local; no twin)
``twin``    the compiled advligorts ``x1hsts`` front-end rung A1/A2 — runs
            ``experiments/rtsfreerun/run_hsts.py`` and harvests its SVG + numbers.
``sos``     the 40m SOS 6-DOF campaign (``system_ident.sos_campaign``): every
            recovered mode against its Cramér–Rao σ.  (local; ~10 min once, then
            cached in ``figs/sos_recovery.json``)
``srm``     the SRM HSTS closed-loop rank-1 modal fit and the 0.672/0.676 Hz
            *spatial* doublet, from the campaign cache the experiment already
            wrote.  (needs the compiled ``x1hsts6dof`` twin)
``darm``    DARM calibration: response R(f) ± CRB, the hierarchical actuation
            hand-off and swept-vs-multisine (computed here, ~1 min), plus the
            drift-tracking panels **reused from ``docs/_freeze``** (example 13 was
            already executed — no campaign is re-run).

Each group runs in its **own subprocess**: rtsfreerun permits exactly one
compiled model per interpreter, so ``twin`` (x1hsts) and ``srm`` (x1hsts6dof)
cannot share one.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_TALKS = Path(__file__).resolve().parent
_ROOT = _TALKS.parent
_DOCS = _ROOT / "docs"
for _p in (_DOCS, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FIGS = _TALKS / "figs"
NUMBERS = FIGS / "numbers.json"
VARIABLES = _TALKS / "_variables.yml"


# ── plumbing ────────────────────────────────────────────────────────────────
def _load_numbers() -> dict:
    return json.loads(NUMBERS.read_text()) if NUMBERS.exists() else {}


def _save_numbers(d: dict) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    NUMBERS.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")


def _write(fig, name: str, *, width=1280, height=None) -> Path:
    """Export one plotly figure to SVG — vector only; the repo forbids raster plots."""
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / f"{name}.svg"
    fig.write_image(out, format="svg", width=width,
                    height=int(height or fig.layout.height or 600))
    print(f"  wrote {out.relative_to(_ROOT)}")
    return out


def _need(name: str, force: bool) -> bool:
    return force or not (FIGS / f"{name}.svg").exists()


# ── group: method — the P&S argument (all local) ────────────────────────────
def group_method(nums: dict, force: bool) -> None:
    if not force and "head_F" in nums and not any(
            _need(n, False) for n in ("problem", "excitations", "leakage", "swept",
                                      "optimal-design", "convergence", "closed-loop",
                                      "crb-pull")):
        print("  [cached] method figures + numbers already present")
        return

    import method_demo as md

    if _need("problem", force):
        _write(md.problem_fig(height=560), "problem")
    if _need("excitations", force):
        _write(md.excitations_fig(height=660), "excitations")

    if _need("leakage", force) or "leak_rmse_ratio" not in nums:
        c = md.leakage_campaign()
        v = md.leakage_variance()
        _write(md.leakage_bode_fig(c, height=560), "leakage")
        nums.update(
            leak_Q=float(md._LEAK.Q),
            leak_T_s=float(c.T),
            leak_bins_per_linewidth=float(v.linewidth_bins),
            leak_ms_bias_pct=float(v.ms_bias * 100.0),
            leak_w_bias_pct=float(v.w_bias * 100.0),
            leak_rmse_ratio=float(v.rmse_ratio),
            leak_n_seeds=int(v.n_seeds),
        )

    if _need("swept", force) or "sweep_cover_factor" not in nums:
        s = md.sweep_campaign()
        _write(md.sweep_fig(s, height=540), "swept")
        nums.update(
            sweep_n_bins=int(s.n_bins), sweep_n_pts=int(s.n_pts),
            sweep_T_s=float(s.T), sweep_T_used_s=float(s.T_used),
            sweep_cover_h=float(s.t_cover / 3600.0),
            sweep_cover_factor=float(s.cover_factor),
        )

    if _need("optimal-design", force) or "head_F" not in nums:
        h = md.headline_campaign()
        _write(md.headline_design_fig(h, height=560), "optimal-design")
        _write(md.headline_convergence_fig(h, height=460), "convergence")
        nums.update(
            head_F=float(h.F),
            head_sigma_flat=float(h.sigma_flat), head_sigma_opt=float(h.sigma_opt),
            head_sigma_match=float(h.sigma_match),
            head_sigma_ratio=float(h.sigma_flat / h.sigma_opt),
            head_ttt_flat=(int(h.ttt_flat) if h.ttt_flat else -1),
            head_ttt_opt=(int(h.ttt_opt) if h.ttt_opt else -1),
            head_rms_drop=float(h.rms_drop), head_target=float(h.target),
            head_t_pass_s=float(h.t_pass),
        )

    if _need("closed-loop", force) or "cl_rel_pct" not in nums:
        c = md.closed_loop_campaign()
        _write(md.closed_loop_fig(c, height=560), "closed-loop")
        nums.update(cl_rel_pct=float(c.rel * 100.0),
                    cl_suppression=float(c.suppression),
                    cl_stable=bool(c.stable))

    if _need("crb-pull", force) or "pull_emp_std" not in nums:
        p = md.crb_pull_campaign()
        _write(md.crb_pull_fig(p, height=460), "crb-pull")
        nums.update(pull_emp_std=float(p.emp_std), pull_emp_mean=float(p.emp_mean),
                    pull_n_seeds=int(p.n_seeds))


# ── group: twin — the compiled advligorts x1hsts front end ──────────────────
_TWIN_RE = {
    "twin_oracle_vs_sos": r"oracle vs realized SOS\s*:\s*max rel = ([\deE\.\+\-]+)",
    "twin_noiseoff_median": r"noise-off P&S recovery\s*:\s*median = ([\deE\.\+\-]+)",
}


def group_twin(nums: dict, force: bool) -> None:
    """Run the canonical A1/A2 rung against the compiled ``x1hsts`` model.

    ``experiments/rtsfreerun/run_hsts.py`` is the repo's own reproducible demo; it
    writes ``hsts_recovery.svg`` next to itself. We run it, keep its log next to the
    figures as provenance, and harvest the numbers the slides quote.
    """
    exp = _ROOT / "experiments" / "rtsfreerun" / "run_hsts.py"
    svg = exp.parent / "hsts_recovery.svg"
    log = FIGS / "hsts_run.log"
    FIGS.mkdir(parents=True, exist_ok=True)

    if force or not log.exists() or not svg.exists():
        print("  running experiments/rtsfreerun/run_hsts.py (compiled x1hsts) …")
        r = subprocess.run([sys.executable, str(exp)], capture_output=True, text=True)
        log.write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            print(f"  [skip] twin rung failed (rc={r.returncode}); see {log}")
            return
    txt = log.read_text()

    for key, pat in _TWIN_RE.items():
        m = re.search(pat, txt)
        if m:
            nums[key] = float(m.group(1))
    passes = re.findall(r"^\s+(\d+)\s+(\d+)\s+([\deE\.\+\-]+)\s+([\d\.]+)\s*$",
                        txt, flags=re.M)
    if passes:
        nums["twin_passes"] = [{"pass": int(p), "peak": float(pk),
                                "median_rel": float(md), "frac": float(fr)}
                               for p, pk, md, fr in passes]
        nums["twin_peak_counts"] = float(max(p["peak"] for p in nums["twin_passes"]))
        nums["twin_frac_first"] = nums["twin_passes"][0]["frac"]
        nums["twin_frac_last"] = nums["twin_passes"][-1]["frac"]
        nums["twin_median_rel_last"] = nums["twin_passes"][-1]["median_rel"]
    modes = re.findall(r"^\s+([\d\.]+) Hz\s+Q=\s*([\d\.]+)\s+\(truth\s+([\d\.]+) Hz"
                       r"\s+Q=\s*([\d\.]+)\)\s*$", txt, flags=re.M)
    if modes:
        nums["twin_modes"] = [{"f0": float(a), "Q": float(b),
                               "f0_true": float(c), "Q_true": float(d)}
                              for a, b, c, d in modes]
        nums["twin_worst_f0_pct"] = float(max(
            abs(m["f0"] - m["f0_true"]) / m["f0_true"] for m in nums["twin_modes"]) * 100)
    nums["twin_coil_limit"] = 30000.0        # COIL_DRIVER_LIMIT in the twin scenario

    if svg.exists() and _need("hsts-recovery", force):
        shutil.copyfile(svg, FIGS / "hsts-recovery.svg")
        print(f"  wrote {(FIGS / 'hsts-recovery.svg').relative_to(_ROOT)}")


# ── group: 40m SOS 6-DOF (Suspensions) ──────────────────────────────────────
def group_sos(nums: dict, force: bool) -> None:
    import plotly.graph_objects as go

    import sysid_plots as sp
    from system_ident import sos_campaign as sc

    cache = FIGS / "sos_recovery.json"
    if force or not cache.exists():
        print("  running the 6-DOF SOS campaign (≈10 min) …")
        FIGS.mkdir(parents=True, exist_ok=True)
        rows = [{k: (v if isinstance(v, str) else float(v))
                 for k, v in vars(r).items()} for r in sc.run_full_recovery()]
        cache.write_text(json.dumps(rows, indent=2) + "\n")
    rows = json.loads(cache.read_text())

    nums["sos_worst_sigma"] = float(max(max(r["n_sigma_f0"], r["n_sigma_Q"]) for r in rows))
    nums["sos_worst_f0_pct"] = float(
        max(abs(r["f0"] - r["f0_true"]) / r["f0_true"] for r in rows) * 100.0)
    nums["sos_n_modes"] = len(rows)
    nums["sos_modes"] = [{k: r[k] for k in
                          ("dof", "f0", "Q", "f0_std", "Q_std", "f0_true", "Q_true",
                           "n_sigma_f0", "n_sigma_Q")} for r in rows]

    if not _need("sos-crb", force):
        return
    order = sorted(rows, key=lambda r: r["f0_true"])
    labels = [f"{r['dof']}<br>{r['f0_true']:.3f} Hz" for r in order]
    pull_f0 = [(r["f0"] - r["f0_true"]) / r["f0_std"] for r in order]
    pull_q = [(r["Q"] - r["Q_true"]) / r["Q_std"] for r in order]

    fig = go.Figure()
    for lo, hi, col in ((-1, 1, "rgba(31,138,192,0.16)"), (-2, 2, "rgba(31,138,192,0.07)")):
        fig.add_hrect(y0=lo, y1=hi, line_width=0, fillcolor=col, layer="below")
    fig.add_hline(y=0.0, line_color="rgba(27,39,51,0.45)", line_width=1.4)
    fig.add_scatter(x=labels, y=pull_f0, mode="markers", name="f₀ pull  (f̂₀−f₀)/σ_CRB",
                    marker=dict(color=sp.SKY, size=sp.MK_BIG, symbol="circle",
                                line=dict(color="white", width=1.4)))
    fig.add_scatter(x=labels, y=pull_q, mode="markers", name="Q pull  (Q̂−Q)/σ_CRB",
                    marker=dict(color=sp.GOLD, size=sp.MK_BIG, symbol="diamond",
                                line=dict(color="white", width=1.4)))
    lim = max(3.0, 1.2 * max(abs(v) for v in pull_f0 + pull_q))
    fig.update_yaxes(title_text="error in units of the CRB σ", range=[-lim, lim],
                     zeroline=False)
    fig.update_xaxes(title_text="40m SOS rigid-body mode  (DOF, oracle f₀)")
    fig.update_layout(title="40m SOS, 6 DOF: every recovered mode lands inside "
                            f"{nums['sos_worst_sigma']:.2f}σ of the Cramér–Rao bound")
    _write(sp.style(fig, height=520), "sos-crb")


# ── group: SRM HSTS modal fit + the spatial doublet (Suspensions) ───────────
def group_srm(nums: dict, force: bool) -> None:
    # Early-out BEFORE importing the demo: `srm_modal_demo` builds the compiled
    # x1hsts6dof model and re-runs the offline modal fit at import/first call
    # (~3 min). Nothing to do if the figures and numbers are already on disk.
    if not force and "srm_n_modes" in nums and not any(
            _need(n, False) for n in ("srm-diag", "srm-modal", "srm-doublet")):
        print("  [cached] srm figures + numbers already present")
        return

    import srm_modal_demo as sm

    h = sm.headline()
    b = sm.doublet()
    nums.update(
        srm_diag_rel_pct=float(h.diag_rel * 100.0),
        srm_n_modes=int(h.n_modes), srm_n_good=int(h.n_good),
        srm_n_wellsep=int(h.n_wellsep),
        srm_q_med_pct=float(h.q_med), srm_df_med_pct=float(h.df_med),
        srm_used_df=float(h.used_df), srm_used_period_s=float(h.used_period),
        srm_dof=int(h.dof),
        srm_doublet_df_mhz=float(b.df_hz * 1e3),
        srm_doublet=[{"plane": lbl,
                      "f0_oracle": float(b.oracle[lbl].f0),
                      "Q_oracle": float(b.oracle[lbl].Q),
                      "f0": float(b.planes[lbl].f0),
                      "f0_std": float(b.planes[lbl].f0_std),
                      "Q": float(b.planes[lbl].Q),
                      "Q_std": float(b.planes[lbl].Q_std)}
                     for _, lbl in sm._PLANES],
        srm_collapsed={"f0": float(b.collapsed.f0), "Q": float(b.collapsed.Q)},
    )
    if _need("srm-diag", force):
        _write(sm.diag_recovery_fig("L", height=540), "srm-diag")
    if _need("srm-modal", force):
        _write(sm.modal_recovery_fig(height=520), "srm-modal")
    if _need("srm-doublet", force):
        _write(sm.doublet_resolved_fig(height=560), "srm-doublet")


# ── group: DARM calibration (Calibration) ───────────────────────────────────
def group_darm(nums: dict, force: bool) -> None:
    if not force and "darm_fcc" in nums and "darm_sweep_cover_h" in nums and not any(
            _need(n, False) for n in ("darm-response", "darm-hierarchical",
                                      "darm-sweep")):
        print("  [cached] darm calibration figures + numbers already present")
        return group_darm_tv(nums, force)

    import darm_demo as dd

    if _need("darm-response", force) or "darm_fcc" not in nums:
        a = dd.pcal_audit(seed=1)
        _write(dd.response_envelope_fig(a, height=520), "darm-response")
        nums.update(
            darm_fcc=float(a.loop.f_cc), darm_fcc_fit=float(a.fit["f_cc"]),
            darm_fcc_sigma=float(a.sigma["f_cc"]),
            darm_tau_us=float(a.loop.tau * 1e6),
            darm_tau_fit_us=float(a.fit["tau"] * 1e6),
            darm_tau_sigma_us=float(a.sigma["tau"] * 1e6),
        )
        d = dd.actuation_campaign(seed=2)
        nums["darm_actuation"] = [{"stage": n, "kappa_true": float(tk),
                                   "kappa": float(k), "sigma": float(ks)}
                                  for (n, tk, k, ks) in d.rows]
        nums["darm_kappa_worst_pct"] = float(max(
            abs(k - tk) / tk for (_n, tk, k, _s) in d.rows) * 100.0)

    if _need("darm-hierarchical", force):
        _write(dd.hierarchical_actuation(height=540), "darm-hierarchical")

    if _need("darm-sweep", force) or "darm_sweep_cover_h" not in nums:
        c = dd.comparison(seed=0)
        _write(dd.comparison_fig(c, height=520), "darm-sweep")
        nums.update(darm_sweep_cover_h=float(c.t_cover / 3600.0),
                    darm_sweep_T_s=float(c.T), darm_sweep_n_bins=int(c.n_bins),
                    darm_sweep_n_pts=int(c.n_pts))

    group_darm_tv(nums, force)


# ── DARM drift: reuse the repo's already-executed example-13 results ────────
_FREEZE_13 = (_ROOT / "docs" / "_freeze" / "examples" / "13-darm-drift-tracking"
              / "execute-results" / "html.json")

#: figure title fragment → deck figure name, in the frozen page's own order.
_TV_FIGS = {
    "drifting ESD strength": "darm-drift",
    "Tracking error vs the Cram": "darm-tracking",
    "Three parameters drifting at once": "darm-joint",
    "A-optimal cal lines": "darm-callines",
}


def group_darm_tv(nums: dict, force: bool) -> None:
    """Harvest the drift-tracking panels + numbers from `docs/_freeze`.

    The drifting-κ campaign is ~20 min of simulation and the repo has ALREADY run
    it: `docs/examples/13-darm-drift-tracking.qmd` is `freeze: true` and its
    executed output — Plotly figure JSON and the resolvability table — is committed
    under `docs/_freeze`. Re-deriving it for a slide deck would be pure waste, so we
    rebuild the figures from that frozen JSON (seconds) and parse the numbers out of
    the same artifact. Re-running the campaign is `quarto render docs/examples/13…`
    with the freeze invalidated — a docs job, not a talk job.
    """
    import plotly.io as pio

    import darm_tv_demo as dtv          # constants only — no campaign is run here

    if not _FREEZE_13.exists():
        print("  [skip] darm drift: docs/_freeze for example 13 is missing")
        return
    md = json.loads(_FREEZE_13.read_text())["result"]["markdown"]

    wanted = {k: v for k, v in _TV_FIGS.items() if _need(v, force)}
    if wanted:
        specs = re.findall(r"Plotly\.newPlot\(\s*\"[^\"]+\",\s*(\[.*?\]),\s*(\{.*?\}),"
                           r"\s*\{\"responsive\"", md, flags=re.S)
        for data, layout in specs:
            lay = json.loads(layout)
            title = lay.get("title") or {}
            title = title.get("text", "") if isinstance(title, dict) else str(title)
            for frag, name in list(wanted.items()):
                if frag in title:
                    fig = pio.from_json(json.dumps({"data": json.loads(data),
                                                    "layout": lay}))
                    _write(fig, name, height=lay.get("height") or 520)
                    wanted.pop(frag)
        for frag in wanted:
            print(f"  [warn] frozen figure not found: {frag!r}")

    # the resolvability table, straight out of the same frozen artifact
    def cell(label, col=1):
        m = re.search(r"^\|\s*" + re.escape(label) + r"[^|]*\|([^|]*)\|([^|]*)\|",
                      md, flags=re.M)
        return (m.group(col).strip() if m else None)

    got = {
        "darm_drift_amp_pct": cell("injected drift amplitude", 2),
        "darm_snap_sigma_pct": cell("per-snapshot", 2),
        "darm_track_sigma_pct": cell("tracking σ_κ", 2),
        "darm_resolve_ratio": cell("resolve ratio", 1),
        "darm_local_stat_pct": cell("local-stationarity error", 1),
    }
    for key, raw in got.items():
        if raw is None:
            print(f"  [warn] frozen resolvability row missing for {key}")
            continue
        m = re.search(r"[-+]?\d*\.?\d+", raw)
        if m:
            nums[key] = float(m.group(0))
    nums["darm_span_min"] = float(dtv.TSPAN / 60.0)
    nums["darm_n_snap"] = int(dtv.N_SNAP)


# ── group: time domain — what the drive and the optic actually do ──────────
# All local (no twin). Dynamics go through python-control; the repo forbids
# hand-rolled state-space / c2d / simulation in numpy.
TD_FS, TD_NPERSEG, TD_NPER = 32.0, 2048, 6
TD_MODES = [(0.67, 300.0), (1.00, 250.0), (1.98, 180.0)]   # HSTS-like, undamped
TD_DAMPED = [(0.67, 6.0), (1.00, 7.0), (1.98, 9.0)]        # dampers engaged
TD_GAIN = 300.0
TD_DAC = 30000.0        # coil-driver count limit (the ceiling drive design respects)


def _td_plant(modes, gain=TD_GAIN):
    """Resonant suspension-like plant as a python-control transfer function."""
    import control
    G = control.tf([gain], [1.0])
    for f0, Q in modes:
        w0 = 2.0 * np.pi * f0
        G = G * control.tf([w0 ** 2], [1.0, w0 / Q, w0 ** 2])
    return G


def _td_sim(modes, u, fs, gain=TD_GAIN):
    """ZOH-discretise and drive the plant — python-control end to end."""
    import control
    Gd = control.c2d(_td_plant(modes, gain), 1.0 / fs, method="zoh")
    t = np.arange(len(u)) / fs
    res = control.forced_response(Gd, T=t, U=u)
    return t, np.asarray(res.outputs).ravel()


def group_td(nums: dict, force: bool) -> None:
    """Time-domain panels: the drive, the periodicity, the ringdown, headroom."""
    names = ("td-drive", "td-periods", "td-ringdown", "td-response", "td-headroom")
    if not force and "td_period_spread_pow" in nums and not any(
            _need(n, False) for n in names):
        print("  [cached] time-domain figures + numbers already present")
        return

    import plotly.graph_objects as go
    import sysid_plots as sp
    from plotly.subplots import make_subplots

    from system_ident.excitation import multisine_from_psd

    fa = np.fft.rfftfreq(TD_NPERSEG, 1 / TD_FS)
    band = (fa >= 0.3) & (fa <= 5.0)
    freq = fa[band]
    T = TD_NPERSEG / TD_FS
    nums.update(td_period_s=float(T), td_df_hz=float(TD_FS / TD_NPERSEG),
                td_n_periods=int(TD_NPER), td_fs=float(TD_FS),
                td_dac=float(TD_DAC))

    # A concentrated (near-optimal) drive: power on the modes. NOT flat/broadband.
    Pxx_opt = np.zeros_like(freq)
    for f0, _ in TD_MODES:
        Pxx_opt += np.exp(-0.5 * ((freq - f0) / 0.05) ** 2)
    Pxx_opt *= 1.0 / np.trapezoid(Pxx_opt, freq)
    drive = multisine_from_psd(Pxx_opt, TD_FS, TD_NPERSEG, TD_NPER, freq,
                               seed=np.random.default_rng(0), t_ramp=6.0)
    drive = drive / np.max(np.abs(drive)) * (0.33 * TD_DAC)   # 1/3 of the DAC range

    # 1 ── the drive itself: full record with ramps, and one period zoomed
    if _need("td-drive", force):
        t = np.arange(len(drive)) / TD_FS
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                            subplot_titles=[
                                f"<b>Full record</b> — {TD_NPER} periods × {T:.0f} s, "
                                "Tukey ramp on and off",
                                "<b>One period</b> — the same waveform repeats exactly"])
        fig.add_trace(go.Scatter(x=t, y=drive, mode="lines",
                                 line=dict(color=sp.SKY, width=1.0),
                                 name="drive"), row=1, col=1)
        for lim in (TD_DAC, -TD_DAC):
            fig.add_hline(y=lim, line=dict(color=sp.RED, width=2, dash="dash"),
                          row=1, col=1)
        fig.add_annotation(x=t[-1], y=TD_DAC, text="coil-driver limit", showarrow=False,
                           yshift=12, xanchor="right", font=dict(color=sp.RED,
                                                                 size=sp.SZ_ANNOT),
                           row=1, col=1)
        one = drive[TD_NPERSEG * 2:TD_NPERSEG * 3]
        fig.add_trace(go.Scatter(x=np.arange(len(one)) / TD_FS, y=one, mode="lines",
                                 line=dict(color=sp.GOLD, width=1.6),
                                 name="period 3"), row=2, col=1)
        fig.update_xaxes(title_text="time [s]", row=2, col=1)
        fig.update_yaxes(title_text="drive [cts]", row=1, col=1)
        fig.update_yaxes(title_text="drive [cts]", row=2, col=1)
        _write(sp.style(fig, height=620), "td-drive")

    # 2 ── periodicity: successive periods lie on top of each other
    if _need("td-periods", force):
        _, y = _td_sim(TD_DAMPED, drive, TD_FS)
        nskip = 2                                  # let the transient die
        tp = np.arange(TD_NPERSEG) / TD_FS
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                            subplot_titles=[
                                "<b>Steady state</b> — periods overlaid, identical",
                                "<b>Including the transient</b> — period 1 is not"])
        for p in range(nskip, TD_NPER):
            fig.add_trace(go.Scatter(x=tp, y=y[p * TD_NPERSEG:(p + 1) * TD_NPERSEG],
                                     mode="lines", line=dict(width=1.1),
                                     opacity=0.85, showlegend=False), row=1, col=1)
        for p in range(0, 3):
            fig.add_trace(go.Scatter(x=tp, y=y[p * TD_NPERSEG:(p + 1) * TD_NPERSEG],
                                     mode="lines", line=dict(width=1.4),
                                     name=f"period {p + 1}"), row=1, col=2)
        for c in (1, 2):
            fig.update_xaxes(title_text="time within period [s]", row=1, col=c)
        fig.update_yaxes(title_text="response", row=1, col=1)
        _write(sp.style(fig, height=470), "td-periods")
        seg = np.array([y[p * TD_NPERSEG:(p + 1) * TD_NPERSEG]
                        for p in range(nskip, TD_NPER)])
        frac = float(np.max(np.std(seg, axis=0)) / np.max(np.abs(seg)))
        nums["td_period_spread_pct"] = frac * 100
        # "1 part in 10^N of full scale" — readable on a slide, unlike 8.6e-10 %
        nums["td_period_spread_pow"] = int(np.floor(-np.log10(frac)))
        nums["td_skip_periods"] = int(nskip)

    # 3 ── ringdown: why record length is set by Q, and what damping buys
    if _need("td-ringdown", force):
        import control
        tt = np.linspace(0, 240.0, int(240.0 * TD_FS))
        fig = go.Figure()
        for modes, col, nm in ((TD_MODES, sp.GRAY, "loops open  (Q≈300)"),
                               (TD_DAMPED, sp.SKY, "dampers engaged  (Q≈6)")):
            r = control.impulse_response(_td_plant(modes), T=tt)
            y = np.asarray(r.outputs).ravel()
            fig.add_trace(go.Scatter(x=tt, y=y / np.max(np.abs(y)), mode="lines",
                                     line=dict(color=col, width=1.6), name=nm))
        tau = TD_MODES[0][1] / (np.pi * TD_MODES[0][0])
        fig.add_vline(x=tau, line=dict(color=sp.RED, width=2, dash="dot"))
        fig.add_annotation(x=tau, y=1.0, text=f"τ = Q/πf₀ ≈ {tau:.0f} s (undamped)",
                           showarrow=False, xshift=6, xanchor="left",
                           font=dict(color=sp.RED, size=sp.SZ_ANNOT))
        fig.update_xaxes(title_text="time [s]")
        fig.update_yaxes(title_text="impulse response  (normalised)")
        _write(sp.style(fig, height=460), "td-ringdown")
        nums.update(td_tau_open_s=float(tau),
                    td_tau_damped_s=float(TD_DAMPED[0][1] / (np.pi * TD_DAMPED[0][0])))

    # 4 ── drive and motion together: coaxing, not slamming
    if _need("td-response", force):
        t, y = _td_sim(TD_DAMPED, drive, TD_FS)
        fig = sp.timeseries(
            t, [("optimal multisine", drive, sp.GOLD)],
            [("optic motion", y, sp.SKY)],
            titles=["<b>Drive</b> — concentrated multisine, ramped on and off",
                    "<b>Optic motion</b> — steady periodic response after the transient"],
            height=470, drive_unit="drive [cts]", motion_unit="motion [a.u.]")
        _write(fig, "td-response")

    # 5 ── headroom: the concentrated drive vs a flat one at equal Fisher weight
    if _need("td-headroom", force) or "td_peak_frac" not in nums:
        flat = multisine_from_psd(np.ones_like(freq) / (freq[-1] - freq[0]),
                                  TD_FS, TD_NPERSEG, TD_NPER, freq,
                                  seed=np.random.default_rng(0), t_ramp=6.0)
        flat = flat / np.std(flat) * np.std(drive)     # same RMS, same power budget
        t = np.arange(len(drive)) / TD_FS
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=flat, mode="lines", name="flat-spectrum drive",
                                 line=dict(color=sp.GRAY, width=1.0)))
        fig.add_trace(go.Scatter(x=t, y=drive, mode="lines", name="optimal drive",
                                 line=dict(color=sp.GOLD, width=1.2)))
        for lim in (TD_DAC, -TD_DAC):
            fig.add_hline(y=lim, line=dict(color=sp.RED, width=2, dash="dash"))
        fig.update_xaxes(title_text="time [s]")
        fig.update_yaxes(title_text="drive [cts]")
        _write(sp.style(fig, height=440), "td-headroom")
        nums.update(
            td_peak_opt=float(np.max(np.abs(drive))),
            td_peak_flat=float(np.max(np.abs(flat))),
            td_peak_frac=float(np.max(np.abs(drive)) / TD_DAC * 100),
            td_crest_opt=float(np.max(np.abs(drive)) / np.std(drive)),
            td_crest_flat=float(np.max(np.abs(flat)) / np.std(flat)),
        )


# ── group: A3/A4 closed-loop tensor recovery, measured here ─────────────────
# Same knobs as tests/test_rtsfreerun_6dof.py so the deck's numbers and the
# test's assertions are the *same* measurement, not two similar ones.
L_FS, L_NPERSEG, L_NPERIODS, L_NPASSES = 256.0, 4096, 6, 2
L_COUPLINGS = [("L", "P"), ("P", "L"), ("R", "Y"), ("Y", "R")]


def _loop_grid():
    fa = np.fft.rfftfreq(L_NPERSEG, 1 / L_FS)
    band = (fa >= 0.3) & (fa <= 8.0)
    return band, fa[band]


def group_loops(nums: dict, force: bool) -> None:
    """Track A3/A4 measured live on the compiled ``x1hsts6dof``.

    Replaces the recorded "< 0.1 %" bullets: the open- and closed-loop 6×6
    tensors are measured here, scored against the analytic state-space oracle,
    and every quoted percentage comes out of this run. ~6 min once, then cached.
    """
    if not force and "loops_diag_closed_max_pct" in nums and not any(
            _need(n, False) for n in ("loops-tensor", "td-junction", "td-cancellation")):
        print("  [cached] A3/A4 tensor numbers + figures already present")
        return

    sys.path.insert(0, str(_ROOT / "experiments" / "rtsfreerun"))
    import hsts6dof_loop as h6

    if not h6.deps_available():
        print("  [skip] x1hsts6dof / twin archives not present on this machine")
        return

    import plotly.graph_objects as go
    import sysid_plots as sp
    from plotly.subplots import make_subplots

    band, freq = _loop_grid()
    model = h6.HSTS6DOF()
    kw = dict(fs=L_FS, nperseg=L_NPERSEG, n_periods=L_NPERIODS, band=band,
              freq=freq, n_passes=L_NPASSES, warmup_s=32.0, seed=0)
    print("  measuring open-loop tensor …", flush=True)
    H_open = model.measure_tensor(closed=False, **kw)
    print("  measuring closed-loop tensor (all six dampers engaged) …", flush=True)
    H_closed = model.measure_tensor(closed=True, **kw)

    M_open = model.rel_err_tensor(H_open, freq)
    M_closed = model.rel_err_tensor(H_closed, freq)
    d_open, d_closed = np.diag(M_open), np.diag(M_closed)
    G = model.oracle_tensor(freq)
    coup = {f"{o}<-{i}": float(np.median(
                np.abs(H_open[model.dofs.index(o), model.dofs.index(i)]
                       - G[:, model.dofs.index(o), model.dofs.index(i)])
                / np.abs(G[:, model.dofs.index(o), model.dofs.index(i)])))
            for o, i in L_COUPLINGS}

    nums.update(
        loops_dofs=list(model.dofs),
        loops_diag_open_pct=[float(v * 100) for v in d_open],
        loops_diag_closed_pct=[float(v * 100) for v in d_closed],
        loops_diag_open_max_pct=float(d_open.max() * 100),
        loops_diag_open_med_pct=float(np.median(d_open) * 100),
        loops_diag_closed_max_pct=float(d_closed.max() * 100),
        loops_diag_closed_med_pct=float(np.median(d_closed) * 100),
        loops_coupling_pct={k: v * 100 for k, v in coup.items()},
        loops_coupling_max_pct=float(max(coup.values()) * 100),
        loops_coupling_min_pct=float(min(coup.values()) * 100),
        loops_fs=L_FS, loops_nperseg=L_NPERSEG, loops_n_periods=L_NPERIODS,
        loops_n_passes=L_NPASSES,
        loops_df_hz=float(L_FS / L_NPERSEG),
    )

    if _need("loops-tensor", force):
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                            subplot_titles=["<b>Loops open</b> — 6×6 recovery",
                                            "<b>Loops closed</b> — 6×6 recovery"])
        for c, M in ((1, M_open), (2, M_closed)):
            fig.add_trace(go.Heatmap(
                z=np.log10(M * 100), x=model.dofs, y=model.dofs,
                colorscale="Blues_r", zmin=-2, zmax=1, showscale=(c == 2),
                colorbar=dict(title="log₁₀ %err", len=0.9),
                hovertemplate="out %{y} ← in %{x}<br>%{customdata:.3f} %<extra></extra>",
                customdata=M * 100), row=1, col=c)
        fig.update_yaxes(title_text="sensor DOF", autorange="reversed", row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        for c in (1, 2):
            fig.update_xaxes(title_text="drive DOF", row=1, col=c)
        _write(sp.style(fig, height=470), "loops-tensor")

    # -- time domain: the "+−" junction that the sign error lived in ---------
    if _need("td-junction", force):
        from system_ident.excitation import multisine_from_psd
        model.set_loops(True)
        model.reset()
        be = model.backend("L", fs=L_FS, warmup_s=32.0, seed=0, closed=True)
        Pxx = np.full(len(freq), 1.0e7 / (freq[-1] - freq[0]))
        drive = multisine_from_psd(Pxx, L_FS, L_NPERSEG, 2, freq,
                                   seed=np.random.default_rng(0))
        be.inject(model.exc("L"), drive, L_FS)
        chans = [model.exc("L"), model.damp_out("L"), model.plant_in("L"),
                 model.readout("L")]
        seg = be.read(chans, L_NPERSEG * 2 / L_FS)
        be.inject(model.exc("L"), np.zeros_like(drive), L_FS)

        n = int(12.0 * L_FS)                       # 12 s window, mid-record
        s0 = (len(seg[chans[0]]) - n) // 2
        t = np.arange(n) / L_FS
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                            subplot_titles=[
                                "<b>Injected drive</b> — DRIVE_EXC_L (what we command)",
                                "<b>Damper feedback</b> — MC2_M1_DAMP_L_OUT (what the loop adds)",
                                "<b>True plant input</b> — PLANT_IN_L = drive − feedback"])
        for r, (ch, col, nm) in enumerate(
                [(chans[0], sp.SKY, "DRIVE_EXC_L"),
                 (chans[1], sp.ROSE, "DAMP_L_OUT"),
                 (chans[2], sp.GOLD, "PLANT_IN_L")], start=1):
            fig.add_trace(go.Scatter(x=t, y=seg[ch][s0:s0 + n], mode="lines",
                                     line=dict(color=col, width=1.6), name=nm),
                          row=r, col=1)
        fig.update_xaxes(title_text="time [s]", row=3, col=1)
        fig.update_yaxes(title_text="counts", row=2, col=1)
        _write(sp.style(fig, height=620, legend="h"), "td-junction")
        nums["loops_fb_to_drive_rms"] = float(
            np.std(seg[chans[1]]) / np.std(seg[chans[0]]))

    # -- the cancellation proof: reference FRF vs naive drive→sense ----------
    if _need("td-cancellation", force) or "loops_naive_bias_pct" not in nums:
        Hr, Hn, Gd = model.measure_cancellation(
            "L", fs=L_FS, nperseg=L_NPERSEG, n_periods=L_NPERIODS, band=band,
            freq=freq, n_passes=L_NPASSES, warmup_s=32.0, seed=0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freq, y=np.abs(Gd), mode="lines",
                                 line=dict(color=sp.INK, width=3),
                                 name="open-loop plant (oracle)"))
        fig.add_trace(go.Scatter(x=freq, y=np.abs(Hr), mode="markers",
                                 marker=dict(color=sp.SKY, size=sp.MK_DATA),
                                 name="reference FRF  READOUT / PLANT_IN"))
        fig.add_trace(go.Scatter(x=freq, y=np.abs(Hn), mode="markers",
                                 marker=dict(color=sp.GRAY, size=sp.MK_DATA,
                                             symbol="x"),
                                 name="naive FRF  READOUT / DRIVE_EXC"))
        fig.update_xaxes(type="log", title_text="frequency [Hz]")
        fig.update_yaxes(type="log", title_text="|G(f)|")
        _write(sp.style(fig, height=520), "td-cancellation")
        nums.update(
            loops_ref_bias_pct=float(np.median(np.abs(Hr - Gd) / np.abs(Gd)) * 100),
            loops_naive_bias_pct=float(np.median(np.abs(Hn - Gd) / np.abs(Gd)) * 100),
            loops_naive_peak_supp=float(
                np.abs(Gd).max() / np.abs(Hn)[np.argmax(np.abs(Gd))]),
        )


def _whiteness_case(n_modes: int, *, seed: int = 5):
    """Fit a 2-mode rank-1 plant with ``n_modes`` and return (report, per-bin power, freq).

    The synthetic campaign mirrors ``tests/test_mimo_fit.py::synth_openloop`` so the deck
    and the unit tests demonstrate the same thing on the same construction. It is local
    and takes seconds — no twin, no compiled model.
    """
    from system_ident.mimo_modal import Rank1ModalModel
    from system_ident.mimo_fit import (MIMOModalEstimator, initial_theta, validate_fit,
                                       whitened_residual)
    modes = [(0.6, 30.0), (1.6, 40.0)]
    phi = np.array([[1., .3, .2], [.2, 1., .4]])
    psi = np.array([[1., .2, .1], [.1, 1., .3]])
    truth, M = Rank1ModalModel(3, 3, 2), 20
    freq = np.linspace(0.3, 3.0, 120)
    truth.set_reference(freq)
    theta_t = truth.pack(truth.ab_from_modes(modes), phi, psi)
    G = truth.eval(theta_t, freq)
    s = 2j * np.pi * freq
    rng = np.random.default_rng(seed)
    nsa, sigZ, exps = truth.n_sens + truth.n_act, 5e-3, []
    for l in range(truth.n_act):
        U = np.zeros((len(freq), truth.n_act), complex)
        U[:, l] = 1.0 + 0.3 * np.cos(s.imag)
        Y = np.einsum('fij,fj->fi', G, U)
        per = []
        for _ in range(M):
            nY = (rng.standard_normal((len(freq), truth.n_sens))
                  + 1j * rng.standard_normal((len(freq), truth.n_sens))) * sigZ / np.sqrt(2)
            nU = (rng.standard_normal((len(freq), truth.n_act))
                  + 1j * rng.standard_normal((len(freq), truth.n_act))) * sigZ / np.sqrt(2)
            per.append(np.concatenate([Y + nY, U + nU], axis=1))
        per = np.array(per)
        Zb = per.mean(0)
        Cz = np.empty((len(freq), nsa, nsa), complex)
        for k in range(len(freq)):
            d = per[:, k, :] - Zb[k]
            Cz[k] = (d.conj().T @ d) / (M - 1) / M
        exps.append((Zb[:, :truth.n_sens], Zb[:, truth.n_sens:], Cz))

    m = Rank1ModalModel(3, 3, n_modes)
    theta = MIMOModalEstimator(m).fit(exps, freq, initial_theta(m, exps, freq, G)).theta
    rep = validate_fit(m, theta, exps, freq, dof=M, modes_hz=[f for f, _ in modes])
    power = (np.abs(whitened_residual(m, theta, exps, freq)) ** 2).mean(axis=(0, 2))
    return rep, power, freq


def group_whiteness(nums: dict, force: bool) -> None:
    """P&S validation test 3 — the residual whiteness check, on a deliberately
    undermodeled fit (2-mode plant fitted with 1 mode) against a correctly ordered one."""
    if not force and "wh_p_bad" in nums and not _need("whiteness", False):
        print("  [cached] whiteness figure + numbers already present")
        return
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import sysid_plots as sp

    good, p_good, freq = _whiteness_case(2)
    bad, p_bad, _ = _whiteness_case(1)
    wg, wb = good["whiteness"], bad["whiteness"]
    exp_pow = wg["mean_power_expected"]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("<b>Residual autocorrelation</b> — structure left over",
                                        "<b>Residual power per bin</b> — where it sits"))
    for w, name, c in ((wg, "correct order (2 modes)", sp.GREEN),
                       (wb, "undermodeled (1 mode)", sp.RED)):
        fig.add_scatter(x=w["lags"], y=w["acf"], mode="lines+markers", name=name,
                        line=dict(color=c, width=2.4), marker=dict(color=c, size=sp.MK_DATA),
                        row=1, col=1)
    fig.add_scatter(x=wg["lags"], y=wg["acf_bound"], mode="lines", name="95% white-noise band",
                    line=dict(color=sp.GRAY, width=1.8, dash="dash"), row=1, col=1)
    for y, name, c in ((p_good, "correct order (2 modes)", sp.GREEN),
                       (p_bad, "undermodeled (1 mode)", sp.RED)):
        fig.add_scatter(x=freq, y=y, mode="lines", name=name, showlegend=False,
                        line=dict(color=c, width=2.0), row=1, col=2)
    fig.add_hline(y=exp_pow, line=dict(color=sp.GRAY, width=1.8, dash="dash"), row=1, col=2)
    fig.add_vline(x=1.6, line=dict(color=sp.INK, width=1.2, dash="dot"), row=1, col=2)
    fig.add_annotation(x=np.log10(1.6), y=1.0, xref="x2", yref="y2 domain", yanchor="bottom",
                       showarrow=False, font=dict(size=sp.SZ_ANNOT, color=sp.INK),
                       text="the mode the model omits")
    fig.update_xaxes(title_text="lag [excited-line index]", row=1, col=1)
    fig.update_yaxes(type="log", title_text="|autocorrelation|", row=1, col=1)
    fig.update_xaxes(type="log", title_text="frequency [Hz]", row=1, col=2)
    fig.update_yaxes(type="log", title_text="mean |whitened residual|²", row=1, col=2)
    _write(sp.style(fig, height=520), "whiteness", height=520)

    nums.update(
        wh_p_good=float(wg["p_value"]), wh_p_bad=float(wb["p_value"]),
        wh_acf1_good=float(wg["acf"][0]), wh_acf1_bad=float(wb["acf"][0]),
        wh_acf_bound=float(wg["acf_bound"][0]),
        wh_powratio_good=float(wg["power_ratio"]), wh_powratio_bad=float(wb["power_ratio"]),
        wh_costratio_good=float(good["cost_ratio"]), wh_costratio_bad=float(bad["cost_ratio"]),
        wh_worst_lo=float(wb["worst_window_hz"][0]), wh_worst_hi=float(wb["worst_window_hz"][1]),
        wh_missing_hz=1.6, wh_n_bins=int(len(freq)), wh_max_lag=int(wg["lags"][-1]),
    )


GROUPS = {"method": group_method, "twin": group_twin, "sos": group_sos,
          "srm": group_srm, "darm": group_darm, "loops": group_loops,
          "td": group_td, "whiteness": group_whiteness}


# ── _variables.yml — every number the deck quotes, pre-formatted ────────────
def _fmt(n: dict) -> dict:
    """Format the raw scalars for the deck's ``{{< var … >}}`` shortcodes.

    Only keys actually present are emitted, so a partial run still writes a valid
    (smaller) variables file rather than inventing placeholders.
    """
    def g(k, spec, scale=1.0):
        if k not in n:
            return None
        v = n[k] * scale if scale != 1.0 else n[k]
        return format(int(round(v)) if spec == "d" else float(v), spec)

    out = {
        "leak": {"Q": g("leak_Q", ".0f"), "T": g("leak_T_s", ".0f"),
                 "bins": g("leak_bins_per_linewidth", ".1f"),
                 "ms_bias": g("leak_ms_bias_pct", "+.1f"),
                 "w_bias": g("leak_w_bias_pct", "+.0f"),
                 "rmse_ratio": g("leak_rmse_ratio", ".0f"),
                 "seeds": g("leak_n_seeds", "d")},
        "sweep": {"bins": g("sweep_n_bins", "d"), "pts": g("sweep_n_pts", "d"),
                  "T": g("sweep_T_s", ".0f"), "cover_h": g("sweep_cover_h", ".0f"),
                  "factor": g("sweep_cover_factor", ".0f")},
        "head": {"F": g("head_F", ".0f"),
                 "sigma_flat": g("head_sigma_flat", ".3f"),
                 "sigma_opt": g("head_sigma_opt", ".3f"),
                 "sigma_ratio": g("head_sigma_ratio", ".1f"),
                 "ttt_opt": g("head_ttt_opt", "d"),
                 "rms_drop": g("head_rms_drop", ".1f"),
                 "target": g("head_target", ".2f"),
                 "t_pass": g("head_t_pass_s", ".0f")},
        "cl": {"rel": g("cl_rel_pct", ".1f"), "supp": g("cl_suppression", ".0f")},
        "pull": {"std": g("pull_emp_std", ".2f"), "mean": g("pull_emp_mean", "+.2f"),
                 "seeds": g("pull_n_seeds", "d")},
        "twin": {"oracle": g("twin_oracle_vs_sos", ".1e"),
                 "noiseoff": g("twin_noiseoff_median", ".1e"),
                 "frac_first": g("twin_frac_first", ".3f"),
                 "frac_last": g("twin_frac_last", ".3f"),
                 "median_rel_pct": g("twin_median_rel_last", ".3f", 100.0),
                 "peak": g("twin_peak_counts", ".0f"),
                 "limit": g("twin_coil_limit", ".0f"),
                 "worst_f0_pct": g("twin_worst_f0_pct", ".2f")},
        "sos": {"worst_sigma": g("sos_worst_sigma", ".2f"),
                "worst_f0_pct": g("sos_worst_f0_pct", ".3f"),
                "n_modes": g("sos_n_modes", "d")},
        "srm": {"diag": g("srm_diag_rel_pct", ".2g"),
                "n_modes": g("srm_n_modes", "d"), "n_good": g("srm_n_good", "d"),
                "n_wellsep": g("srm_n_wellsep", "d"),
                "q_med": g("srm_q_med_pct", ".1f"),
                "df_med": g("srm_df_med_pct", ".2f"),
                "df_hz": g("srm_used_df", ".5f"),
                "period": g("srm_used_period_s", ".0f"),
                "dof": g("srm_dof", "d"),
                "doublet_mhz": g("srm_doublet_df_mhz", ".1f")},
        "darm": {"fcc": g("darm_fcc", ".0f"), "fcc_fit": g("darm_fcc_fit", ".1f"),
                 "fcc_sigma": g("darm_fcc_sigma", ".1f"),
                 "tau": g("darm_tau_us", ".0f"),
                 "tau_fit": g("darm_tau_fit_us", ".1f"),
                 "tau_sigma": g("darm_tau_sigma_us", ".2f"),
                 "kappa_worst": g("darm_kappa_worst_pct", ".2f"),
                 "sweep_cover_h": g("darm_sweep_cover_h", ".1f"),
                 "sweep_T": g("darm_sweep_T_s", ".0f"),
                 "sweep_bins": g("darm_sweep_n_bins", "d"),
                 "sweep_pts": g("darm_sweep_n_pts", "d"),
                 "drift_amp": g("darm_drift_amp_pct", ".0f"),
                 "span_min": g("darm_span_min", ".0f"),
                 "n_snap": g("darm_n_snap", "d"),
                 "snap_sigma": g("darm_snap_sigma_pct", ".2f"),
                 "track_sigma": g("darm_track_sigma_pct", ".2f"),
                 "resolve": g("darm_resolve_ratio", ".0f"),
                 "local_stat": g("darm_local_stat_pct", ".2f")},
        "loops": {"open_max": g("loops_diag_open_max_pct", ".2f"),
                  "open_med": g("loops_diag_open_med_pct", ".2f"),
                  "closed_max": g("loops_diag_closed_max_pct", ".2f"),
                  "closed_med": g("loops_diag_closed_med_pct", ".2f"),
                  "coup_min": g("loops_coupling_min_pct", ".2f"),
                  "coup_max": g("loops_coupling_max_pct", ".2f"),
                  "ref_bias": g("loops_ref_bias_pct", ".2f"),
                  "naive_bias": g("loops_naive_bias_pct", ".1f"),
                  "naive_supp": g("loops_naive_peak_supp", ".1f"),
                  "fb_rms": g("loops_fb_to_drive_rms", ".2f"),
                  "df_hz": g("loops_df_hz", ".4f"),
                  "n_periods": g("loops_n_periods", "d"),
                  "n_passes": g("loops_n_passes", "d")},
        "td": {"period": g("td_period_s", ".0f"), "df": g("td_df_hz", ".4f"),
               "n_periods": g("td_n_periods", "d"),
               "spread_pow": g("td_period_spread_pow", "d"),
               "skip": g("td_skip_periods", "d"),
               "tau_open": g("td_tau_open_s", ".0f"),
               "tau_damped": g("td_tau_damped_s", ".1f"),
               "peak_frac": g("td_peak_frac", ".0f"),
               "crest_opt": g("td_crest_opt", ".1f"),
               "crest_flat": g("td_crest_flat", ".1f"),
               "dac": g("td_dac", ".0f")},
        "wh": {"p_good": g("wh_p_good", ".2f"),
               "acf1_good": g("wh_acf1_good", ".3f"),
               "acf1_bad": g("wh_acf1_bad", ".2f"),
               "bound": g("wh_acf_bound", ".3f"),
               "powratio_good": g("wh_powratio_good", ".2f"),
               "costratio_good": g("wh_costratio_good", ".2f"),
               "worst_lo": g("wh_worst_lo", ".2f"),
               "worst_hi": g("wh_worst_hi", ".2f"),
               "missing": g("wh_missing_hz", ".1f"),
               "bins": g("wh_n_bins", "d"),
               "max_lag": g("wh_max_lag", "d")},
    }
    # per-stage actuation rows, flattened for the shortcodes
    for row in n.get("darm_actuation", []):
        s = row["stage"].lower()
        out["darm"][f"k_{s}_true"] = f"{row['kappa_true']:.3f}"
        out["darm"][f"k_{s}"] = f"{row['kappa']:.4f}"
        out["darm"][f"k_{s}_sigma"] = f"{row['sigma']:.4f}"
        out["darm"][f"k_{s}_pct"] = (
            f"{abs(row['kappa'] - row['kappa_true']) / row['kappa_true'] * 100:.2f}")
        out["darm"][f"k_{s}_pull"] = (
            f"{abs(row['kappa'] - row['kappa_true']) / row['sigma']:.1f}")

    # per-plane doublet rows, flattened for the shortcodes
    for i, row in enumerate(n.get("srm_doublet", [])):
        out["srm"][f"plane{i}"] = row["plane"]
        out["srm"][f"plane{i}_f0"] = f"{row['f0']:.5f}"
        out["srm"][f"plane{i}_f0_sigma"] = f"{row['f0_std']:.1e}"
        out["srm"][f"plane{i}_Q"] = f"{row['Q']:.2f}"
    return {k: {kk: vv for kk, vv in v.items() if vv is not None}
            for k, v in out.items()}


def write_variables(nums: dict) -> None:
    import yaml
    VARIABLES.write_text(
        "# GENERATED by talks/make_figs.py — every number the deck quotes, as computed.\n"
        "# Do not hand-edit: re-run `conda run -n sysid python talks/make_figs.py`.\n"
        + yaml.safe_dump(_fmt(nums), sort_keys=True, default_flow_style=False))
    print(f"variables -> {VARIABLES.relative_to(_ROOT)}")


def main(argv: list[str]) -> int:
    force = "--force" in argv
    one = None
    if "--one" in argv:
        one = argv[argv.index("--one") + 1]
    names = [a for a in argv if not a.startswith("-") and a != one] or list(GROUPS)

    if one:                                   # child: run exactly one group in-process
        nums = _load_numbers()
        GROUPS[one](nums, force)
        _save_numbers(nums)
        return 0

    unknown = [n for n in names if n not in GROUPS]
    if unknown:
        sys.exit(f"unknown group(s) {unknown}; known: {sorted(GROUPS)}")
    for n in names:                           # parent: one subprocess per group
        print(f"[{n}]", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--one", n]
        if force:
            cmd.append("--force")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  [warn] group {n} exited {rc} — its figures may be missing")
    write_variables(_load_numbers())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
