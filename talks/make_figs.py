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


GROUPS = {"method": group_method, "twin": group_twin, "sos": group_sos,
          "srm": group_srm, "darm": group_darm}


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
