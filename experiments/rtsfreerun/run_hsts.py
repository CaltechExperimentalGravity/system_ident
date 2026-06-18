"""Reproducible HSTS demo: identify the compiled x1hsts drive->sensor plant.

The Pintelon-Schoukens **optimal-excitation** campaign — a broad prior-robust
multisine designed from the (perturbed) prior on pass 1, then point-optimal
multisines from the refined model on later passes; each pass measured by the
leakage-free reference FRF (input X = the after-actuator drive monitor) and folded
in by inverse-variance accumulation + ML refit — driving the *compiled* rtsfreerun
CDS suspension under the twin's own seismic + readout noise, scored against the
analytic oracle. Same campaign the double-pendulum example runs (`run_siso_passes`).

Two rungs (see .llm/roadmap.md Track A and tests/test_rtsfreerun_real_model.py):
  A1  wiring + oracle : scenario init realises the plant; the analytic oracle agrees
                        with the model's loaded SOS; a noise-off pass recovers it.
  A2  open-loop SISO  : with noise on, recover the order-10 HSTS_DRV_TF cascade
                        (5 modes ~0.67-3.78 Hz, Q~50) and match the oracle.

Run:  conda run -n sysid python experiments/rtsfreerun/run_hsts.py
Needs the x1hsts model built into the env (README §"Run against the RTSfreerun
digital twin"). Writes a Bode overlay PNG next to this file.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs"))            # the run_siso_passes P&S campaign

from system_ident.backends import rtsfreerun_oracle as orc
from system_ident.backends.rtsfreerun_adapter import RTSfreerunBackend
from system_ident.config import RunConfig
from sysid_campaign import run_siso_passes        # noqa: E402

CONFIG = ROOT / "src/system_ident/configs/rtsfreerun_hsts.yml"
OUT_SVG = Path(__file__).resolve().parent / "hsts_recovery.svg"   # vector, per the graphics rule
PX_TOTAL = 1.0e7          # drive power budget (counts^2); peak stays < COIL_DRIVER_LIMIT
N_PASSES = 3


def main() -> int:
    if importlib.util.find_spec("x1hsts") is None:
        sys.exit("x1hsts not installed — build it into this env (see README).")
    import yaml
    raw = yaml.safe_load(open(CONFIG))
    rc = RunConfig(raw=raw)
    scen_path = raw["rtsfreerun"]["scenario"]
    if not Path(scen_path).exists():
        sys.exit(f"scenario not found: {scen_path}")
    scen = orc.load_scenario(scen_path)
    oracle = orc.analytic_plant(scen)

    fs = rc.fs
    nper = int(round(raw["measurement"]["segment_duration"] * fs))
    nseg = int(raw["measurement"]["n_segments"])
    fa = np.fft.rfftfreq(nper, 1 / fs)
    band = (fa >= raw["measurement"]["freq_min"]) & (fa <= raw["measurement"]["freq_max"])
    freq = fa[band]
    exc = raw["channels"]["excitation"]["POS"]
    xmon = raw["channels"]["drive"]["POS"]
    rb = raw["channels"]["readback"]["POS"]
    warmup = raw["rtsfreerun"]["warmup_s"]
    Pyy = np.ones_like(freq)

    import x1hsts
    mdl = x1hsts.x1hsts()
    orc.apply_scenario_init(mdl, scen)
    init_modules = sorted({op["fm"] for op in scen.get("init", []) if "fm" in op})

    print("=" * 72)
    print("HSTS drive->sensor plant (analytic oracle):")
    for f0, q in orc.plant_modes(oracle):
        print(f"    f0 = {f0:6.3f} Hz   Q = {q:5.1f}")
    print(f"    sample_rate = {mdl.sample_rate:.0f} Hz  ->  fs = {fs:.0f} Hz "
          f"(x{mdl.sample_rate / fs:.0f} decimation)")
    ff = np.geomspace(raw["measurement"]["freq_min"], raw["measurement"]["freq_max"], 400)
    Ho = oracle.eval(ff)

    def campaign(noise, seed, n_passes):
        mdl.fm_clear_history(*init_modules)        # clean state per rung (one model/process)
        be = RTSfreerunBackend(mdl=mdl, exc_channels={exc: "POS"},
                               readback_channels={rb: "POS"}, noise=noise,
                               fs=fs, warmup_s=warmup, seed=seed)
        prior = orc.prior_from_scenario(scen, perturb=0.08, rng=np.random.default_rng(7))
        return run_siso_passes(be, exc, rb, prior, x_ch=xmon, fs=fs, nperseg=nper,
                               n_periods=nseg, band=band, freq=freq, Pyy=Pyy,
                               px_total=PX_TOTAL, n_passes=n_passes,
                               prior_uncertainty=0.6, seed=seed)

    # ---- A1: noise off -----------------------------------------------------
    Hsos = orc.realized_plant_response(mdl, freq, scen)
    h1 = campaign([], seed=0, n_passes=1)[-1]
    rel1 = np.abs(h1["model"].eval(ff) - Ho) / np.abs(Ho)
    print("\n[A1] wiring + oracle")
    print(f"    recorded length        : {h1['response'].size}  (expected {nper * nseg})")
    print(f"    oracle vs realized SOS : max rel = {np.max(np.abs(Hsos - oracle.eval(freq)) / np.abs(oracle.eval(freq))):.2e}")
    print(f"    noise-off P&S recovery : median = {np.median(rel1):.2e}, p90 = {np.percentile(rel1, 90):.2e}")

    # ---- A2: noise on, optimal-excitation refinement -----------------------
    noise = raw["rtsfreerun"]["noise"]
    hist = campaign(noise, seed=0, n_passes=N_PASSES)
    print("\n[A2] open-loop SISO recovery under seismic + readout noise (P&S optimal)")
    print("    pass   drive-peak   median |H| err vs oracle   frac-uncertainty")
    for h in hist:
        rel = np.abs(h["model"].eval(ff) - Ho) / np.abs(Ho)
        print(f"     {h['pass']:>2}     {h['peak']:>8.0f}        {np.median(rel):.2e}"
              f"                {h['frac']:.3f}")
    fit = hist[-1]["model"]
    print("    recovered modes (f0, Q) vs truth:")
    for (f0, q), (ft, qt) in zip(orc.plant_modes(fit), orc.plant_modes(oracle)):
        print(f"        {f0:6.3f} Hz  Q={q:5.1f}    (truth {ft:6.3f} Hz  Q={qt:5.1f})")

    _save_plot(ff, oracle, fit, freq, hist[-1])
    print(f"\nwrote {OUT_SVG}")
    return 0


def _save_plot(ff, oracle, fit, freq, last):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(8, 9.5))
    ax[0].loglog(ff, np.abs(oracle.eval(ff)), "k-", lw=2, label="analytic oracle")
    ax[0].loglog(ff, np.abs(fit.eval(ff)), "r--", lw=2, label="recovered (ML fit)")
    ax[0].loglog(freq, np.abs(last["H_acc"]), "C0.", ms=4, alpha=0.5, label="measured FRF (accum)")
    ax[0].set_ylabel("|H|"); ax[0].legend()
    ax[0].set_title("HSTS drive->sensor: P&S optimal-excitation recovery under twin noise")
    ax[1].semilogx(ff, np.unwrap(np.angle(oracle.eval(ff))) * 180 / np.pi, "k-", lw=2)
    ax[1].semilogx(ff, np.unwrap(np.angle(fit.eval(ff))) * 180 / np.pi, "r--", lw=2)
    ax[1].set_ylabel("phase [deg]")
    ax[2].loglog(freq, np.sqrt(last["Pxx"]), "C3-", label="final optimal excitation ASD")
    ax[2].set_ylabel(r"$\sqrt{P_{xx}}$"); ax[2].set_xlabel("frequency [Hz]"); ax[2].legend()
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_SVG, format="svg"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
