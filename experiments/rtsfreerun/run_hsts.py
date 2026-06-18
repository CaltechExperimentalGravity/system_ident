"""Reproducible HSTS demo: identify the compiled x1hsts drive->sensor plant.

The proven Pintelon-Schoukens path — periodic multisine -> leakage-free reference
FRF -> maximum-likelihood fit — driving the *compiled* rtsfreerun CDS suspension
under the twin's own seismic + readout noise, scored against the analytic oracle.

Two rungs (see .llm/roadmap.md Track A and tests/test_rtsfreerun_real_model.py):
  A1  wiring + oracle : scenario init realises the plant; the analytic oracle agrees
                        with the model's loaded SOS; a noise-off FRF matches it.
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

from system_ident.backends import rtsfreerun_oracle as orc
from system_ident.backends.rtsfreerun_adapter import RTSfreerunBackend
from system_ident.config import RunConfig
from system_ident.estimators.gml import GMLEstimator
from system_ident.excitation import multisine_from_psd
from system_ident.loop import SysIDLoop

CONFIG = Path(__file__).resolve().parents[2] / "src/system_ident/configs/rtsfreerun_hsts.yml"
DRIVE_PEAK = 6000.0       # counts; well under COIL_DRIVER_LIMIT (30000)
OUT_PNG = Path(__file__).resolve().parent / "hsts_recovery.png"


def _require_model():
    if importlib.util.find_spec("x1hsts") is None:
        sys.exit("x1hsts not installed — build it into this env (see README).")


def main() -> int:
    _require_model()
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
    drv = raw["channels"]["drive"]["POS"]
    rb = raw["channels"]["readback"]["POS"]

    import x1hsts
    mdl = x1hsts.x1hsts()
    orc.apply_scenario_init(mdl, scen)

    print("=" * 70)
    print("HSTS drive->sensor plant (analytic oracle):")
    for f0, q in orc.plant_modes(oracle):
        print(f"    f0 = {f0:6.3f} Hz   Q = {q:5.1f}")
    print(f"    sample_rate = {mdl.sample_rate:.0f} Hz  ->  fs = {fs:.0f} Hz "
          f"(x{mdl.sample_rate / fs:.0f} decimation)")

    def measure(noise, seed):
        be = RTSfreerunBackend(mdl=mdl, exc_channels={exc: "POS"},
                               readback_channels={rb: "POS"}, noise=noise,
                               fs=fs, warmup_s=raw["rtsfreerun"]["warmup_s"], seed=seed)
        d = multisine_from_psd(np.ones_like(freq), fs, nper, nseg, freq, seed=seed)
        d = d / np.max(np.abs(d)) * DRIVE_PEAK
        be.inject(exc, d, fs)
        seg = be.read([rb, drv], nper * nseg / fs)
        H, He, coh = SysIDLoop._estimate_tf_periodic(seg[drv], seg[rb], fs, nper, band)
        return seg, H, He, coh

    # ---- A1: noise off -----------------------------------------------------
    seg, H1, _, coh1 = measure([], seed=0)
    Ho = oracle.eval(freq)
    rel1 = np.abs(np.abs(H1) - np.abs(Ho)) / np.abs(Ho)
    Hsos = orc.realized_plant_response(mdl, freq, scen)
    print("\n[A1] wiring + oracle")
    print(f"    recorded length        : {seg[rb].size}  (expected {nper * nseg})")
    print(f"    oracle vs realized SOS : max rel = {np.max(np.abs(Hsos - Ho) / np.abs(Ho)):.2e}")
    print(f"    noise-off FRF vs oracle: median = {np.median(rel1):.2e}, "
          f"p90 = {np.percentile(rel1, 90):.2e}, coherence = {np.median(coh1):.5f}")

    # ---- A2: noise on, recover --------------------------------------------
    noise = raw["rtsfreerun"]["noise"]
    seg, H2, He2, coh2 = measure(noise, seed=0)
    prior = orc.prior_from_scenario(scen, perturb=0.12, rng=np.random.default_rng(101))
    fit = GMLEstimator().fit(freq, H2, He2, prior)
    ff = np.geomspace(raw["measurement"]["freq_min"], raw["measurement"]["freq_max"], 400)
    rel2 = np.abs(fit.eval(ff) - oracle.eval(ff)) / np.abs(oracle.eval(ff))
    print("\n[A2] open-loop SISO recovery under seismic + readout noise")
    print(f"    drive peak             : {np.max(np.abs(seg[drv])):.0f} counts (limit 30000)")
    print(f"    coherence              : {np.median(coh2):.5f}")
    print(f"    recovered vs oracle    : median = {np.median(rel2):.2e}, p90 = {np.percentile(rel2, 90):.2e}")
    print("    recovered modes (f0, Q) vs truth:")
    rec = orc.plant_modes(fit)
    for (f0, q), (ft, qt) in zip(rec, orc.plant_modes(oracle)):
        print(f"        {f0:6.3f} Hz  Q={q:5.1f}    (truth {ft:6.3f} Hz  Q={qt:5.1f})")

    _save_plot(ff, oracle, fit, freq, H2, coh2)
    print(f"\nwrote {OUT_PNG}")
    return 0


def _save_plot(ff, oracle, fit, freq, H_meas, coh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax[0].loglog(ff, np.abs(oracle.eval(ff)), "k-", lw=2, label="analytic oracle")
    ax[0].loglog(ff, np.abs(fit.eval(ff)), "r--", lw=2, label="recovered (ML fit)")
    ax[0].loglog(freq, np.abs(H_meas), "C0.", ms=4, alpha=0.5, label="measured FRF")
    ax[0].set_ylabel("|H|"); ax[0].legend(); ax[0].set_title("HSTS drive->sensor: P&S recovery under twin noise")
    ax[1].semilogx(ff, np.unwrap(np.angle(oracle.eval(ff))) * 180 / np.pi, "k-", lw=2)
    ax[1].semilogx(ff, np.unwrap(np.angle(fit.eval(ff))) * 180 / np.pi, "r--", lw=2)
    ax[1].set_ylabel("phase [deg]")
    ax[2].semilogx(freq, coh, "C2-"); ax[2].set_ylabel("coherence"); ax[2].set_ylim(0.9, 1.001)
    ax[2].set_xlabel("frequency [Hz]")
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=110); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
