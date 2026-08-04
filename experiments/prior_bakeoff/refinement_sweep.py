"""Refinement-efficiency sweep (loop-based, multiprocess).

The meaningful question once cold-start is solved: with a GOOD prior (±5-20%) and
LOW-SNR measurements, does the Bayesian prior buy *fewer* passes to reach a target
accuracy than prior-ignoring broadband_ls — and at what prior strength? This must
run in the real loop (concentrated/optimal excitation), so each job runs a full
SysIDLoop campaign. Embarrassingly parallel over (mode, prior_strength, case, SNR,
seed) → ProcessPoolExecutor across all cores.

Metric (mode-agnostic): passes to reach f0 within 2% AND Q within 15% of truth.

Run:
    cd <repo> && conda run -n sysid python experiments/prior_bakeoff/refinement_sweep.py
"""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from system_ident.config import RunConfig
from system_ident.loop import SysIDLoop

TRUE = (1.0, 20.0, 100.0)        # f0, Q, gain
MAX_ITER = 10
F0_TOL, Q_TOL = 0.02, 0.15

# good-prior cases (refinement regime): (f0, Q, gain)
CASES = [
    (0.95, 19.0, 95.0), (1.05, 21.0, 105.0),
    (0.90, 18.0, 90.0), (1.10, 22.0, 110.0),
    (0.80, 16.0, 80.0), (1.20, 24.0, 120.0),
]
SNR_LEVELS = {"mod": (3e-3, 3e-3), "low": (1e-2, 1e-2)}      # (sensor_asd, disturbance_asd)
# (mode, prior_uncertainty) — prior_uncertainty ignored for broadband_ls/hybrid
MODES = [
    ("broadband_ls", None),
    ("bayesian", 0.1), ("bayesian", 0.2), ("bayesian", 0.4), ("bayesian", 0.8),
    ("hybrid", None),
]
SEEDS = range(5)


def _cfg(mode, punc, pf, pq, pg, sens, dist):
    strat = {"estimator": "invfreqs", "input_designer": "pintelon_schoukens",
             "n_design_iter": 3, "loop": mode, "exploration": 0.2}
    if mode == "bayesian":
        strat["prior_uncertainty"] = punc
    if mode == "hybrid":
        strat["n_locate"] = 2
        strat["lock_uncertainty"] = 0.15
    return {
        "run": {"excitation_mode": "sequential"},
        "channels": {"excitation": {"POS": "X1:EXC"}, "readback": {"POS": "X1:RSP"}},
        "measurement": {"fs": 32, "freq_min": 0.1, "freq_max": 5.0,
                        "segment_duration": 64.0, "n_segments": 4, "px_total": 1.0, "t_ramp": 4.0},
        "twin": {"sensor_asd": sens, "disturbance_asd": dist,
                 "plant": {"POS": {"resonances": [[TRUE[0], TRUE[1]]], "gain": TRUE[2]}}},
        "priors": {"POS": {"resonances": [[pf, pq]], "gain": pg}},
        "strategy": strat,
        "safety": {"actuator_sat": 1e12, "rms_ceiling": {"POS": 1e12}, "ramp_down_secs": 2.0},
        "stop_criteria": {"uncertainty_target": 1e-12, "max_iter": MAX_ITER},
    }


def _f0Q(snap):
    if "model_f0" in snap:
        return snap["model_f0"], snap["model_Q"]
    den = np.asarray(snap["model_den"], float)
    p = np.roots(den / den[0])
    p = p[p.imag > 0]
    pp = p[np.argmax(np.abs(p.imag))]
    return abs(pp) / (2 * np.pi), abs(pp) / (2 * abs(pp.real))


def _run(job):
    mode, punc, case_idx, snr_key, seed = job
    pf, pq, pg = CASES[case_idx]
    sens, dist = SNR_LEVELS[snr_key]
    rc = RunConfig(raw=_cfg(mode, punc, pf, pq, pg, sens, dist))
    be = rc.build_twin_backend(seed=seed)
    snaps = []
    loop = SysIDLoop(be, rc.build_estimator(), rc.build_designer(),
                     rc.build_watchdog(be), listener=snaps.append)
    loop.run(rc.raw, rc.build_priors(), seed=seed)
    ptt = MAX_ITER + 1
    fe = qe = float("nan")
    for i, s in enumerate(snaps):
        f0, Q = _f0Q(s)
        fe, qe = abs(f0 - 1.0), abs(Q - 20.0) / 20.0
        if ptt > MAX_ITER and fe < F0_TOL and qe < Q_TOL:
            ptt = i + 1
    key = mode if punc is None else f"{mode}[u={punc}]"
    return {"key": key, "snr": snr_key, "ptt": ptt, "f0_err": fe, "q_err": qe,
            "converged": ptt <= MAX_ITER}


def main():
    jobs = [(mode, punc, ci, snr, seed)
            for (mode, punc) in MODES
            for ci in range(len(CASES))
            for snr in SNR_LEVELS
            for seed in SEEDS]
    nw = os.cpu_count() or 4
    print(f"running {len(jobs)} loop campaigns over {nw} workers...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for r in ex.map(_run, jobs, chunksize=2):
            results.append(r)

    for snr in SNR_LEVELS:
        rs_by_key = defaultdict(list)
        for r in results:
            if r["snr"] == snr:
                rs_by_key[r["key"]].append(r)
        print(f"\n=== SNR={snr} (passes-to-target: f0<2%, Q<15%; over {len(CASES)} cases x {len(SEEDS)} seeds) ===")
        print(f"{'mode':18} {'conv_rate':>9} {'mean_ptt*':>10} {'med_f0e':>8} {'med_qe':>7}")
        rows = []
        for key, rs in rs_by_key.items():
            cr = np.mean([r["converged"] for r in rs])
            conv = [r["ptt"] for r in rs if r["converged"]]
            mean_ptt = np.mean(conv) if conv else float("nan")  # mean passes among converged
            rows.append((key, cr, mean_ptt, np.median([r["f0_err"] for r in rs]),
                         np.median([r["q_err"] for r in rs])))
        for key, cr, mp, mf, mq in sorted(rows, key=lambda x: (-x[1], x[2] if x[2] == x[2] else 1e9)):
            print(f"{key:18} {cr:9.2f} {mp:10.2f} {mf:8.3f} {mq:7.2f}")
    print("\n(* mean_ptt = mean passes-to-target among converged runs; lower = more measurement-efficient)")


if __name__ == "__main__":
    main()
