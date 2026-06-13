"""Multiprocess sweep of prior-strategy hyperparameters across all CPU cores.

Embarrassingly parallel: every (strategy, config, case, seed) campaign is
independent, so we fan them out over a ProcessPoolExecutor. This is the right
tool for a single multi-core box (no MPI needed). It complements the live
strategy bake-off by exhaustively + multi-seed sweeping the two most promising
spaces — uniform prior strength (the "weaker prior" hypothesis) and per-parameter
Q/gain anchoring (the Q/gain-collapse diagnosis) — and is the reusable engine for
the future ~12-case benchmark.

Run:
    cd <repo> && conda run -n sysid python experiments/prior_bakeoff/parallel_sweep.py
"""

from __future__ import annotations

import os
import sys
import itertools
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for `import harness`

import numpy as np

import harness
from system_ident.resonator import ResonatorModel
from system_ident.estimators.bayesian import bayesian_update, prior_precision


# --- strategy builders: (strategy, config) -> init_fn(f0,Q,gain)->(model,Lambda) ---
# All step with the default package bayesian_update (picklable, no closures).

def _build_init(strategy: str, cfg: dict):
    if strategy == "baseline":
        def init(f0, Q, g):
            m = ResonatorModel.from_resonances([(f0, Q)], g)
            return m, prior_precision(m, 0.4)
        return init
    if strategy == "weaker":
        U = cfg["U"]
        def init(f0, Q, g):
            m = ResonatorModel.from_resonances([(f0, Q)], g)
            return m, prior_precision(m, U)
        return init
    if strategy == "log_weaker":
        U = cfg["U"]
        def init(f0, Q, g):
            m = ResonatorModel.from_resonances([(f0, Q)], g, log=True)
            return m, prior_precision(m, U)
        return init
    if strategy == "qanchor":
        uf, uq, ug = cfg["uf"], cfg["uq"], cfg["ug"]
        def init(f0, Q, g):
            m = ResonatorModel.from_resonances([(f0, Q)], g)   # params [f0, Q, gain]
            sigma0 = np.array([uf * f0, uq * Q, ug * abs(g)])
            return m, np.diag(1.0 / sigma0 ** 2)
        return init
    if strategy == "log_qanchor":
        uf, uq, ug = cfg["uf"], cfg["uq"], cfg["ug"]
        def init(f0, Q, g):
            m = ResonatorModel.from_resonances([(f0, Q)], g, log=True)  # params [logf0, logQ, log|g|]
            sigma0 = np.array([uf, uq, ug])   # log-space sigmas ~ fractional
            return m, np.diag(1.0 / sigma0 ** 2)
        return init
    raise ValueError(strategy)


def _cfgkey(strategy: str, cfg: dict) -> str:
    if not cfg:
        return strategy
    return strategy + "[" + ",".join(f"{k}={v}" for k, v in cfg.items()) + "]"


def _run_case(job):
    """One campaign: (strategy, cfg, case_idx, seed) -> result dict. Module-level (picklable)."""
    strategy, cfg, case_idx, seed = job
    label, f0, Q, g = harness.CASES[case_idx]
    init = _build_init(strategy, cfg)
    model, Lam = init(f0, Q, g)
    rng = np.random.default_rng(seed)
    for _ in range(harness.N_PASS):
        Hm, He = harness.simulate(rng)
        model, Lam = bayesian_update(harness.FREQ, model, Hm, He, Lam)
    ff, QQ, gg = float(model.f0[0]), float(model.Q[0]), float(model.gain)
    f0e, Qe, ge = abs(ff - 1.0), abs(QQ - 20.0) / 20.0, abs(gg - 100.0) / 100.0
    conv = f0e < harness.F0_TOL and Qe < harness.Q_TOL and ge < harness.G_TOL
    return {"key": _cfgkey(strategy, cfg), "case": label, "seed": seed,
            "f0_err": f0e, "Q_err": Qe, "gain_err": ge, "converged": conv}


def _build_jobs():
    jobs = []
    grids = {
        "baseline": [{}],
        "weaker": [{"U": u} for u in (0.4, 0.8, 1.5, 3.0, 8.0, 30.0)],
        "log_weaker": [{"U": u} for u in (0.2, 0.4, 0.8, 1.5, 3.0)],
        "qanchor": [{"uf": uf, "uq": uq, "ug": ug}
                    for uf in (0.5, 1.0, 2.0) for uq in (0.05, 0.1, 0.2) for ug in (0.05, 0.1, 0.2)],
        "log_qanchor": [{"uf": uf, "uq": uq, "ug": ug}
                        for uf in (0.3, 0.6, 1.0) for uq in (0.05, 0.1) for ug in (0.05, 0.1)],
    }
    seeds = range(6)
    for strategy, cfgs in grids.items():
        for cfg in cfgs:
            for case_idx in range(len(harness.CASES)):
                for seed in seeds:
                    jobs.append((strategy, cfg, case_idx, seed))
    return jobs


def main():
    jobs = _build_jobs()
    n_workers = os.cpu_count() or 4
    print(f"sweeping {len(jobs)} campaigns over {n_workers} workers "
          f"({len(set(j[0] + str(j[1]) for j in jobs))} configs x {len(harness.CASES)} cases x 6 seeds)...")
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in ex.map(_run_case, jobs, chunksize=4):
            results.append(r)

    # aggregate by config: convergence rate over (case x seed), and per-seed
    # cases-converged averaged over seeds
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in results:
        by_key[r["key"]].append(r)
    rows = []
    for key, rs in by_key.items():
        conv_rate = np.mean([r["converged"] for r in rs])
        # mean cases-converged per seed (out of 7)
        per_seed = defaultdict(int)
        for r in rs:
            per_seed[r["seed"]] += int(r["converged"])
        mean_cases = np.mean(list(per_seed.values()))
        med_f0 = np.median([r["f0_err"] for r in rs])
        med_Q = np.median([r["Q_err"] for r in rs])
        rows.append((key, conv_rate, mean_cases, med_f0, med_Q))
    rows.sort(key=lambda x: (-x[1], x[3]))

    print(f"\n{'config':36} {'conv_rate':>9} {'cases/7':>8} {'med_f0e':>8} {'med_Qe':>7}")
    for key, cr, mc, mf, mq in rows[:25]:
        print(f"{key:36} {cr:9.2f} {mc:8.2f} {mf:8.3f} {mq:7.2f}")


if __name__ == "__main__":
    main()
