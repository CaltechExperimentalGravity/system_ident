"""Shared yardstick for the prior/convergence-strategy bake-off.

Every candidate strategy is evaluated on the SAME cases, noise, and metric so the
comparison is apples-to-apples. A strategy is just an ``init_fn`` (build the
starting model + prior precision Lambda from a prior (f0,Q,gain)) and an optional
``step_fn`` (default: the package's model-agnostic ``bayesian_update``).

Run the baseline:
    cd <repo> && conda run -n sysid python experiments/prior_bakeoff/harness.py
"""

from __future__ import annotations

import numpy as np

from system_ident.resonator import ResonatorModel
from system_ident.estimators.bayesian import bayesian_update, prior_precision

# --- ground truth, measurement grid, noise (low-SNR), pass budget ----------
TRUE = ResonatorModel.from_resonances([(1.0, 20.0)], 100.0)
FREQ = np.linspace(0.2, 3.0, 400)
PEAK = float(np.max(np.abs(TRUE.eval(FREQ))))
NOISE_FLOOR = 0.10 * PEAK     # flat absolute -> low SNR (SNR~10 at peak, <1 off-resonance)
N_PASS = 30

# --- prior cases: (label, f0, Q, gain), truth = (1.0, 20, 100) -------------
CASES = [
    ("f0-10%", 0.90, 18.0, 90.0),
    ("f0+10%", 1.10, 22.0, 110.0),
    ("f0-20%", 0.80, 16.0, 80.0),
    ("f0+20%", 1.20, 24.0, 120.0),
    ("f0-50%", 0.50, 10.0, 50.0),
    ("f0+50%", 1.50, 30.0, 150.0),
    ("mixed",  1.30, 12.0, 70.0),   # f0 high, Q low, gain low
]

# convergence thresholds (relative)
F0_TOL, Q_TOL, G_TOL = 0.05, 0.25, 0.25


def simulate(rng):
    """One noisy broadband measurement of TRUE on FREQ."""
    H = TRUE.eval(FREQ)
    He = NOISE_FLOOR * np.ones_like(FREQ)
    noise = (rng.standard_normal(FREQ.size) + 1j * rng.standard_normal(FREQ.size)) * He / np.sqrt(2)
    return H + noise, He


def baseline_init(f0, Q, g):
    """Default strategy init: linear ResonatorModel + uniform prior_uncertainty=0.4."""
    m = ResonatorModel.from_resonances([(f0, Q)], g)
    return m, prior_precision(m, 0.4)


def evaluate(init_fn, step_fn=None, seed=0, n_pass=N_PASS):
    """Run a strategy over all CASES; return a list of per-case result dicts.

    init_fn(f0, Q, gain) -> (model, Lambda)
    step_fn(freq, model, H_meas, H_err, Lambda, k) -> (model, Lambda)
        (default: bayesian_update with the package defaults)
    """
    if step_fn is None:
        def step_fn(freq, m, Hm, He, L, k):
            return bayesian_update(freq, m, Hm, He, L)

    rows = []
    for (label, f0, Q, g) in CASES:
        rng = np.random.default_rng(seed)
        model, Lam = init_fn(f0, Q, g)
        for k in range(n_pass):
            Hm, He = simulate(rng)
            model, Lam = step_fn(FREQ, model, Hm, He, Lam, k)
        ff, QQ, gg = float(model.f0[0]), float(model.Q[0]), float(model.gain)
        f0e, Qe, ge = abs(ff - 1.0) / 1.0, abs(QQ - 20.0) / 20.0, abs(gg - 100.0) / 100.0
        rows.append({
            "case": label, "f0": ff, "Q": QQ, "gain": gg,
            "f0_err": f0e, "Q_err": Qe, "gain_err": ge,
            "converged": bool(f0e < F0_TOL and Qe < Q_TOL and ge < G_TOL),
        })
    return rows


def report(name, rows):
    nconv = sum(r["converged"] for r in rows)
    print(f"\n=== {name}: {nconv}/{len(rows)} converged ===")
    print(f"{'case':8} {'f0':>6} {'Q':>6} {'gain':>7} {'f0_err':>7} {'Q_err':>6} {'g_err':>6} {'conv':>5}")
    for r in rows:
        print(f"{r['case']:8} {r['f0']:6.3f} {r['Q']:6.1f} {r['gain']:7.1f} "
              f"{r['f0_err']:7.3f} {r['Q_err']:6.2f} {r['gain_err']:6.2f} {str(r['converged']):>5}")
    return nconv


if __name__ == "__main__":
    report("BASELINE (linear, prior_uncertainty=0.4)", evaluate(baseline_init))
