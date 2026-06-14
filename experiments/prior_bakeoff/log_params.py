"""STRATEGY: log_params

Estimate the resonator in log-space: theta = [log f0, log Q, log|gain|], and
solve each pass's MAP problem with a short *inner* Newton loop instead of a
single tiny step.

Why log-space:
  * log|gain| and log Q move *additively*, so they cannot "collapse" toward 0
    the way the linear baseline does (a 25% error is a bounded ~0.22 log step).
  * dH/d(log x) = x * dH/dx, so the Gauss-Newton direction is naturally scaled
    by the (decade-spanning) physical magnitudes -> far better conditioned.
  * the per-pass relative step-cap becomes a true *fractional* move on every
    parameter, so a far-off f0/gain prior still moves instead of freezing.

Why an inner loop (the step_fn):
  The package default takes ONE damped, step-capped GN step per pass.  From a
  *misaligned* sharp-peak prior that single step barely moves and, worse, the
  least-squares fit prefers to *flatten Q* (a broad low bump fits a misaligned
  peak better than a sharp one), so Q and gain collapse before f0 can relocate.
  Running a few damped GN iterations per pass -- all regularised by the SAME
  incoming prior precision, accruing exactly one unit of Fisher information at
  the end -- lets the model actually reach that pass's regularised MAP point,
  relocating f0 into the basin and recovering Q/gain.  Over passes the low-SNR
  noise averages out.

Hyperparameters swept: prior_uncertainty in {0.2,0.4,0.8,1.5}; an optional
log-Q-only prior tightening (q_tighten); inner-iteration count nit; per-pass
relative step cap mrs.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import harness
from system_ident.resonator import ResonatorModel
from system_ident.estimators.bayesian import bayesian_update, prior_precision


def make_init(prior_uncertainty, q_tighten=1.0, floor=1e-3):
    """Log-space model + diagonal prior precision, optionally tightening log-Q."""
    def init_fn(f0, Q, g):
        m = ResonatorModel.from_resonances([(f0, Q)], g, log=True)
        Lam = prior_precision(m, prior_uncertainty, floor=floor)
        if q_tighten != 1.0:
            Lam = Lam.copy()
            Lam[1, 1] *= q_tighten ** 2     # index 1 = log Q
        return m, Lam
    return init_fn


def make_step(nit=6, mrs=0.3, lm=1e-3):
    """Inner-Newton step: nit damped GN steps regularised by the *fixed* incoming
    prior precision L; return the model and L + one unit of Fisher information."""
    def step_fn(freq, m, Hm, He, L, k):
        m2, L2 = m, L
        for _ in range(nit):
            m2, L2 = bayesian_update(freq, m2, Hm, He, L,
                                     max_rel_step=mrs, lm_init=lm)
        return m2, L2
    return step_fn


def run(label, init_fn, step_fn=None, seed=0, quiet=False):
    rows = harness.evaluate(init_fn, step_fn=step_fn, seed=seed)
    if quiet:
        nconv = sum(r["converged"] for r in rows)
    else:
        nconv = harness.report(label, rows)
    return nconv, rows


def total_err(rows):
    return sum(r["f0_err"] + r["Q_err"] + r["gain_err"] for r in rows)


# ---------------------------------------------------------------------------
# BEST CONFIG (selected by the sweep below): log-space, uniform prior_uncertainty
# = 0.8, inner-Newton step with nit=6 iterations and a per-pass relative step cap
# mrs=0.5.  An explicit log-Q tightening was tested (q_tighten) but is NOT needed:
# the inner solve already prevents the Q/gain collapse, and the plain uniform
# prior gives the lowest total error.
# ---------------------------------------------------------------------------
BEST = dict(prior_uncertainty=0.8, q_tighten=1.0, nit=6, mrs=0.5)


def best_init():
    return make_init(BEST["prior_uncertainty"], q_tighten=BEST["q_tighten"])


def best_step():
    return make_step(nit=BEST["nit"], mrs=BEST["mrs"])


if __name__ == "__main__":
    results = {}

    # prior_uncertainty sweep (the assigned sweep) x inner-iteration count x cap
    for pu in (0.2, 0.4, 0.8, 1.5):
        for nit in (4, 6, 8):
            for mrs in (0.2, 0.3, 0.5):
                key = ("uniform", pu, 1.0, nit, mrs)
                results[key] = run(f"pu={pu} nit={nit} mrs={mrs}",
                                   make_init(pu), make_step(nit=nit, mrs=mrs),
                                   quiet=True)

    # optionally tighten log-Q on top of the priors (assigned option)
    for pu in (0.4, 0.8):
        for qt in (3.0, 10.0):
            for nit in (6, 8):
                key = ("qtight", pu, qt, nit, 0.3)
                results[key] = run(f"pu={pu} qt={qt} nit={nit}",
                                   make_init(pu, q_tighten=qt),
                                   make_step(nit=nit, mrs=0.3), quiet=True)

    # rank: most converged, then smallest total relative error
    ranked = sorted(results.items(),
                    key=lambda kv: (kv[1][0], -total_err(kv[1][1])), reverse=True)

    print("\n#### sweep summary (top 12 by nconv then total error) ####")
    print(f"{'key (kind,pu,qt,nit,mrs)':40} {'nconv':>5} {'tot_err':>8}")
    for key, (nconv, rows) in ranked[:12]:
        print(f"{str(key):40} {nconv:>5} {total_err(rows):8.3f}")

    # --- the OFFICIAL report for the chosen BEST config (harness seed=0) ----
    print("\n##### BEST CONFIG #####", BEST)
    _, best_rows = run("BEST", best_init(), best_step())  # prints harness.report

    # robustness: re-run BEST across several seeds (seed=0 is the official one)
    print("\n#### BEST-config robustness across seeds ####")
    tot = 0
    for seed in range(12):
        n, _ = run("", best_init(), best_step(), seed=seed, quiet=True)
        tot += n
        print(f"  seed={seed}: {n}/7 converged")
    print(f"  cross-seed total: {tot}/{12 * 7}")
