"""Multi-start prior-convergence strategy for the bake-off.

Idea
----
The baseline fails three ways: Q collapses, gain collapses, and a far-off f0
freezes (the resonance peak sits where the Jacobian dH/df0 is weak, so the
local step can never walk the peak across the band).

Two fixes:

1. **log-space model** (``log=True``).  Parameters become
   ``[log f0, log Q, log|gain|]``; steps are multiplicative so Q and gain
   cannot collapse toward zero, and a "fractional uncertainty" is literally
   the log-sigma.  This kills the Q/gain-collapse failure mode.

2. **f0 multistart** on the first measurement.  We seed several f0 values
   ``prior_f0 * factors`` (clipped to the measurement band), run a few
   ``bayesian_update`` probe steps on the first measurement for each, and keep
   the seed with the lowest weighted residual to the data.  That seed's peak is
   in roughly the right place, so the remaining passes refine it normally.
   This kills the frozen-far-f0 failure mode.

State across passes is carried in a closure keyed on the pass index ``k``
(``k == 0`` is the probe/selection pass; ``k >= 1`` are refinement passes).
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import harness
from system_ident.resonator import ResonatorModel
from system_ident.estimators.bayesian import bayesian_update, prior_precision

F_LO, F_HI = float(harness.FREQ[0]), float(harness.FREQ[-1])


# Best config found by the sweep below (7/7 across noise seeds 0..9).
BEST = dict(
    log=True,
    prior_uncertainty=0.4,
    n_probe=8,
    max_rel_step=0.3,
    reset_lambda=True,
    seed_factors=(0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0),
)


def make_strategy(
    prior_uncertainty=0.4,
    seed_factors=(0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0),
    n_probe=8,
    max_rel_step=0.3,
    log=True,
    reset_lambda=True,
    lm_init=1e-3,
):
    """Return (init_fn, step_fn) for harness.evaluate.

    The closure carries the prior (f0,Q,gain) for the current case so the
    probe pass (k==0) can rebuild the f0 seed grid.
    """
    state = {}

    def init_fn(f0, Q, gain):
        m = ResonatorModel.from_resonances([(f0, Q)], gain, log=log)
        Lam = prior_precision(m, prior_uncertainty)
        state["prior"] = (f0, Q, gain)
        return m, Lam

    def step_fn(freq, model, Hm, He, Lam, k):
        if k == 0:
            f0, Q, gain = state["prior"]
            valid = np.isfinite(He) & (He > 0)
            wt = np.where(valid, 1.0 / He ** 2, 0.0)
            best = None
            for fac in seed_factors:
                f0_seed = float(np.clip(f0 * fac, F_LO, F_HI))
                mm = ResonatorModel.from_resonances([(f0_seed, Q)], gain, log=log)
                LL = prior_precision(mm, prior_uncertainty)
                for _ in range(n_probe):
                    mm, LL = bayesian_update(
                        freq, mm, Hm, He, LL,
                        max_rel_step=max_rel_step, lm_init=lm_init,
                    )
                resid = Hm - mm.eval(freq)
                wres = float(np.sum(wt * np.abs(resid) ** 2))
                if best is None or wres < best[0]:
                    best = (wres, mm, LL)
            _, bm, bL = best
            if reset_lambda:
                bL = prior_precision(bm, prior_uncertainty)
            return bm, bL
        return bayesian_update(
            freq, model, Hm, He, Lam,
            max_rel_step=max_rel_step, lm_init=lm_init,
        )

    return init_fn, step_fn


def run(name="multistart", **kw):
    init_fn, step_fn = make_strategy(**kw)
    rows = harness.evaluate(init_fn, step_fn)
    nconv = harness.report(name, rows)
    return nconv, rows


if __name__ == "__main__":
    import itertools

    # --- baseline reference ---
    harness.report("BASELINE", harness.evaluate(harness.baseline_init))

    # --- coarse grid sweep over hyperparameters ---
    grid = {
        "log": [True, False],
        "prior_uncertainty": [0.3, 0.4, 0.5],
        "n_probe": [3, 5, 8],
        "max_rel_step": [0.2, 0.3],
        "reset_lambda": [True, False],
        "seed_factors": [
            (0.5, 0.7, 1.0, 1.4, 2.0),
            (0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0),
        ],
    }
    keys = list(grid.keys())
    best = None
    results = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, combo))
        init_fn, step_fn = make_strategy(**kw)
        rows = harness.evaluate(init_fn, step_fn)
        nconv = sum(r["converged"] for r in rows)
        tot_err = sum(r["f0_err"] + r["Q_err"] + r["gain_err"] for r in rows)
        results.append((nconv, -tot_err, kw, rows))
        if best is None or (nconv, -tot_err) > (best[0], best[1]):
            best = (nconv, -tot_err, kw, rows)
            print(f"NEW BEST nconv={nconv} tot_err={tot_err:.3f} kw={kw}")

    results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print("\n=== TOP 5 CONFIGS ===")
    for nconv, neg_err, kw, rows in results[:5]:
        print(f"nconv={nconv} tot_err={-neg_err:.3f} {kw}")

    print("\n=== BEST CONFIG DETAIL ===")
    harness.report(f"multistart BEST {best[2]}", best[3])
