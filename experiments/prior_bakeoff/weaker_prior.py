"""STRATEGY: weaker_prior.

Sweep a UNIFORM prior strength `prior_uncertainty` (U) for a linear
ResonatorModel, using the package's default `bayesian_update` step. The
hypothesis is that a weaker-but-not-flat prior lets the MAP step relocate a
far-off resonance (the +-50% cases) while a too-weak/flat prior lets Q and
gain collapse on the low-SNR measurements.

init_fn(f0,Q,gain) builds ResonatorModel.from_resonances([(f0,Q)], gain) and
prior_precision(model, U). step_fn = package bayesian_update (default).

Run:
    cd <repo> && conda run --no-capture-output -n sysid \
        python experiments/prior_bakeoff/weaker_prior.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "experiments/prior_bakeoff")

import harness  # noqa: E402

from system_ident.resonator import ResonatorModel  # noqa: E402
from system_ident.estimators.bayesian import prior_precision  # noqa: E402


def make_init(U: float):
    """Build an init_fn that uses a uniform prior_uncertainty = U."""
    def init_fn(f0, Q, g):
        m = ResonatorModel.from_resonances([(f0, Q)], g)
        return m, prior_precision(m, U)
    return init_fn


# U=0.4 is the baseline; sweep weaker priors plus a near-flat one.
# Also include tighter values to map the "too-tight -> can't move" end so the
# full prior-strength tradeoff is characterized.
U_GRID = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 8.0, 1.0e3]


def sweep():
    results = {}
    for U in U_GRID:
        rows = harness.evaluate(make_init(U))   # default step_fn = bayesian_update
        label = f"weaker_prior U={U:g}" + ("  (near-flat)" if U >= 100 else "")
        nconv = harness.report(label, rows)
        results[U] = (nconv, rows)
    return results


if __name__ == "__main__":
    results = sweep()
    # Pick the best U: maximize converged, then minimize total relative error.
    def total_err(rows):
        return sum(r["f0_err"] + r["Q_err"] + r["gain_err"] for r in rows)
    best_U = max(results, key=lambda U: (results[U][0], -total_err(results[U][1])))
    nconv, rows = results[best_U]
    print(f"\n##### BEST: U={best_U:g} -> {nconv}/7 converged, "
          f"total_err={total_err(rows):.3f} #####")
    harness.report(f"BEST weaker_prior U={best_U:g}", rows)
