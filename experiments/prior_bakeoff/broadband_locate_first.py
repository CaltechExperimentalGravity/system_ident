"""STRATEGY: broadband_locate_first

Two-stage "locate then refine":

* Pass 0 (k==0): a GLOBAL fit that *ignores the prior values*. We seed a neutral
  broadband resonator at the geometric-mean of the band (low Q, so the invfreqs
  weighting is nearly flat) and run a few Sanathanan-Koerner (SK) iterations of
  the package ``InvfreqsEstimator`` against the measured response. The dominant
  ``den`` root pair gives (f0, Q); the overall gain is recovered by a weighted
  least-squares match of the unit-gain resonator to the data. We rebuild a
  physical ``ResonatorModel`` from THAT and reset the prior precision around it.

* Passes 1..N (k>0): refine with the package ``bayesian_update``.

State is carried via a closure on ``k``.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import harness

from system_ident.resonator import ResonatorModel
from system_ident.model import pole_pair_f0_Q
from system_ident.estimators.invfreqs import InvfreqsEstimator
from system_ident.estimators.bayesian import bayesian_update, prior_precision

FREQ = harness.FREQ
BAND_GEO = float(np.sqrt(FREQ[0] * FREQ[-1]))   # geometric-mean of the band


def _global_locate(freq, Hm, He, seed_f0, seed_Q, n_sk):
    """Prior-agnostic global fit -> (f0, Q, gain)."""
    est = InvfreqsEstimator()
    tf = ResonatorModel.from_resonances([(seed_f0, seed_Q)], 1.0).to_tf()
    for _ in range(n_sk):
        tf = est.fit(freq, Hm, He, tf)
    roots = np.roots(tf.den)
    cand = [r for r in roots if r.imag > 0] or list(roots)
    # dominant resonance = highest-Q in-band pole pair
    best, bestQ = None, -np.inf
    for r in cand:
        f0, Q = pole_pair_f0_Q(r.real, abs(r.imag))
        if (freq[0] * 0.5 < f0 < freq[-1] * 2.0) and Q > bestQ:
            bestQ, best = Q, (f0, Q)
    if best is None:
        best = pole_pair_f0_Q(cand[0].real, abs(cand[0].imag))
    f0, Q = best
    # gain by weighted LS on the linear-in-gain unit resonator
    Hb = ResonatorModel.from_resonances([(f0, Q)], 1.0).eval(freq)
    wt = 1.0 / He ** 2
    gain = float(np.sum(wt * np.real(np.conj(Hb) * Hm)) / np.sum(wt * np.abs(Hb) ** 2))
    return f0, Q, gain


def make_strategy(seed_Q=1.0, n_sk=8, refine_unc=0.2, log=True,
                  bu_kwargs=None, refine=True):
    # BEST CONFIG defaults: neutral broadband seed (Q=1) at the band geo-mean,
    # 8 SK iterations to locate, then log-space bayesian_update refinement with
    # prior_uncertainty=0.2.
    bu_kwargs = bu_kwargs or {}

    def init_fn(f0, Q, gain):
        m = ResonatorModel.from_resonances([(f0, Q)], gain, log=log)
        return m, prior_precision(m, refine_unc)

    def step_fn(freq, model, Hm, He, Lam, k):
        if k == 0:
            f0, Q, gain = _global_locate(freq, Hm, He, BAND_GEO, seed_Q, n_sk)
            m = ResonatorModel.from_resonances([(f0, Q)], gain, log=log)
            return m, prior_precision(m, refine_unc)
        if not refine:
            return model, Lam
        return bayesian_update(freq, model, Hm, He, Lam, **bu_kwargs)

    return init_fn, step_fn


def run(name, **kw):
    init_fn, step_fn = make_strategy(**kw)
    rows = harness.evaluate(init_fn, step_fn)
    n = harness.report(name, rows)
    return n, rows


if __name__ == "__main__":
    # BEST CONFIG (defaults of make_strategy): n_sk=8, refine_unc=0.2, log=True.
    run("broadband_locate_first  (BEST: n_sk=8, refine_unc=0.2, log=True)")
