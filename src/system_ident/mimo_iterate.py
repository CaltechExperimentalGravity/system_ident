"""Iterative P&S estimate → redesign → re-measure loop for MIMO modal ID.

The MIMO analog of ``loop.py::SysIDLoop.run`` (the mature SISO loop): a prior-robust
FIRST pass (drive spread over the prior modes' plausible band), then POINT-OPTIMAL drives
designed from the now-trusted FITTED modes, until the worst-case fractional per-mode
uncertainty (the A2 DONE criterion, ``mimo_fit.modal_frac_uncertainty``) drops below a
target — or ``max_passes`` is hit.

It is callback-based so the orchestration is independent of the twin/backend/designer and
can be unit-tested without running campaigns:

    design(modes, u) -> Pxx      the drive PSD (prior-robust over ``modes`` with fractional
                                 uncertainty ``u``; u = prior_u on pass 0, then 0 =
                                 point-optimal from the trusted fitted modes)
    measure(Pxx, k)  -> exps     run the campaign with drive ``Pxx`` on pass index ``k``
    fit(exps)        -> dict     a BLIND fit + CRB assessment; MUST contain
                                 ``"modes": [(f0,Q),...]`` and ``"frac_unc": float``
                                 (plus anything else to record, e.g. recovery rel-err).

Re-fits each pass on that pass's data (no cross-pass inverse-variance accumulation yet —
a refinement the SISO loop has; noted for follow-up).
"""
from __future__ import annotations

import numpy as np


def iterate_mimo(design, measure, fit, *, prior_modes, prior_u=0.5,
                 target_frac_unc, max_passes=5):
    """Run the estimate→redesign→re-measure loop. Returns ``(final_result, history)``.

    ``final_result`` is the last pass's ``fit`` dict, annotated with ``pass``, ``u`` and
    ``converged``. ``history`` is the per-pass list of the same. The first pass uses
    ``prior_u`` (prior-robust); every later pass uses ``u=0`` (point-optimal) designed from
    the previous pass's fitted modes. Stops as soon as ``frac_unc <= target_frac_unc``.
    """
    modes = list(prior_modes)
    u = float(prior_u)
    history = []
    converged = False
    for k in range(int(max_passes)):
        Pxx = design(modes, u)
        exps = measure(Pxx, k)
        res = dict(fit(exps))
        res["pass"] = k
        res["u"] = u
        history.append(res)
        modes = res["modes"]                       # trust the fit for the next design
        if float(res.get("frac_unc", np.inf)) <= float(target_frac_unc):
            converged = True
            break
        u = 0.0                                     # point-optimal from the trusted modes
    if history:
        history[-1]["converged"] = converged
    return (history[-1] if history else None), history
