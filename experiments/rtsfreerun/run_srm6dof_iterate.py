"""FAR-PRIOR iterative loop: the P&S estimate → redesign → re-measure loop for the MIMO
SRM modal ID, driven by ``mimo_iterate.iterate_mimo`` with the A2 DONE criterion.

The prior modes are deliberately offset +30% (a FAR prior). Pass 0 designs a prior-robust
drive (u=0.5) from that wrong prior — which still puts real power at the true modes — and a
BLIND fit (``find_modes`` on the recovered FRF, no oracle) recovers them. Later passes
design POINT-OPTIMAL drives (u=0) from the now-trusted FITTED modes, until the worst-case
fractional per-mode uncertainty (``modal_frac_uncertainty``) drops below TARGET_FRAC_UNC.

This is the MIMO analog of ``loop.py::SysIDLoop.run``; the loop policy lives in
``system_ident.mimo_iterate`` (unit-tested) and this script supplies the twin callbacks.

NOTE: the demo pins a small fixed peak-fraction floor (DEMO_FLOOR) so the drive genuinely
concentrates and the robust-vs-point-optimal distinction is real (production uses the
derived-α FLOOR_ENERGY). Each pass is a ~13-min twin campaign; caches per pass.

Run:  conda run -n sysid python experiments/rtsfreerun/run_srm6dof_iterate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import srm6dof_loop as s6
import run_srm6dof_modal as base
from system_ident.mimo_iterate import iterate_mimo
from system_ident.mimo_loop import recover_open_loop
from system_ident.mimo_modal import Rank1ModalModel
from system_ident.mimo_fit import (find_modes, init_residues, MIMOModalEstimator,
                                   mimo_parameter_covariance, modal_uncertainty,
                                   modal_frac_uncertainty)

OFFSET = 1.30           # prior modes 30% too HIGH — a far prior (only the DRIVE sees it)
DEMO_FLOOR = 0.005      # small fixed floor so the drive genuinely concentrates
TARGET_FRAC_UNC = 5e-6  # DONE: stop when the worst-case fractional per-mode CRB < this
MAX_PASSES = 2          # bound the twin campaigns (each pass ≈ 13 min)


def blind_assess(exps, freq, ora, dof):
    """BLIND fit + A2 CRB on one pass's campaign: find_modes (no oracle) -> seed+anchor to
    the DATA modes -> fit -> mimo_parameter_covariance -> modal_frac_uncertainty. Returns
    the dict iterate_mimo needs ('modes', 'frac_unc') plus scoring vs the oracle."""
    X = np.stack([exps[l][1] for l in range(6)], axis=-1)
    Y = np.stack([exps[l][0] for l in range(6)], axis=-1)
    Gnp = recover_open_loop(X, Y)
    found = find_modes(Gnp, freq)
    m = Rank1ModalModel(6, 6, n_modes=len(found)).set_reference(freq)
    ab = m.ab_from_modes(found); phi, psi = init_residues(m, ab, exps, freq)
    res = MIMOModalEstimator(m).fit(exps, freq, m.pack(ab, phi, psi),
                                    pole_prior_hz=[f for f, _ in found],
                                    prior_weight=base.PRIOR_WEIGHT)
    Ct = mimo_parameter_covariance(m, res.theta, exps, freq, dof=dof, n_sens=6)
    mu = modal_uncertainty(m, res.theta, Ct)
    fitted = sorted((x["f0"], x["Q"]) for x in mu if np.isfinite(x["Q"]) and x["Q"] > 0)
    n_well = sum(1 for x in mu if min(abs(x["f0"] - t) / t for t, _ in ora) < 0.01)
    return {"modes": fitted, "frac_unc": modal_frac_uncertainty(m, res.theta, Ct),
            "n_found": len(found), "n_well": n_well}


def main():
    print("=== SRM 6-DoF FAR-PRIOR iterative loop (iterate_mimo + A2 DONE criterion) ===")
    m6 = s6.SRM6DOF()
    sel, freq = base._grid()
    ora = base.oracle_modes(m6)
    offset_prior = base.distinct_oracle_modes([(f * OFFSET, q) for f, q in ora])
    dof = base.N_PERIODS - base.N_TRANSIENT
    print(f"[prior] TRUE fundamental {ora[0][0]:.4f} Hz; FAR prior seeded at "
          f"{offset_prior[0][0]:.4f} Hz (+{(OFFSET-1)*100:.0f}%). "
          f"target frac-unc={TARGET_FRAC_UNC:.0e}, max {MAX_PASSES} passes.")
    d = np.load(base._cal_cache())
    cal = {k: float(d["cal"][i]) for i, k in enumerate(m6.dofs)}
    m6.set_cal(cal)

    def design(modes, u):
        Pxx, _ = base.design_drive(m6, freq, modes=modes, u=u, floor_frac=DEMO_FLOOR)
        return Pxx

    def measure(Pxx, k):
        kind = "robust (u=0.5) from the offset prior" if k == 0 else "point-optimal from fitted"
        print(f"\n[pass {k}] {kind} — campaign [~13 min]:")
        exps, _, _ = base.run_campaign(m6, cal, freq, Pxx,
                                       cache_path=HERE / f"srm_iterate_pass{k}.npz")
        return exps

    def fit(exps):
        r = blind_assess(exps, freq, ora, dof)
        print(f"  → BLIND fit: {r['n_found']} modes found; {r['n_well']}/13 within 1%; "
              f"worst-case frac-unc = {r['frac_unc']:.2e}")
        return r

    final, history = iterate_mimo(design, measure, fit, prior_modes=offset_prior,
                                  prior_u=0.5, target_frac_unc=TARGET_FRAC_UNC,
                                  max_passes=MAX_PASSES)

    print("\n=== ITERATION RESULT ===")
    for h in history:
        print(f"  pass {h['pass']} (u={h['u']}): {h['n_well']}/13 within 1%, "
              f"frac-unc = {h['frac_unc']:.2e}")
    print(f"  → {'CONVERGED' if final['converged'] else 'stopped at max_passes'} after "
          f"{len(history)} pass(es); worst-case frac-unc {history[0]['frac_unc']:.2e} → "
          f"{final['frac_unc']:.2e} "
          f"({history[0]['frac_unc']/final['frac_unc']:.1f}× tighter by concentrating on the "
          f"modes the robust first pass located).")
    return final, history


if __name__ == "__main__":
    main()
