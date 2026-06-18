"""A3+A4 demo: identify the compiled ``x1hsts6dof`` HSTS plant under the real
production L1-MC2 dampers closed around all six DOFs.

Runs the leakage-free reference-FRF measurement (the P&S path) two ways and scores
both against the analytic state-space oracle:

* **A4 (loops open)** — the full 6×6 drive→sense FRF tensor (diagonal anti-resonances
  + dominant L↔P / R↔Y cross-coupling).
* **A3 (loops closed)** — the same reference FRF with the true plant input
  ``DRIVE_EXC − damper_feedback``: recovers the *open-loop* diagonal through the active
  loops (controller cancelled).

Writes ``hsts6dof_recovery.png`` (6×6 |FRF| grid, oracle vs open-loop recovered, with
the closed-loop diagonal overlaid). Requires the twin model + archives (see
``hsts6dof_loop.py``); prints a notice and exits otherwise.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hsts6dof_loop as h6

FS, NPERSEG, NPERIODS, NPASSES = 256.0, 4096, 4, 2


def main():
    if not h6.deps_available():
        print("x1hsts6dof / twin example / production archives not present — skipping.")
        return
    model = h6.HSTS6DOF()
    fa = np.fft.rfftfreq(NPERSEG, 1 / FS)
    band = (fa >= 0.3) & (fa <= 8.0)
    freq = fa[band]
    kw = dict(fs=FS, nperseg=NPERSEG, n_periods=NPERIODS, band=band, freq=freq,
              n_passes=NPASSES, warmup_s=16.0, seed=0)

    print("measuring open-loop MIMO tensor (A4) ...")
    H_open = model.measure_tensor(closed=False, **kw)
    print("measuring closed-loop tensor (A3, all 6 real loops engaged) ...")
    H_closed = model.measure_tensor(closed=True, **kw)

    M_open = model.rel_err_tensor(H_open, freq)
    M_closed = model.rel_err_tensor(H_closed, freq)
    dofs = model.dofs

    def show(tag, M, diag_only=False):
        print(f"\n=== {tag} : median rel-err vs SS oracle ===")
        print("        " + "".join(f"{d:>8}" for d in dofs) + "   <- drive")
        for i, di in enumerate(dofs):
            row = "".join((f"{M[i, j]:8.4f}" if (not diag_only or i == j) else f"{'·':>8}")
                          for j in range(len(dofs)))
            print(f"out {di}: " + row)

    show("A4 open-loop tensor", M_open)
    show("A3 closed-loop (diagonal = open-loop plant recovered)", M_closed, diag_only=True)
    print("\nA4 diagonal:", dict(zip(dofs, np.round(np.diag(M_open), 4))))
    print("A3 diagonal:", dict(zip(dofs, np.round(np.diag(M_closed), 4))))

    _figure(model, freq, H_open, H_closed)


def _figure(model, freq, H_open, H_closed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dofs = model.dofs
    G = model.oracle_tensor(freq)
    n = len(dofs)
    fig, axs = plt.subplots(n, n, figsize=(15, 12), sharex=True)
    fig.suptitle("HSTS 6-DOF: analytic oracle (line) vs P&S reference-FRF recovery — "
                 "open-loop tensor (A4); closed-loop diagonal in green (A3)", fontsize=12)
    for i in range(n):
        for j in range(n):
            ax = axs[i, j]
            ax.loglog(freq, np.abs(G[:, i, j]), "k-", lw=1.0, alpha=0.7)
            ax.loglog(freq, np.abs(H_open[i, j]), "C3.", ms=3)
            if i == j:
                ax.loglog(freq, np.abs(H_closed[i, j]), "C2x", ms=4, alpha=0.7)
            if i == 0:
                ax.set_title(f"drive {dofs[j]}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"sense {dofs[i]}", fontsize=9)
            ax.tick_params(labelsize=6)
            ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "hsts6dof_recovery.png"
    fig.savefig(out, dpi=130)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
