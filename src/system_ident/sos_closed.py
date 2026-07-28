"""Stage 2 (pyctl) — closed-loop identification of the 40m SOS through its nominal loops.

Closes the nominal ``OPT_CTRL_SUS{POS,SIDE,PIT,YAW}`` damping controllers on the Stage-0 SOS
plant and asks the campaign's real question: **does Pintelon–Schoukens period-averaging remove
the closed-loop identification bias** that any naive open-loop-method estimate suffers when the
loop is closed and ground/sensor disturbance is present?

Answer (see :mod:`system_ident.closed_loop_id` for the mechanism and ``tests/test_sos_closed.py``
for the gated result): yes. On every damped DOF the naive estimate plateaus at a bias floor
independent of the period count ``P``, while the P&S estimate converges as ``1/sqrt(P)`` to the
true (discrete) plant. The undamped V and R modes — the SOS deliberately does not damp bounce or
roll — have *open* loops and therefore show naive == P&S: a built-in null control that the bias is
specifically the closed-loop effect.

Controllers are loaded from the committed ``models/sos_nominal_controllers.json`` (extracted once
from the salvaged 16384 Hz foton bank), so this module and its tests need neither ``twin`` nor
``aligo-suspension-models``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import control
import scipy.signal as sig

from .sos_plant import (
    DOFS, MEASURED_MODES_HZ, SOSOptic, build_plant, fit_suspension, inertia_vector,
)
from . import closed_loop_id as clid

DAMPED = ("L", "T", "P", "Y")          # V (bounce), R (roll) are undamped by design
UNDAMPED = ("V", "R")
_CTRL_JSON = Path(__file__).resolve().parent / "models" / "sos_nominal_controllers.json"


def load_nominal_controllers() -> dict[str, control.TransferFunction]:
    """The four nominal SOS damping controllers as continuous ``control`` systems."""
    data = json.loads(_CTRL_JSON.read_text())["controllers"]
    out = {}
    for dof, c in data.items():
        z = [complex(*x) for x in c["zeros"]]
        p = [complex(*x) for x in c["poles"]]
        out[dof] = control.tf(*sig.zpk2tf(z, p, c["gain"]))
    return out


def dof_calibrations(kappa_ref: float = 3.0) -> dict[str, float]:
    """Per-DOF loop-gain calibration ``kappa_d``, plant-derived (not hand-tuned).

    The raw-SI plant peak gain ``Q/(inertia·ω0²)`` spans ~1e4x between translational and angular
    DOF; the identical nominal filter would be wildly over-gained on the angular ones. Each loop
    is normalized to the one that already damps well (T), and ``kappa_ref`` sets how hard. This
    is the loop-gain role that the real front end's OSEM sensor/coil calibration and moment-arm
    projection (EUL2OSEM) provide physically.
    """
    inert = inertia_vector(SOSOptic())
    g = {d: 50.0 / (inert[DOFS.index(d)] * (2 * np.pi * MEASURED_MODES_HZ[d]) ** 2) for d in DOFS}
    return {d: kappa_ref * g["T"] / g[d] for d in DAMPED}


def build_closed_loop(kappa_ref: float = 3.0):
    """Return ``(G, Kfull, kappa)``: the 6-DOF SOS plant, the block-diagonal calibrated nominal
    controller (zero on V, R), and the per-DOF calibrations."""
    G = build_plant(SOSOptic(), fit_suspension(SOSOptic()))
    Kd = load_nominal_controllers()
    kappa = dof_calibrations(kappa_ref)
    blocks = [control.tf2ss(kappa[d] * Kd[d]) if d in DAMPED
              else control.ss([], [], [], [[0.0]]) for d in DOFS]
    return G, control.append(*blocks), kappa


def closed_loop_poles(G, Kfull):
    """Poles of the reference→output closed loop (carries all loop states)."""
    return control.poles(control.feedback(G, Kfull))


def damped_Q(G, Kfull) -> dict[str, float]:
    """Closed-loop quality factor of each DOF's mode (open-loop Q is 50)."""
    p = closed_loop_poles(G, Kfull)
    out = {}
    for d in DOFS:
        f0 = MEASURED_MODES_HZ[d]
        near = [q for q in p if q.imag > 0.1 and abs(abs(q) / (2 * np.pi) - f0) < 0.15 * f0]
        if near:
            pp = near[int(np.argmin([abs(abs(q) / (2 * np.pi) - f0) for q in near]))]
            out[d] = float(abs(pp) / (-2 * pp.real)) if pp.real < 0 else np.inf
    return out


def resonance_lines(n_per_mode: int = 11, span_linewidths: float = 5.0) -> np.ndarray:
    """Excited lines clustered on every SOS mode (Fisher-optimal)."""
    lines = []
    for d in DOFS:
        lines += list(MEASURED_MODES_HZ[d] * (1 + np.linspace(
            -span_linewidths, span_linewidths, n_per_mode) / 50.0))
    return np.unique(np.round(lines, 5))


def diag_resonance_error(freq_hz):
    """Return an ``err_fn(Ghat, Gtrue)`` scoring per-DOF diagonal recovery **at each DOF's
    resonance line** — the informative bins (off-resonance is disturbance-dominated by nature).
    """
    res_k = {d: int(np.argmin(np.abs(freq_hz - MEASURED_MODES_HZ[d]))) for d in DOFS}

    def err_fn(Ghat, Gtrue):
        errs = []
        for i, d in enumerate(DOFS):
            k = res_k[d]
            errs.append(abs(Ghat[k, i, i] - Gtrue[k, i, i]) / abs(Gtrue[k, i, i]))
        return float(np.mean(errs))
    return err_fn


@dataclass
class SOSClosedExperiment:
    """Assembled, balanced SOS closed loop ready for the bias/consistency experiment."""
    freq_hz: np.ndarray
    Gk: np.ndarray              # balanced true FRF (n_lines, 6, 6)
    Sk: np.ndarray
    SKk: np.ndarray
    scale: np.ndarray           # per-DOF balancing scale
    R: np.ndarray               # per-line reference phasors
    kappa: dict
    Q_closed: dict

    def per_dof_errors(self, *, n_periods, sigma, seeds):
        """Seed-averaged naive & P&S per-DOF diagonal on-resonance error at one period count."""
        errs_n = {d: 0.0 for d in DOFS}
        errs_p = {d: 0.0 for d in DOFS}
        res_k = {d: int(np.argmin(np.abs(self.freq_hz - MEASURED_MODES_HZ[d]))) for d in DOFS}
        for sd in range(seeds):
            V, Y = clid.simulate(self.Sk, self.SKk, self.Gk, self.R,
                                 n_periods=n_periods, sigma=sigma, rng=np.random.default_rng(sd))
            Gn, Gp = clid.naive_frf(V, Y), clid.ps_frf(V, Y)
            for i, d in enumerate(DOFS):
                k = res_k[d]
                errs_n[d] += abs(Gn[k, i, i] - self.Gk[k, i, i]) / abs(self.Gk[k, i, i])
                errs_p[d] += abs(Gp[k, i, i] - self.Gk[k, i, i]) / abs(self.Gk[k, i, i])
        return ({d: errs_n[d] / seeds for d in DOFS}, {d: errs_p[d] / seeds for d in DOFS})


def assemble(fs: float = 2048.0, kappa_ref: float = 3.0, seed: int = 0) -> SOSClosedExperiment:
    """Build the discretized, balanced closed loop and reference — ready to run the experiment."""
    G, Kfull, kappa = build_closed_loop(kappa_ref)
    Q_closed = damped_Q(G, Kfull)
    Gd = control.c2d(G, 1.0 / fs, "tustin")
    Kdz = control.c2d(Kfull, 1.0 / fs, "tustin")
    freq = resonance_lines()
    scale, Gk, Sk, SKk = clid.loop_frf_maps(Gd, Kdz, freq, balance=True)
    R = np.exp(2j * np.pi * np.random.default_rng(seed).random(len(freq)))
    return SOSClosedExperiment(freq, Gk, Sk, SKk, scale, R, kappa, Q_closed)


def _main() -> None:
    exp = assemble()
    print("per-DOF kappa:", {d: f"{v:.2e}" for d, v in exp.kappa.items()})
    print("closed-loop Q:", {d: round(v, 1) for d, v in exp.Q_closed.items()},
          "  (open-loop Q=50; V,R undamped by design)")
    print(f"\nper-DOF on-resonance recovery error (sigma=0.5, 30 seeds)")
    print(f"  {'P':>4s}  " + "  ".join(f"{d:>6s}" for d in DOFS) + "   estimator")
    for P in (8, 32, 128):
        en, ep = exp.per_dof_errors(n_periods=P, sigma=0.5, seeds=30)
        print(f"  {P:4d}  " + "  ".join(f"{en[d]:6.3f}" for d in DOFS) + "   naive")
        print(f"  {P:4d}  " + "  ".join(f"{ep[d]:6.3f}" for d in DOFS) + "   P&S")
    print("\nnaive plateaus on L/T/P/Y (bias); P&S falls ~1/sqrt(P); V,R (open loop) show naive==P&S.")


if __name__ == "__main__":
    _main()
