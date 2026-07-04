"""Assemble + discretize the coupled closed loop, and recover the open-loop plant.

Closed-loop input sensitivity  S = (I+L)^-1,  L = M_out · C_d · M_in · G  (n_act×n_act),
built with python-control and discretized to fs. The twin then simulates CONSISTENTLY:
u_drive = Sd · u_exc, then y_sens = Gd · u_drive — so Y = Gd·X and the matrix recovery
G = Y_mat · X_mat^-1 is exact off-resonance (verified, 2e-5 in a 2×2 prototype).
"""
from __future__ import annotations
import warnings
import numpy as np
import control


def velocity_damper(k, fc_hz):
    """Simple k·s·wc/(s+wc) damping filter."""
    wc = 2 * np.pi * fc_hz
    return control.tf([k * wc, 0.0], [1.0, wc])


class CoupledLoop:
    """Coupled closed-loop model: S = (I+L)^-1, L = Mout·Cd·Min·G, discretized to fs."""

    def __init__(self, plant_ss, controllers, M_in, M_out_ss, *, fs):
        self.fs = float(fs)
        self.n_sens = plant_ss.noutputs
        self.n_act = plant_ss.ninputs
        self.n_dof = len(controllers)

        G = plant_ss
        # Block-diagonal n_dof×n_dof controller; wrap in tf2ss in case append rejects SISO TFs
        Cd = control.tf2ss(control.append(*[control.tf2ss(c) for c in controllers]))
        Min = control.ss([], [], [], np.asarray(M_in, float))   # constant n_dof×n_sens
        Mout = M_out_ss                                          # n_act×n_dof

        L = control.minreal(Mout * Cd * Min * G, verbose=False)  # n_act×n_act open-loop gain
        eye = control.ss([], [], [], np.eye(self.n_act))
        S = control.minreal(control.feedback(eye, L), verbose=False)  # (I+L)^-1

        self.Sd = control.c2d(S, 1.0 / self.fs, "tustin")
        self.Gd = control.c2d(G, 1.0 / self.fs, "tustin")

    def is_stable(self):
        """Return True if all discrete poles of Sd lie strictly inside the unit circle."""
        p = control.poles(self.Sd)
        return bool(np.all(np.abs(p) < 1.0 - 1e-9))

    def oracle(self, freq):
        """Evaluate the discrete plant Gd at frequencies freq (Hz).

        Returns array of shape (n_sens, n_act, len(freq)).
        """
        z = np.exp(2j * np.pi * np.asarray(freq, float) / self.fs)
        return self.Gd(z)


def recover_open_loop(Xmat, Ymat, *, rcond=1e-10, cond_warn=1e8):
    """Per-bin ``G = Y·X^-1`` (the closed-loop MIMO reference-FRF recovery).

    Conditioning guard (A5): uses a ``rcond``-limited pseudo-inverse (== the plain inverse
    for well-conditioned X) so an ill-conditioned drive matrix — a bin where the excitation
    under-drives some actuator direction — cannot blow the recovery up into garbage. Warns
    once if any bin's ``cond(X)`` exceeds ``cond_warn`` (a diagnostic that the drive is
    starving a direction there, so the recovered FRF at that bin is untrustworthy).
    """
    X = np.asarray(Xmat); Y = np.asarray(Ymat)
    assert X.shape[1] == X.shape[2], \
        "recover_open_loop needs square X (n_act drive monitors per actuator)"
    out = np.einsum('fij,fjk->fik', Y, np.linalg.pinv(X, rcond=rcond))   # Y @ pinv(X) per bin
    worst = float(np.max(np.linalg.cond(X))) if X.shape[0] else 0.0
    if worst > cond_warn:
        warnings.warn(f"recover_open_loop: worst drive-matrix cond = {worst:.1e} "
                      f"(> {cond_warn:.0e}) — the excitation under-drives an actuator "
                      f"direction at some line; the recovered FRF there is uncertain.",
                      RuntimeWarning)
    return out


def off_resonance_mask(freq, modes_hz, frac=0.12):
    freq = np.asarray(freq, float)
    keep = np.ones(freq.shape, bool)
    for f0 in modes_hz:
        keep &= np.abs(freq/f0 - 1.0) > frac
    return keep
