"""Coupled MIMO suspension plant + decoupling matrices for the closed-loop twin.

A shared normal-mode expansion: every (output, input) element shares the same modal
poles, so the diagonal carries anti-resonance notches and the off-diagonal carries
notch-free coupling (the matrix generalisation of ``plant.coupled_suspension``).
Returned as python-control systems so the closed loop assembles natively.
"""
from __future__ import annotations
import numpy as np
import control


def mimo_suspension(modes, *, n_sens, n_act, coupling=0.2, gain=100.0, seed=0):
    """Coupled ``n_sens × n_act`` plant from shared modes ``[(f0,Q),...]``."""
    rng = np.random.default_rng(seed)
    n_modes = len(modes)
    # mode shapes: each mode k has a sensor pattern phi (n_sens) and actuator pattern psi (n_act),
    # ~ unit on the "home" DoF with small alternating cross terms -> genuine coupling.
    phi = np.eye(n_sens, n_modes) + coupling * np.cos(rng.uniform(0, np.pi, (n_sens, n_modes)))
    psi = np.eye(n_act, n_modes) + coupling * np.cos(rng.uniform(0, np.pi, (n_act, n_modes)))
    # shared denominator (product of the mode 2nd-order factors)
    dens = []
    for (f0, q) in modes:
        w = 2 * np.pi * f0
        dens.append(np.array([1.0, w / q, w * w]))
    den = np.array([1.0])
    for d in dens:
        den = np.polymul(den, d)
    num = [[None] * n_act for _ in range(n_sens)]
    for i in range(n_sens):
        for j in range(n_act):
            acc = np.zeros(1)
            for k in range(n_modes):
                term = np.array([gain * phi[i, k] * psi[j, k]])
                for m, dm in enumerate(dens):
                    if m != k:
                        term = np.polymul(term, dm)
                acc = np.polyadd(acc, term)
            num[i][j] = list(map(float, np.atleast_1d(acc)))
    return control.tf2ss(
        control.tf(num, [[list(map(float, den))] * n_act for _ in range(n_sens)])
    )


def input_matrix(n_dof, n_sens, *, kind="identity", seed=0):
    """Constant sensor→DOF matrix ``M_in`` (n_dof × n_sens)."""
    M = np.eye(n_dof, n_sens)
    if kind == "perturbed":
        M = M + 0.1 * np.random.default_rng(seed).standard_normal((n_dof, n_sens))
    return M


def output_matrix(plant_ss, n_act, n_dof, *, basis="euler"):
    """Frequency-dependent DOF-control→actuator matrix ``M_out(s)`` (n_act × n_dof).

    ``euler``: a static pass-through (identity-shaped) embedded as a system — the DOF
    basis *is* the actuator basis. ``eigenmode``: a representative modal decoupler — a
    constant orthogonal-ish mixing with a gentle 1-pole roll shaping (so it is genuinely
    frequency-dependent), standing in for a real decoupling-filter design.
    """
    if basis not in ("euler", "eigenmode"):
        raise ValueError(f"unknown basis {basis!r}")
    if basis == "euler":
        return control.ss([], [], [], np.eye(n_act, n_dof))
    # eigenmode: constant mix * a shared 1-pole shaping per channel (frequency-dependent).
    # control.tf(num_array, den_array) fights on shapes in 0.10.2, so use the fallback:
    # static gain matrix * block-diagonal shaped system.
    mix = np.eye(n_act, n_dof) + 0.15 * np.cos(
        np.arange(n_act)[:, None] + np.arange(n_dof)[None, :]
    )
    shape = control.tf([1.0], [1 / (2 * np.pi * 5.0), 1.0])  # gentle pole at 5 Hz
    return control.ss([], [], [], mix) * control.tf2ss(control.append(*[shape] * n_dof))
