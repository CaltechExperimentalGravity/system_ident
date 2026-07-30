"""Hierarchical DARM actuation — the nested-offload distribution filters.

Reproduced verbatim (FRF-identical, verified) from the twin experiment
``digital_twin/twin/experiments/cavity_arm_lsc_hierarchical/lib.py`` (``offload_filters``),
so ``system_ident`` stays self-contained. The DARM control feeds the fast ESD (TST) stage
directly; two shaped offload controllers push drive UP the chain with integral action:

    O_A  — PUM offload: integ(8.2·f_ep) · lead(0.185·f_ep, 5.5·f_ep)² · gain-bump(1.70 Hz)
    O_B  — TOP offload: integ(2.4·f_pt) · lowpass(2.0·f_pt, 2) · notch(1.72 Hz)

so each stage's drive, relative to the ESD command, is the distribution filter

    D_TST = 1 ,   D_PUM = O_A ,   D_M0 = O_A · O_B          (M0 = TOP)

with design crossovers f_ep = 10 Hz (ESD/PUM) and f_pt = 0.5 Hz (PUM/TOP). The margin biquads
are pinned to the QUAD forest (1.70/1.72 Hz) — a property of the plant, not the crossover
targets. Drop the result into ``DARMLoop.distribution`` so the per-stage actuation becomes
``A_i = κ_i · D_i · N_i`` and the inter-stage crossovers become measurable with cal lines.
"""
from __future__ import annotations

import numpy as np
import control as ct

# Design constants (from the twin experiment; the crossover labels are tunable, the margin
# biquad centres are pinned to the QUAD suspension-mode forest).
F_EP = 10.0            # Hz — ESD/PUM crossover target
F_PT = 0.5             # Hz — PUM/TOP crossover target
F_MARGIN_A_HZ = 1.70   # O_A resonant gain-bump centre (PUM-offload conditional GM)
F_MARGIN_B_HZ = 1.72   # O_B notch centre (kills the TOP-offload forest re-crossing)


def offload_controller(integ_hz, leads=(), lowpass=(), biquads=()):
    """``O(s) = (2π·integ_hz/s) · Π lead · Π lowpass · Π biquad`` (see module docstring)."""
    s = ct.tf("s")
    O = (2 * np.pi * integ_hz) / s
    for f_z, f_p in leads:
        w_z, w_p = 2 * np.pi * f_z, 2 * np.pi * f_p
        O = O * ((s + w_z) / (s + w_p)) * (w_p / w_z)          # unity-normalised lead
    for f_p, n in lowpass:
        w_p = 2 * np.pi * f_p
        O = O * (w_p / (s + w_p)) ** n
    for f_0, q_z, q_p in biquads:
        w_0 = 2 * np.pi * f_0
        O = O * ct.tf([1.0, w_0 / q_z, w_0 ** 2], [1.0, w_0 / q_p, w_0 ** 2])
    return ct.tf2ss(O)


def offload_filters(f_ep: float = F_EP, f_pt: float = F_PT):
    """The two shaped offload controllers ``(O_A, O_B)`` — PUM and TOP offload."""
    O_A = offload_controller(8.2 * f_ep, leads=((0.185 * f_ep, 5.5 * f_ep),) * 2,
                             biquads=((F_MARGIN_A_HZ, 2.0, 6.0),))
    O_B = offload_controller(2.4 * f_pt, lowpass=((2.0 * f_pt, 2),),
                             biquads=((F_MARGIN_B_HZ, 7.0, 1.1),))
    return O_A, O_B


class _CtrlFilter:
    """Wrap a ``control`` LTI as the ``.eval(freq)`` interface ``DARMLoop.distribution`` uses."""

    def __init__(self, sys):
        self._sys = sys

    def eval(self, freq) -> np.ndarray:
        w = 2 * np.pi * np.asarray(freq, dtype=float)
        return np.asarray(ct.frequency_response(self._sys, w).frdata).ravel()


class _UnityFilter:
    def eval(self, freq) -> np.ndarray:
        return np.ones_like(np.asarray(freq, dtype=float), dtype=complex)


def hierarchical_distribution(f_ep: float = F_EP, f_pt: float = F_PT) -> dict:
    """Per-stage distribution filters ``{M0, PUM, TST}`` for the nested-offload DARM actuation,
    ready to assign to ``DARMLoop.distribution``. ``D_TST = 1``, ``D_PUM = O_A``,
    ``D_M0 = O_A·O_B`` (the drive each stage receives relative to the ESD command)."""
    O_A, O_B = offload_filters(f_ep, f_pt)
    return {"M0": _CtrlFilter(O_A * O_B),
            "PUM": _CtrlFilter(O_A),
            "TST": _UnityFilter()}
