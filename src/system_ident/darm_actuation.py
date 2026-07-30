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

from .reduced_plant import ReducedStateSpacePlant

# Design constants (from the twin experiment; the crossover labels are tunable, the margin
# biquad centres are pinned to the QUAD suspension-mode forest).
F_EP = 10.0            # Hz — ESD/PUM crossover target
F_PT = 0.5             # Hz — PUM/TOP crossover target
F_MARGIN_A_HZ = 1.70   # O_A resonant gain-bump centre (PUM-offload conditional GM)
F_MARGIN_B_HZ = 1.72   # O_B notch centre (kills the TOP-offload forest re-crossing)

# M0 local-damping design. The twin builds the hierarchical loop on the *damped* QUAD
# (experiments/cavity_arm_lsc_hierarchical/lib.py: build_damped_plant closes the 6-DOF M0
# velocity-damping loop around the reduced quad FIRST, then designs the offload filters against
# the damped compliances). The exact H1 ETMX_M0_DAMP banks live in an aLIGO foton file that is
# not vendored here, but the damping's ROLE is only to knock the 0.4–3 Hz suspension-mode forest
# down from Q~1e3 to Q~few so the smooth 1/f² compliance tails — which set the inter-stage
# crossovers — are not masked by resonant spikes. Those crossovers are governed by the offload
# filters + the smooth compliance ratios and are essentially independent of the damping detail
# (verified: PUM/TST crossover stays at 10.3 Hz from undamped through every damping level), so a
# generic 6-DOF M0 velocity damper is a faithful, self-contained stand-in for the H1 banks.
DAMP_GAIN = 1.0e3      # 6-DOF M0 velocity-damping gain (per DOF)
DAMP_ROLL_HZ = 4.0     # 2nd-order rolloff above the suspension band [Hz]
_DOFS = ["L", "T", "V", "R", "P", "Y"]


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


# ---------------------------------------------------------------------------
# M0-damped QUAD compliances — the mechanical response each hierarchical stage
# drives, AFTER local M0 damping (mirrors the twin's build_damped_plant).
# ---------------------------------------------------------------------------
#: Reduced-quad channels: the three DARM actuation stages (top=M0, pum=L2, esd=L3) and the
#: M0 6-DOF ports the local damping loop is closed around, all read out at the test mass L3.
_STAGE_DRIVE = {"M0": "M0.drive.L", "PUM": "L2.drive.L", "TST": "L3.drive.L"}
_L3_DISP = "L3.disp.L"


class DampedQuadCompliance:
    """The three DARM-stage compliances ``{M0,PUM,TST}.drive → L3.disp.L`` of the reduced QUAD
    with the 6-DOF M0 local-damping loop closed (as the twin builds it: damp first, then design
    the hierarchy on the damped plant).

    The damping loop ``u_M0 = −K(f)·y_M0`` (``K`` diagonal velocity damping over the six M0 DOFs)
    is closed exactly in the frequency domain per bin — no state-space realisation of the damper
    needed — and the external stage drives enter on top (the top stage adds into ``M0.drive.L``).
    ``.eval(freq)`` returns an ``(F, 3)`` array of columns ordered ``[M0, PUM, TST]``, memoised per
    ``freq``-array identity so repeated snapshots on one grid don't re-solve the plant per bin.
    """

    STAGES = ("M0", "PUM", "TST")

    def __init__(self, plant: ReducedStateSpacePlant | None = None,
                 gain: float = DAMP_GAIN, roll_hz: float = DAMP_ROLL_HZ):
        self._p = plant if plant is not None else ReducedStateSpacePlant.load("quad")
        self._gain = float(gain)
        self._roll = float(roll_hz)
        p = self._p
        self._i_m0 = [p.inputs.index(f"M0.drive.{d}") for d in _DOFS]
        self._o_m0 = [p.outputs.index(f"M0.disp.{d}") for d in _DOFS]
        self._i_ext = [p.inputs.index(_STAGE_DRIVE[s]) for s in self.STAGES]
        self._o_l3 = p.outputs.index(_L3_DISP)
        self._sel_top = np.zeros(6)          # top stage adds into M0.drive.L
        self._sel_top[_DOFS.index("L")] = 1.0
        self._cache: dict = {}

    def _damper(self, freq: np.ndarray) -> np.ndarray:
        """Per-DOF M0 velocity damping K(f) = gain·(2πi f)/(1 + i f/roll)² [same on all 6 DOFs]."""
        return self._gain * (2j * np.pi * freq) / (1.0 + 1j * freq / self._roll) ** 2

    def eval(self, freq) -> np.ndarray:
        freq = np.asarray(freq, dtype=float)
        key = (id(freq), freq.shape, float(freq[0]), float(freq[-1]))
        out = self._cache.get(key)
        if out is not None:
            return out
        G = self._p.eval(freq)                      # (F, nout, nin)
        Kd = self._damper(freq)
        sel = self._sel_top
        cols = np.empty((len(freq), 3), dtype=complex)
        for k in range(len(freq)):
            Gmm = G[k][np.ix_(self._o_m0, self._i_m0)]     # 6x6 M0.drive -> M0.disp
            Gme = G[k][np.ix_(self._o_m0, self._i_ext)]    # 6x3 ext.drive -> M0.disp
            Glm = G[k][self._o_l3, :][self._i_m0]          # (6,) M0.drive -> L3.disp
            Gle = G[k][self._o_l3, :][self._i_ext]         # (3,) ext.drive -> L3.disp
            K = np.diag(np.full(6, Kd[k]))
            M = np.linalg.inv(np.eye(6) + Gmm @ K)         # (I + Gmm K)^-1
            for j in range(3):
                st = sel if j == 0 else np.zeros(6)        # top drives M0.drive.L
                y_m0 = M @ (Gme[:, j] + Gmm @ st)          # closed-loop M0 motion
                u_m0 = -K @ y_m0 + st                      # net M0 drive
                cols[k, j] = Glm @ u_m0 + Gle[j]           # L3.disp.L
        self._cache[key] = cols
        return cols


class _DampedStageShape:
    """One column of a shared :class:`DampedQuadCompliance` as a ``.eval(freq)`` stage shape
    (duck-types ``TFModel``), scaled by a common ``gain``. The absolute scale is irrelevant to the
    closed-loop FRFs (``D = G/(A·C)`` is derived); a single shared ``gain`` across the stages keeps
    numbers O(1) while preserving the relative compliance magnitudes that carry the stage hierarchy.
    """

    def __init__(self, damped: DampedQuadCompliance, col_idx: int, gain: float = 1.0):
        self._damped = damped
        self._j = int(col_idx)
        self.gain = float(gain)

    def eval(self, freq) -> np.ndarray:
        return self.gain * self._damped.eval(freq)[:, self._j]


def hierarchical_stage_shapes(plant: ReducedStateSpacePlant | None = None,
                              f_anchor: float = 100.0) -> dict:
    """The three M0-damped DARM-stage shapes ``{M0, PUM, TST}`` for the hierarchical loop.

    Each is a column of the M0-damped reduced QUAD (``{M0,L2,L3}.drive → L3.disp.L``), so the
    stages carry the true multi-resonance mechanics AND their natural relative strengths — the
    offload runs in force units, exactly as the twin's does, so NO separate per-stage authority
    weighting is applied (that would double-count what the compliances already encode). A single
    common gain normalises |TST| to 1 at ``f_anchor`` purely to keep magnitudes O(1)."""
    damped = DampedQuadCompliance(plant)
    f0 = np.array([float(f_anchor)])
    c_ref = abs(damped.eval(f0)[0, DampedQuadCompliance.STAGES.index("TST")])
    gain = 1.0 / c_ref
    return {s: _DampedStageShape(damped, i, gain)
            for i, s in enumerate(DampedQuadCompliance.STAGES)}
