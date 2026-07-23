"""OSEM <-> Euler-DOF projection for the 40m SOS.

The SOS is actuated and sensed by five OSEM coil-magnet pairs: four on the optic face
(UL, UR, LR, LL) and one on the side (SIDE). The sysID pipeline identifies in the Euler
basis (L T V R P Y) — see :mod:`system_ident.sos_plant` — so this module is the thin,
hardware-facing adapter between the two.

Rank, stated up front
---------------------
Five OSEMs **cannot** span six DOF, and the deficiency is worse than 5-vs-6:

* The four face coils span only **three** DOF (L, P, Y). Their fourth independent
  combination, ``(+1, -1, +1, -1)`` in ``COIL_ORDER``, is a "butterfly" pattern that applies
  no net force or torque to a rigid optic — it lies in the null space.
* SIDE adds **T**.
* **V (vertical) and R (roll) are not actuated by the OSEMs at all.**

So the OSEM basis reaches 4 of 6 DOF: ``OSEM_DOFS = (L, T, P, Y)``. :func:`osem_to_euler`
therefore returns a pseudo-inverse that is exact on that 4-D subspace and *zero* on V/R —
it does not silently invent V/R content. Callers wanting a 6-DOF identification must drive
V and R some other way (or accept that those modes are unobservable through OSEMs).

Provenance and a real trap
--------------------------
Corroborated against the CDS matrices recovered in ``digital_twin/simplant-salvage/matrices/``
(from ``simplant`` commit ``d158001^``):

* ``X1:SUS-ITMXP_COIL_IN_C2DOF_default.txt`` — coil -> DOF, columns ordered
  ``(UL, UR, LR, LL, SIDE)``. Its PIT row ``(1, 1, -1, -1)`` and YAW row ``(1, -1, -1, 1)``
  agree with the geometry encoded here.
* ``X1:SUS-ITMXC_TO_COIL_default.txt`` — DOF -> coil, but its **rows are ordered
  ``(UL, UR, LL, LR, SIDE)``**, i.e. the last two face coils are swapped relative to the
  C2DOF file.

**Composing those two files as-shipped, assuming a common coil order, sends YAW to exactly
zero** (verified numerically: ``C2DOF @ TO_COIL`` has an all-zero YAW row, while reordering
TO_COIL's rows yields a clean ``diag(4, 4, 4, 16)``). Anything that consumes the site's
matrices must pin the coil ordering explicitly rather than trusting file order. This module
derives its matrices from geometry for exactly that reason.
"""
from __future__ import annotations

import numpy as np

from .sos_plant import DOFS

#: Coil ordering used by every matrix in this module. Pin it; do not infer it from a file.
COIL_ORDER: tuple[str, ...] = ("UL", "UR", "LR", "LL", "SIDE")

#: The Euler DOF the OSEM basis can actually reach.
OSEM_DOFS: tuple[str, ...] = ("L", "T", "P", "Y")

#: DOF with no OSEM actuation path.
UNACTUATED_DOFS: tuple[str, ...] = ("V", "R")

# Per-DOF coil patterns, in COIL_ORDER. Geometry:
#   L (POS)  — all four face coils push together
#   P (PIT)  — upper (UL, UR) against lower (LR, LL)
#   Y (YAW)  — left (UL, LL) against right (UR, LR)
#   T (SIDE) — the side coil alone
_PATTERNS: dict[str, tuple[float, ...]] = {
    "L": (1.0,  1.0,  1.0,  1.0, 0.0),
    "P": (1.0,  1.0, -1.0, -1.0, 0.0),
    "Y": (1.0, -1.0, -1.0,  1.0, 0.0),
    "T": (0.0,  0.0,  0.0,  0.0, 1.0),
}

#: The face-coil combination that produces no rigid-body force or torque.
BUTTERFLY: tuple[float, ...] = (1.0, -1.0, 1.0, -1.0, 0.0)


def euler_to_osem() -> np.ndarray:
    """``(5, 6)`` matrix mapping Euler DOF drive (L T V R P Y) to the five coil drives.

    The V and R columns are identically zero — the OSEMs cannot drive them.
    """
    M = np.zeros((len(COIL_ORDER), len(DOFS)))
    for dof, pattern in _PATTERNS.items():
        M[:, DOFS.index(dof)] = pattern
    return M


def osem_to_euler() -> np.ndarray:
    """``(6, 5)`` pseudo-inverse mapping coil signals back to Euler DOF.

    Exact on the 4-D span of :data:`OSEM_DOFS`; the V and R rows are identically zero, and
    the butterfly combination maps to zero. Round-trips as
    ``osem_to_euler() @ euler_to_osem() == diag(1, 1, 0, 0, 1, 1)`` in L T V R P Y order.
    """
    return np.linalg.pinv(euler_to_osem())


def actuated_mask() -> np.ndarray:
    """Boolean ``(6,)`` mask over L T V R P Y — True where the OSEMs have authority."""
    return np.array([d in OSEM_DOFS for d in DOFS])


def coil_force_to_dof() -> np.ndarray:
    """``(6, 5)`` map from the five coil forces to the six generalized DOF forces/torques.

    This is the **transpose** of :func:`euler_to_osem`, not its pseudo-inverse. Each coil
    applies a force along its own axis; the generalized force on DOF *j* is the sum over
    coils of that coil's force weighted by the same geometric pattern that determines how
    DOF *j* displaces the coil. Sensing and actuation share the geometry, so they share the
    matrix — one transposed relative to the other. Using ``pinv`` here would silently
    normalize away the factor-of-4 coil gain and misstate the actuator strength.
    """
    return euler_to_osem().T


def project_plant(sys, *, inputs: bool = True, outputs: bool = True):
    """Wrap a 6-DOF Euler plant so its inputs and/or outputs are in the OSEM basis.

    With both flags set the result is 5-in / 5-out: coil forces in, coil readouts out.
    Uses python-control's static-system algebra rather than reaching into A/B/C/D by hand.
    """
    import control

    out = sys
    if inputs:
        # coil forces (5) -> DOF forces (6) -> plant
        out = out * control.ss([], [], [], coil_force_to_dof())
    if outputs:
        # plant -> DOF displacements (6) -> coil readouts (5)
        out = control.ss([], [], [], euler_to_osem()) * out
    return out
