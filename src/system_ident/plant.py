"""Resonant suspension plant: the per-DoF transfer functions the twin drives.

The digital twin (build step 4) integrates these continuous-time ``TFModel``\\ s
to answer ``read`` for an injected drive. A plant is just a named set of
SISO transfer functions (one per degree of freedom) plus a sample rate, built
from a resonance spec — the same ``(f0, Q)`` -> conjugate-pole-pair construction
the ``double_pend_demo`` uses.

The full ``Plant_Model`` state-space model (control + slycot) is a separate,
heavier path; this resonant form is what the loop and twin need and what the
demo validates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .model import TFModel


@dataclass
class SuspensionPlant:
    """A named collection of per-DoF transfer functions plus a sample rate."""

    transfer_functions: dict[str, TFModel]
    fs: float

    @property
    def dofs(self) -> list[str]:
        return list(self.transfer_functions)

    def __getitem__(self, dof: str) -> TFModel:
        return self.transfer_functions[dof]

    @classmethod
    def from_resonance_spec(
        cls,
        spec: dict[str, dict],
        fs: float,
    ) -> "SuspensionPlant":
        """Build from ``{dof: {"resonances": [(f0, Q), ...], "gain": k}}``."""
        tfs = {
            dof: TFModel.from_resonances(
                d["resonances"], d["gain"], zeros=d.get("zeros", ())
            )
            for dof, d in spec.items()
        }
        return cls(transfer_functions=tfs, fs=fs)


def double_pendulum(
    resonances: Sequence[tuple[float, float]] = ((0.6, 20.0), (1.5, 30.0)),
    gain: float = 300.0,
    fs: float = 32.0,
) -> TFModel:
    """The ``double_pend_demo`` plant: an L2-torque -> L3-angle double pendulum.

    Two resonances at 0.6 Hz (Q=20) and 1.5 Hz (Q=30) with gain 300 and no
    zeros. Returned as a single :class:`TFModel`; this is the canonical
    validation fixture for the Fisher / excitation machinery.
    """
    return TFModel.from_resonances(resonances, gain)


# Representative Advanced-LIGO quadruple-suspension rigid-body mode frequencies
# (Hz). Longitudinal and pitch sets from the aLIGO quad design + FEM/measured
# identification (Aston et al., Class. Quantum Grav. 29 (2012) 235004,
# DCC P1200056; Sauter et al., Phys. Rev. D 109 (2024) 064033, arXiv:2306.13755).
ALIGO_LONG_MODES_HZ = (0.43, 1.00, 2.01, 3.42)
ALIGO_PITCH_MODES_HZ = (0.56, 1.31, 2.81)


def coupled_suspension(
    long_modes: Sequence[tuple[float, float]],
    pitch_modes: Sequence[tuple[float, float]],
    *,
    coupling: float = 0.18,
    gain: float = 100.0,
) -> dict[tuple[str, str], TFModel]:
    """A 2×2 longitudinal/pitch suspension TF matrix from a coupled modal expansion.

    Unlike a "scaled-diagonal" toy, this is a genuine coupled system: every matrix
    element shares the same normal-mode poles, so

    * the **diagonals** ``H[POS←POS]``, ``H[PIT←PIT]`` show an *anti-resonance
      notch* at the partner DoF's modes (same-sign modal residues), and
    * the **off-diagonals** ``H[PIT←POS] = H[POS←PIT]`` are notch-free and set by
      the cross-coupling — *not* a rescaled copy of a diagonal.

    Each normal mode ``k`` carries a mode shape ``(φ_L, φ_P)``: a *longitudinal*
    mode moves mostly in L (``φ = (1, c_k)``), a *pitch* mode mostly in P
    (``φ = (c_k, 1)``), with the small cross term ``c_k = coupling·(−1)^k`` giving
    the off-diagonal its own sign-alternating zero structure. The element TFs are
    the modal sum ``H_ij(s) = gain · Σ_k φ_i,k φ_j,k / (s² + (ω_k/Q_k)s + ω_k²)``.

    Parameters
    ----------
    long_modes, pitch_modes:
        ``(f0 [Hz], Q)`` pairs for the longitudinal and pitch normal modes.
        Use :data:`ALIGO_LONG_MODES_HZ` / :data:`ALIGO_PITCH_MODES_HZ` for
        representative Advanced-LIGO frequencies.
    coupling:
        L↔P cross-participation (0 → fully diagonal; ~0.1–0.3 is realistic).
    gain:
        Overall scale.

    Returns
    -------
    dict keyed by ``(out_dof, in_dof)`` over ``POS``/``PIT`` — the four element
    :class:`TFModel`\\ s, ready for the ``coupling=`` argument of the twin.

    Notes
    -----
    This is a *simplified, physically-motivated* model (lumped normal modes with a
    single L↔P coupling), not a validated full aLIGO suspension state-space model;
    the frequencies are representative, not a specific instrument's calibration.
    """
    modes = []  # (ω0, a=ω0/Q, (φ_L, φ_P))
    for k, (f, q) in enumerate(long_modes):
        w = 2.0 * np.pi * f
        modes.append((w, w / q, (1.0, coupling * (-1) ** k)))
    for k, (f, q) in enumerate(pitch_modes):
        w = 2.0 * np.pi * f
        modes.append((w, w / q, (coupling * (-1) ** k, 1.0)))

    dens = [np.array([1.0, a, w * w]) for (w, a, _phi) in modes]
    den = np.array([1.0])
    for d in dens:
        den = np.polymul(den, d)

    def element(i: int, j: int) -> TFModel:
        num = np.zeros(1)
        for k, (_w, _a, phi) in enumerate(modes):
            term = np.array([gain * phi[i] * phi[j]])
            for m, dm in enumerate(dens):
                if m != k:
                    term = np.polymul(term, dm)
            num = np.polyadd(num, term)
        return TFModel(num=num, den=den)

    idx = {"POS": 0, "PIT": 1}
    keys = [("POS", "POS"), ("PIT", "PIT"), ("PIT", "POS"), ("POS", "PIT")]
    return {key: element(idx[key[0]], idx[key[1]]) for key in keys}
