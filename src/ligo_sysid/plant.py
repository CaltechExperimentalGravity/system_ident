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
