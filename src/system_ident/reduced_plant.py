"""Reduced-order suspension plant: a modal-truncation state-space (A,B,C,D) with
labelled physical channels and an exact mode table. Pure numpy/scipy — no twin, no
`control`/`slycot`. The FRF G(f) = C(2πif·I − A)^-1 B + D is the sysID target; the
eigen-modes are the oracle. See src/system_ident/models/ and the regen script there.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

_MODELS = Path(__file__).resolve().parent / "models"


@dataclass
class ReducedStateSpacePlant:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    inputs: list[str]
    outputs: list[str]
    _modes: list[tuple[float, float]]
    f_mode_cut: float

    @classmethod
    def load(cls, name: str) -> "ReducedStateSpacePlant":
        npz = np.load(_MODELS / f"{name}_reduced_50hz.npz")
        side = json.loads((_MODELS / f"{name}_reduced_50hz.json").read_text())
        return cls(
            A=npz["A"], B=npz["B"], C=npz["C"], D=npz["D"],
            inputs=list(side["inputs"]), outputs=list(side["outputs"]),
            _modes=[(float(f), float(q)) for f, q in side["modes"]],
            f_mode_cut=float(npz["f_mode_cut"]),
        )

    def eval(self, freq) -> np.ndarray:
        """FRF tensor (F, n_out, n_in): G(f) = C (2πif I − A)^-1 B + D."""
        freq = np.asarray(freq, float)
        I = np.eye(self.A.shape[0])
        # solve (sI − A) X = B per frequency, then G = C X + D
        return np.array([self.C @ np.linalg.solve(2j * np.pi * f * I - self.A, self.B) + self.D
                         for f in freq])

    def modes(self) -> list[tuple[float, float]]:
        return list(self._modes)

    def subplant(self, sensors: list[str], actuators: list[str]) -> "ReducedStateSpacePlant":
        oi = [self.outputs.index(s) for s in sensors]
        ii = [self.inputs.index(a) for a in actuators]
        return ReducedStateSpacePlant(
            A=self.A, B=self.B[:, ii], C=self.C[oi, :], D=self.D[np.ix_(oi, ii)],
            inputs=list(actuators), outputs=list(sensors),
            _modes=list(self._modes), f_mode_cut=self.f_mode_cut,
        )
