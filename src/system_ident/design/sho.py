"""Specialized input designer for low-DOF / SHO-structured systems.

Wraps ``SHO/SHO_OptimalInput_Functions.py`` (sympy Jacobian + scipy.optimize).
Drop-in alternative; build step 10.
"""

from __future__ import annotations

import numpy as np

from ..model import TFModel
from .base import InputDesigner


class SHODesigner(InputDesigner):
    """Fisher-optimal excitation via the SHO sympy/scipy machinery."""

    def design(
        self,
        freq: np.ndarray,
        model: TFModel,
        Pyy: np.ndarray,
        Px_tot: float,
        n_iter: int = 3,
    ) -> np.ndarray:
        raise NotImplementedError("SHODesigner.design lands in build step 10")
