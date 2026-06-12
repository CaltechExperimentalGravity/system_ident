"""The ``InputDesigner`` strategy interface.

An input designer computes the optimal excitation power spectrum for the next
iteration, given the current model and the measurement-noise floor, subject to
a total drive-power budget. The loop only talks to this interface, so the
optimal-excitation method can be swapped from config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..model import TFModel


class InputDesigner(ABC):
    """Design the excitation PSD for the next measurement."""

    @abstractmethod
    def design(
        self,
        freq: np.ndarray,
        model: TFModel,
        Pyy: np.ndarray,
        Px_tot: float,
        n_iter: int = 3,
    ) -> np.ndarray:
        """Return the optimal input PSD ``Pxx`` over ``freq``.

        Parameters
        ----------
        freq:
            Frequency grid [Hz].
        model:
            Current best model (sets where information is sensitive).
        Pyy:
            Output/measurement-noise PSD over ``freq``.
        Px_tot:
            Total input-power budget (the constraint to distribute).
        n_iter:
            Number of dispersion-function refinement iterations.
        """
        raise NotImplementedError
