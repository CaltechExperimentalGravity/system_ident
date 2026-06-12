"""The ``Estimator`` strategy interface.

An estimator takes a measured frequency response (with per-point uncertainty)
and a prior model, and returns an updated :class:`~system_ident.model.TFModel`.
Swapping estimators must not require touching the sysID loop, so the loop only
ever talks to this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..model import TFModel


class Estimator(ABC):
    """Fit / update a transfer-function model from measured data."""

    @abstractmethod
    def fit(
        self,
        freq: np.ndarray,
        H_meas: np.ndarray,
        H_err: np.ndarray,
        model: TFModel,
    ) -> TFModel:
        """Return an updated model given measured ``H_meas`` ± ``H_err``.

        Parameters
        ----------
        freq:
            Frequency grid [Hz].
        H_meas:
            Complex measured transfer function at ``freq``.
        H_err:
            Per-point (real, positive) uncertainty on ``H_meas`` used to weight
            the fit.
        model:
            Prior model (sets the order and the starting point).
        """
        raise NotImplementedError
